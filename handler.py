"""RunPod serverless handler: audio separation (demucs htdemucs, vocals / no_vocals) and forced alignment of known text
(ctc-forced-aligner, MMS). Generic: the job carries the audio (URL or base64), the text, and where to put results.

input = {"audio_url": "https://…", | "audio_base64": "…",            # any format ffmpeg reads
         "tasks": ["separate", "align"],
         "separate": {"model": "htdemucs", "shifts": 1, "overlap": 0.25, "format": "flac"},
         "align": {"language": "eng", "lines": ["…", "…"], "audio": "vocals|input"},   # words from the separated vocals by default
         "upload": {"url": "https://…", "token": "…"},                  # results POSTed here (multipart); else returned inline if small
         "return_inline_max_mb": 18}
output = {"separate": {"vocals": {"url"|"base64", "sha256", "bytes"}, "no_vocals": {...}, "sec": 12.3},
          "align": {"frame_s": 0.02, "words": [{"w", "t0", "t1", "score"}], "lines": [[i0, i1], …], "sec": 4.1},
          "timing": {"download_s", "total_s"}, "device": "cuda"}
"""
import base64, hashlib, io, json, os, subprocess, tempfile, time
import numpy as np
import requests
import runpod
import soundfile as sf
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_demucs = {}
_aligner = {}


def _audio(job_input, d):
    t = time.time()
    src = os.path.join(d, "input")
    if job_input.get("audio_url"):
        with requests.get(job_input["audio_url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(src, "wb") as f:
                for ch in r.iter_content(1 << 20):
                    f.write(ch)
    elif job_input.get("audio_base64"):
        with open(src, "wb") as f:
            f.write(base64.b64decode(job_input["audio_base64"]))
    else:
        raise ValueError("audio_url or audio_base64 required")
    wav = os.path.join(d, "input.wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ar", "44100", "-ac", "2", wav], check=True)
    return wav, round(time.time() - t, 2)


def _separate(wav, d, opts):
    import demucs.apply, demucs.pretrained
    from demucs.audio import save_audio
    name = opts.get("model", "htdemucs")
    if name not in _demucs:
        _demucs[name] = demucs.pretrained.get_model(name).to(DEV)
    model = _demucs[name]
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    mix = torch.from_numpy(x.T).unsqueeze(0).to(DEV)
    ref = mix.mean(0); mix = (mix - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = demucs.apply.apply_model(model, mix, shifts=int(opts.get("shifts", 1)), overlap=float(opts.get("overlap", 0.25)),
                                       split=True, device=DEV, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    stems = dict(zip(model.sources, out))
    vocals = stems["vocals"]
    no_vocals = sum(v for k, v in stems.items() if k != "vocals")
    fmt = opts.get("format", "flac")
    paths = {}
    for k, y in (("vocals", vocals), ("no_vocals", no_vocals)):
        p = os.path.join(d, f"{k}.{fmt}")
        sf.write(p, y.cpu().numpy().T, sr, format=fmt.upper(), subtype="PCM_24" if fmt == "flac" else None)
        paths[k] = p
    return paths, sr


def _align(wav_path, opts):
    import ctc_forced_aligner as cfa
    import librosa
    if "a" not in _aligner:
        _aligner["a"] = cfa.AlignmentSingleton()
    a = _aligner["a"]
    y, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    x = librosa.resample(y.mean(1), orig_sr=sr, target_sr=16000).astype(np.float32)
    em, _ = cfa.generate_emissions(a.model, x, batch_size=4)
    shag = len(x) / 16000 / em.shape[0]
    lines = [l for l in opts.get("lines", []) if l and l.strip()]
    tok, txt = cfa.preprocess_text(" ".join(lines), romanize=True, language=opts.get("language", "eng"), split_size="word")
    seg, sc, blank = cfa.get_alignments(em, tok, a.tokenizer)
    spans = cfa.get_spans(tok, seg, blank)
    words = []
    for i, t in enumerate(txt):
        if t == "<star>":
            continue
        s0, s1 = spans[i][0].start, spans[i][-1].end
        words.append({"w": t, "t0": round(s0 * shag, 3), "t1": round(s1 * shag, 3),
                      "score": round(float(np.exp(float(sc[s0:s1].sum() if s1 > s0 else sc[s0]) / max(1, s1 - s0 + 1))), 3)})
    per_line = [len(cfa.preprocess_text(l, romanize=True, language=opts.get("language", "eng"), split_size="word")[1]) // 2 for l in lines]
    idx, i = [], 0
    for n in per_line:
        idx.append([i, i + n]); i += n
    return {"frame_s": round(shag, 6), "words": words, "lines": idx, "count_matches": i == len(words)}


def _deliver(paths, job_input, max_mb):
    out = {}
    up = job_input.get("upload") or {}
    for k, p in paths.items():
        data = open(p, "rb").read()
        rec = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if up.get("url"):                               # raw body, one request per file: Authorization: Bearer <token>, X-File: <name>
            r = requests.post(up["url"], data=data, timeout=300,
                              headers={"Authorization": f"Bearer {up.get('token', '')}", "X-File": os.path.basename(p),
                                       "Content-Type": "application/octet-stream"})
            r.raise_for_status()
            rec["url"] = (r.json() or {}).get("url") if r.headers.get("content-type", "").startswith("application/json") else up["url"]
        elif len(data) <= max_mb * (1 << 20):
            rec["base64"] = base64.b64encode(data).decode()
        else:
            rec["error"] = f"result {len(data) >> 20} MB exceeds inline limit and no upload url given"
        out[k] = rec
    return out


def handler(job):
    t0 = time.time()
    inp = job.get("input") or {}
    tasks = inp.get("tasks") or ["separate", "align"]
    res = {"device": DEV, "timing": {}}
    with tempfile.TemporaryDirectory() as d:
        wav, res["timing"]["download_s"] = _audio(inp, d)
        paths = {}
        if "separate" in tasks:
            t = time.time()
            paths, sr = _separate(wav, d, inp.get("separate") or {})
            res["separate"] = _deliver(paths, inp, float(inp.get("return_inline_max_mb", 18)))
            res["separate"]["sec"] = round(time.time() - t, 2)
        if "align" in tasks:
            t = time.time()
            a = inp.get("align") or {}
            src = paths.get("vocals") if a.get("audio", "vocals") == "vocals" and paths.get("vocals") else wav
            res["align"] = _align(src, a)
            res["align"]["sec"] = round(time.time() - t, 2)
    res["timing"]["total_s"] = round(time.time() - t0, 2)
    return res


runpod.serverless.start({"handler": handler})

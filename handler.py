"""RunPod serverless handler: audio separation (demucs htdemucs, vocals / no_vocals) and forced alignment of known text
(ctc-forced-aligner, MMS). Generic: the job carries the audio (URL or base64), the text, and where to put results.

input = {"audio_url": "https://…", | "audio_base64": "…",            # any format ffmpeg reads
         "crop": {"start_s": 0, "end_s": 30},                         # optional: process only this window of the audio
         "tasks": ["separate", "align"],
         "separate": {"model": "htdemucs", "shifts": 1, "overlap": 0.25, "format": "flac"},
         "align": {"language": "eng", "lines": ["…", "…"], "audio": "vocals|input", "batch_size": 16},
         "upload": {"url": "https://…", "token": "…", "prefix": "part01_"},   # results POSTed here (raw body); else inline if small
         "return_inline_max_mb": 18}
output = {"separate": {"vocals": {"url"|"base64", "sha256", "bytes"}, "no_vocals": {...}, "sec": 12.3},
          "align": {"frame_s": 0.02, "words": [{"w", "t0", "t1", "score"}], "lines": [[i0, i1], …], "sec": 4.1,
                    "emissions_s": 2.0, "providers": ["CUDAExecutionProvider", …]},
          "timing": {"download_s", "total_s"}, "device": "cuda", "gpu": "NVIDIA …", "startup_s": 9.1}
Models are loaded once at worker start (before the first job), so a warm worker answers without loading anything.
"""
import base64, hashlib, io, json, os, subprocess, tempfile, time
_T_START = time.time()
import numpy as np
import requests
import runpod
import soundfile as sf
import torch                                   # imported before onnxruntime: loads the CUDA libraries the GPU session uses
import onnxruntime as ort

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU = torch.cuda.get_device_name(0) if DEV == "cuda" else None
# ctc-forced-aligner creates its ONNX session without providers, and onnxruntime then runs on the CPU even on a GPU worker
# (measured: 58 s of alignment for a 4-min song). Every session created in this process asks for CUDA first.
PROVIDERS = (["CUDAExecutionProvider", "CPUExecutionProvider"] if DEV == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
             else ["CPUExecutionProvider"])
if DEV == "cuda" and hasattr(ort, "preload_dlls"):    # onnxruntime ≥ 1.21: load CUDA/cuDNN from the installed packages
    try:
        ort.preload_dlls()
    except Exception:
        pass
_ort_session = ort.InferenceSession


def _session(path, *args, **kwargs):
    kwargs.setdefault("providers", PROVIDERS)
    return _ort_session(path, *args, **kwargs)


ort.InferenceSession = _session
_demucs = {}
_aligner = {}


def _load_demucs(name):
    import demucs.pretrained
    if name not in _demucs:
        _demucs[name] = demucs.pretrained.get_model(name).to(DEV).eval()
    return _demucs[name]


def _load_aligner():
    import ctc_forced_aligner as cfa
    if "a" not in _aligner:
        _aligner["a"] = cfa.AlignmentSingleton()
    return _aligner["a"]


_load_demucs("htdemucs")                       # warm start: models in memory before the first job
ALIGN_PROVIDERS = _load_aligner().model.get_providers()
STARTUP_S = round(time.time() - _T_START, 2)


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
    crop = job_input.get("crop") or {}
    okno = []
    if crop.get("start_s"):
        okno += ["-ss", str(float(crop["start_s"]))]
    if crop.get("end_s"):
        okno += ["-to", str(float(crop["end_s"]))]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, *okno, "-ar", "44100", "-ac", "2", wav], check=True)
    return wav, round(time.time() - t, 2)


def _separate(wav, d, opts):
    import demucs.apply
    model = _load_demucs(opts.get("model", "htdemucs"))
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


def _words(cfa, tokenizer, em, step_s, text, lang, frame0=0):
    """Words of the text over the emission frames em; time = (frame0 + frame) × step_s."""
    tok, txt = cfa.preprocess_text(text, romanize=True, language=lang, split_size="word")
    seg, sc, blank = cfa.get_alignments(em, tok, tokenizer)
    spans = cfa.get_spans(tok, seg, blank)
    out = []
    for i, t_ in enumerate(txt):
        if t_ == "<star>":
            continue
        s0, s1 = spans[i][0].start, spans[i][-1].end
        score = float(sc[s0:s1].sum()) if s1 > s0 else float(sc[s0])
        out.append({"w": t_, "t0": round((frame0 + s0) * step_s, 3), "t1": round((frame0 + s1) * step_s, 3),
                    "score": round(score, 2), "uverennost": round(float(np.exp(score / max(1, s1 - s0 + 1))), 3)})
    return out


def _lead(line):
    """Line → (lead text, backing text in parentheses, backing only?). Backing vocals are sung over the lead and break the order."""
    import re
    backing = " ".join(x.strip() for x in re.findall(r"\(([^)]*)\)", line) if x.strip())
    lead = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", line)).strip()
    if not re.findall(r"\w+", lead):
        return backing, "", True
    return lead, backing, False


def _align(wav_path, opts):
    """Two passes, like a careful human: (1) the whole lead text in one go; (2) if approximate line start times are given
    ("times", e.g. from synced lyrics), each line again inside its own window [time + global offset − before_s,
    next time + offset + after_s]. Both results are returned; the caller picks per line (windowed wins on long breaks and
    repeated sections, where a single pass can drift by a whole section)."""
    import ctc_forced_aligner as cfa
    import librosa
    a = _load_aligner()
    lang = opts.get("language", "eng")
    y, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    x = librosa.resample(y.mean(1), orig_sr=sr, target_sr=16000).astype(np.float32)
    t = time.time()
    em, _ = cfa.generate_emissions(a.model, x, batch_size=int(opts.get("batch_size", 16 if "CUDAExecutionProvider" in ALIGN_PROVIDERS else 4)))
    emissions_s = round(time.time() - t, 2)
    step = len(x) / 16000 / em.shape[0]
    lines = list(opts.get("lines", []))
    times = list(opts.get("times") or [None] * len(lines))
    win = {"before_s": 1.5, "after_s": 0.3, "last_s": 8.0, **(opts.get("window") or {})}
    parsed = [_lead(l) for l in lines]
    lead = [p[0] if not p[2] else "" for p in parsed]
    all_words = _words(cfa, a.tokenizer, em, step, " ".join(l for l in lead if l), lang)
    per_line = [len(cfa.preprocess_text(l, romanize=True, language=lang, split_size="word")[1]) // 2 if l else 0 for l in lead]
    by_line, i = [], 0
    for n in per_line:
        by_line.append(all_words[i:i + n]); i += n
    count_ok = i == len(all_words)
    if not count_ok:
        by_line = None
    diffs = [w[0]["t0"] - tt for w, tt in zip(by_line or [], times) if w and tt is not None]
    offset = float(np.median(diffs)) if diffs else 0.0
    windowed, dur = [], len(x) / 16000
    for k, (line, tt) in enumerate(zip(lines, times)):
        if tt is None:
            windowed.append({"n": k + 1, "slova": by_line[k] if by_line else [], "okno": None})
            continue
        text = parsed[k][0]
        nxt = next((v for v in times[k + 1:] if v is not None), None)
        lo = max(0.0, tt + offset - win["before_s"])
        hi = min(dur, (nxt + offset + win["after_s"]) if nxt is not None else tt + offset + win["last_s"])
        hi = max(hi, min(dur, lo + 1.0))
        k0, k1 = int(lo / step), int(hi / step)
        try:
            windowed.append({"n": k + 1, "slova": _words(cfa, a.tokenizer, np.ascontiguousarray(em[k0:k1]), step, text, lang, frame0=k0),
                             "okno": [round(lo, 3), round(hi, 3)]})
        except Exception as e:                       # window shorter than the text etc.: line without words, flagged downstream
            windowed.append({"n": k + 1, "slova": [], "okno": [round(lo, 3), round(hi, 3)], "oshibka": repr(e)[:160]})
    # field names follow the common alignment-file layout used downstream (slova = all words in order, vse_po_strokam = the
    # single pass split by line, po_strokam = the windowed pass)
    return {"pribor": "ctc-forced-aligner (MMS)", "yazyk": lang, "shag_kadra_s": round(step, 6),
            "sdvig_zapisi_ot_lrclib_s": round(offset, 3) if diffs else None, "slov": len(all_words), "schet_soshelsya": count_ok,
            "slova": all_words, "vse_po_strokam": by_line, "po_strokam": windowed,
            "emissions_s": emissions_s, "providers": ALIGN_PROVIDERS}


def _deliver(paths, job_input, max_mb):
    out = {}
    up = job_input.get("upload") or {}
    for k, p in paths.items():
        data = open(p, "rb").read()
        rec = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if up.get("url"):                               # raw body, one request per file: Authorization: Bearer <token>, X-File: <name>
            r = requests.post(up["url"], data=data, timeout=300,
                              headers={"Authorization": f"Bearer {up.get('token', '')}",
                                       "X-File": up.get("prefix", "") + os.path.basename(p),
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
    res = {"device": DEV, "gpu": GPU, "startup_s": STARTUP_S, "timing": {}}
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

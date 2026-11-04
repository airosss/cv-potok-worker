"""RunPod serverless handler: audio separation (demucs htdemucs, vocals / no_vocals) and forced alignment of known text
(ctc-forced-aligner, MMS). Generic: the job carries the audio (URL or base64), the text, and where to put results.

input = {"audio_url": "https://…", | "audio_base64": "…",            # any format ffmpeg reads ("audio_bytes": size, optional)
         "crop": {"start_s": 0, "end_s": 30},                         # optional: process only this window of the audio
         "tasks": ["separate", "align"],
         "separate": {"model": "htdemucs", "shifts": 1, "overlap": 0.25, "format": "flac", "subtype": "PCM_24|PCM_16", "seed": 0},
         "align": {"language": "eng", "lines": ["…", "…"], "audio": "vocals|input", "batch_size": 16},
         "upload": {"url": "https://…", "token": "…", "prefix": "part01_"},   # results POSTed here (raw body); else inline if small
         "return_inline_max_mb": 18}
output = {"separate": {"vocals": {"url"|"base64", "sha256", "bytes"}, "no_vocals": {...}, "sec": 12.3,
                       "demucs_s", "write_s", "upload_s", "subtype", "clipped": {stem: samples clipped to the integer range}},
          "align": {"frame_s": 0.02, "words": [{"w", "t0", "t1", "score"}], "lines": [[i0, i1], …], "sec": 4.1,
                    "emissions_s": 2.0, "providers": ["CUDAExecutionProvider", …]},
          "timing": {"download_s", "fetch_s", "streams", "input_mb", "upload_wait_s", "total_s"}, "device": "cuda", "gpu": "NVIDIA …",
          "startup_s": 9.1, "worker": {"job_n": 3, "ready_for_s": 120.5}}      # job_n 1 + small ready_for_s = a cold worker
Models are loaded once at worker start (before the first job), so a warm worker answers without loading anything.
Separation results are uploaded in the background while the GPU aligns words and measures pitch.
"""
import base64, hashlib, io, json, os, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
_T_START = time.time()
import numpy as np
import requests
import runpod
import soundfile as sf
import torch                                   # imported before onnxruntime: loads the CUDA libraries the GPU session uses
import onnxruntime as ort
from fastaudio import resample16 as _resample16

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


ROFORMER = {"bs_roformer_1297": ("/models/roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                                 "/models/roformer/model_bs_roformer_ep_317_sdr_12.9755.yaml"),
            # lead vs backing vocals (Mel-Band RoFormer karaoke, aufr33 & viperx, SDR 10.2): on a vocal stem "vocals" = lead,
            # "no_vocals" = everything else = backing vocals; loaded on demand only
            "mel_karaoke_aufr33": ("/models/roformer/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
                                   "/models/roformer/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml")}
_roformer = {}


def _load_roformer(name):
    """BS-RoFormer vocals model (viperx ep 317, public UVR weights checked by sha256 at build): (model, config)."""
    if name not in _roformer:
        import sys
        sys.path.insert(0, "/models/roformer")
        from roformer_arch.load import build
        sys.path.pop(0)
        m, cfg = build(*ROFORMER[name])
        _roformer[name] = (m.to(DEV), cfg)
    return _roformer[name]


def _demix_roformer(model, cfg, mix, overlap, batch):
    """Chunked inference as in ZFTurbo/Music-Source-Separation-Training (demix_track): chunks of C samples every C/overlap,
    reflect padding of C − step at both ends, linear 10 % fades, mixed precision as in training. The window is copied per
    chunk (the reference reused one array, so the first chunk's "no fade-in" leaked into every later chunk).
    mix: (channels, samples) float32 on the CPU → estimate of the target stem, same shape."""
    import torch.nn.functional as F
    C = int(cfg["audio"]["chunk_size"])
    step = C // overlap
    fade, border = C // 10, C - step
    L = mix.shape[-1]
    padded = L > 2 * border and border > 0
    if padded:
        mix = F.pad(mix[None], (border, border), mode="reflect")[0]
    total = mix.shape[-1]
    base = torch.ones(C)
    base[:fade] = torch.linspace(0, 1, fade)
    base[-fade:] = torch.linspace(1, 0, fade)
    result = torch.zeros_like(mix)
    counter = torch.zeros_like(mix)
    starts = list(range(0, total, step))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
        for b0 in range(0, len(starts), batch):
            parts, locs = [], []
            for i in starts[b0:b0 + batch]:
                part = mix[:, i:i + C]
                n = part.shape[-1]
                if n < C:
                    part = F.pad(part[None], (0, C - n), mode="reflect")[0] if n > C // 2 + 1 else F.pad(part, (0, C - n))
                parts.append(part)
                locs.append((i, n))
            out = model(torch.stack(parts).to(DEV)).float().cpu()
            for (i, n), y in zip(locs, out):
                w = base.clone()
                if i == 0:
                    w[:fade] = 1
                if i + step >= total:
                    w[-fade:] = 1
                result[:, i:i + n] += y[..., :n] * w[:n]
                counter[:, i:i + n] += w[:n]
    est = torch.nan_to_num(result / counter)
    return est[..., border:-border] if padded else est


_rmvpe = {}


def _load_rmvpe():
    """RMVPE pitch network (RVC weights rmvpe.pt, 128 mels, 4 blocks, kernel (2,2), 1-layer BiGRU)."""
    if "m" in _rmvpe:
        return _rmvpe["m"]
    import sys
    from torch import nn
    sys.path.insert(0, "/models/rmvpe")
    from rmvpe_arch.deepunet import DeepUnet0
    from rmvpe_arch.seq import BiGRU
    sys.path.pop(0)

    class E2E(nn.Module):
        def __init__(s):
            super().__init__()
            s.unet = DeepUnet0((2, 2), 4, 5, 4, 1, 16)
            del s.unet.tf                 # branch unused in the forward pass, absent from the weights
            s.cnn = nn.Conv2d(16, 3, (3, 3), padding=(1, 1))
            s.fc = nn.Sequential(BiGRU(3 * 128, 256, 1), nn.Linear(512, 360), nn.Dropout(0.25), nn.Sigmoid())

        def forward(s, mel):
            x = mel.transpose(-1, -2).unsqueeze(1)
            x = s.cnn(s.unet(x)).transpose(1, 2).flatten(-2)
            return s.fc(x)

    m = E2E()
    m.load_state_dict(torch.load("/models/rmvpe/rmvpe.pt", map_location="cpu", weights_only=True), strict=True)
    _rmvpe["m"] = m.eval().to(DEV)
    return _rmvpe["m"]


MEL_PATH = "/models/rmvpe/mel_16k_1024_128_30_8000_htk.npy"   # librosa.filters.mel(...) computed at image build


def _mel_basis():
    if "mel" not in _rmvpe:
        if os.path.exists(MEL_PATH):
            _rmvpe["mel"] = np.load(MEL_PATH)
        else:                                                   # same numbers, slower first call
            import librosa
            _rmvpe["mel"] = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    return _rmvpe["mel"]


PRELOAD = [m.strip() for m in os.environ.get("PRELOAD", "htdemucs").split(",") if m.strip()]
for _m in PRELOAD:                              # warm start: models in memory before the first job, the rest on first use
    _load_roformer(_m) if _m in ROFORMER else _load_demucs(_m)
ALIGN_PROVIDERS = _load_aligner().model.get_providers()
_load_rmvpe()
LOAD_S = round(time.time() - _T_START, 2)


def _warm_up():
    """One tiny pass through every model before the first job: the first GPU run pays kernel selection / session init
    (measured: the first alignment of a fresh worker took ~16 s, the next ones ~2 s)."""
    import ctc_forced_aligner as cfa
    import demucs.apply
    t = time.time()
    x = np.random.default_rng(0).standard_normal(16000 * 4).astype(np.float32) * 0.01
    for m in PRELOAD:
        if m in ROFORMER:
            model, cfg = _load_roformer(m)
            _demix_roformer(model, cfg, torch.from_numpy(np.stack([x, x])).repeat(1, 3), 2, 1)
        else:
            with torch.no_grad():
                demucs.apply.apply_model(_load_demucs(m), torch.from_numpy(np.stack([x, x]))[None].to(DEV), shifts=1, split=True,
                                         overlap=0.25, device=DEV, progress=False)
    a = _load_aligner()
    em, _ = cfa.generate_emissions(a.model, x, batch_size=4)
    tok, _ = cfa.preprocess_text("warm up", romanize=True, language="eng", split_size="word")
    cfa.get_alignments(em, tok, a.tokenizer)
    _resample16(np.random.default_rng(1).standard_normal(44100).astype(np.float32) * 0.01, 44100)
    mel = torch.from_numpy(_mel_basis()).float().to(DEV)
    with torch.no_grad():
        _load_rmvpe()(torch.log(torch.clamp(mel @ torch.rand(513, 64, device=DEV), min=1e-5))[None])
    return round(time.time() - t, 2)


try:
    WARM_S = _warm_up()
except Exception as e:                          # a failed warm-up must not stop the worker: the first job just pays the init
    WARM_S = f"warm-up failed: {repr(e)[:120]}"
STARTUP_S = round(time.time() - _T_START, 2)


def _fetch(url, path, parts=4, min_mb=8, size_hint=None):
    """Download; a large file in parallel byte ranges (one long-distance TCP stream is limited by its window, several are not).
    Any doubt (no ranges, no length, a short part) → one plain stream. Returns how many streams were used.
    size_hint (bytes, from the caller) below the threshold skips the size question: one round trip less across the world."""
    ses = requests.Session()                            # the plain download reuses the connection of the size question
    size, ranges_ok = 0, False
    if size_hint is None or int(size_hint) >= min_mb << 20:
        try:
            h = ses.head(url, timeout=20, allow_redirects=True)
            size = int(h.headers.get("content-length") or 0)
            ranges_ok = h.ok and h.headers.get("accept-ranges", "").lower() == "bytes"
        except Exception:
            pass
    if ranges_ok and size >= min_mb << 20:
        step = -(-size // parts)
        spans = [(a, min(size, a + step) - 1) for a in range(0, size, step)]

        def part(span):
            r = requests.get(url, headers={"Range": f"bytes={span[0]}-{span[1]}"}, timeout=120)
            if r.status_code != 206 or len(r.content) != span[1] - span[0] + 1:
                raise IOError(f"range {span}: HTTP {r.status_code}, {len(r.content)} bytes")
            return r.content

        try:
            with ThreadPoolExecutor(len(spans)) as ex:
                chunks = list(ex.map(part, spans))
            with open(path, "wb") as f:
                for c in chunks:
                    f.write(c)
            if os.path.getsize(path) == size:
                return len(spans)
        except Exception:
            pass
    with ses.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for ch in r.iter_content(1 << 20):
                f.write(ch)
    return 1


def _audio(job_input, d):
    t = time.time()
    src = os.path.join(d, "input")
    info = {"streams": 1}
    if job_input.get("audio_url"):
        info["streams"] = _fetch(job_input["audio_url"], src, size_hint=job_input.get("audio_bytes"))
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
    info["input_mb"] = round(os.path.getsize(src) / 2 ** 20, 2)
    info["fetch_s"] = round(time.time() - t, 2)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, *okno, "-ar", "44100", "-ac", "2", wav], check=True)
    return wav, round(time.time() - t, 2), info


def _separate(wav, d, opts):
    """→ ({stem: path}, sample rate, info). subtype PCM_16 / PCM_24 (default 24); integer formats get values clipped to the
    integer range first (libsndfile does not clip by itself: an over-full sample would wrap around into a click), and the
    number of clipped samples is reported."""
    t = time.time()
    name = opts.get("model", "htdemucs")
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    loaded = name in _roformer or name in _demucs
    t_load = time.time()
    if name in ROFORMER:
        # vocals model: the minus is the mix minus the vocals (minus + vocals = the mix exactly)
        model, cfg = _load_roformer(name)
        load_s = 0.0 if loaded else round(time.time() - t_load, 2)
        mix = torch.from_numpy(np.ascontiguousarray(x.T))
        overlap = int(opts.get("num_overlap", cfg["inference"].get("num_overlap", 4)))
        for b in dict.fromkeys([int(opts.get("batch", 4)), 1]):    # out of GPU memory on a 16 GB card → one chunk at a time
            try:
                vocals = _demix_roformer(model, cfg, mix, overlap, b)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if b == 1:
                    raise
        no_vocals = mix - vocals
    else:
        import demucs.apply
        model = _load_demucs(name)
        load_s = 0.0 if loaded else round(time.time() - t_load, 2)
        mix = torch.from_numpy(x.T).unsqueeze(0).to(DEV)
        ref = mix.mean(0); mix = (mix - ref.mean()) / (ref.std() + 1e-8)
        # demucs shifts the audio by a random offset (0…0.5 s) per shift: two runs of one song differed by −28 dB. A seed makes
        # the offsets fixed (same quality, repeatable; a crop from 0 then matches the start of the whole song)
        if opts.get("seed") is not None:
            import random
            random.seed(int(opts["seed"]))
        with torch.no_grad():
            out = demucs.apply.apply_model(model, mix, shifts=int(opts.get("shifts", 1)), overlap=float(opts.get("overlap", 0.25)),
                                           split=True, device=DEV, progress=False)[0]
        out = out * (ref.std() + 1e-8) + ref.mean()
        stems = dict(zip(model.sources, out))
        vocals = stems["vocals"]
        no_vocals = sum(v for k, v in stems.items() if k != "vocals")
    arrays = {"vocals": vocals.cpu().numpy().T, "no_vocals": no_vocals.cpu().numpy().T}
    info = {"model": name, "model_s": round(time.time() - t - load_s, 2), "load_s": load_s}
    t = time.time()
    fmt = opts.get("format", "flac")
    subtype = opts.get("subtype", "PCM_24") if fmt in ("flac", "wav") else None
    top = {"PCM_16": 32767 / 32768, "PCM_24": 8388607 / 8388608}.get(subtype)
    clipped = {}

    def write(k):
        y = arrays[k]
        if top is not None:
            clipped[k] = int(np.count_nonzero((y > top) | (y < -1.0)))
            y = np.clip(y, -1.0, top)
        p = os.path.join(d, f"{k}.{fmt}")
        sf.write(p, y, sr, format=fmt.upper(), subtype=subtype)
        return k, p

    with ThreadPoolExecutor(2) as ex:
        paths = dict(ex.map(write, arrays))
    info.update(write_s=round(time.time() - t, 2), subtype=subtype, clipped=clipped, seed=opts.get("seed"), shifts=int(opts.get("shifts", 1)))
    return paths, sr, info


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
    a = _load_aligner()
    lang = opts.get("language", "eng")
    t_r = time.time()
    y, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    read_s = round(time.time() - t_r, 2)
    t_r = time.time()
    x = _resample16(y.mean(1), sr)
    resample_s = round(time.time() - t_r, 2)
    t = time.time()
    # batch by GPU memory: 16 windows ran out of memory on a 20 GB card (onnxruntime "Reduce…" runtime error), 24 GB was fine;
    # on any GPU error retry with smaller batches, then on a CPU session, and say so in the output
    if "CUDAExecutionProvider" in ALIGN_PROVIDERS:
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        batch = int(opts.get("batch_size", 16 if mem_gb >= 23 else (8 if mem_gb >= 15 else 4)))
    else:
        batch = int(opts.get("batch_size", 4))
    em, used = None, None
    for b in dict.fromkeys([batch, 4, 2]):
        try:
            em, _ = cfa.generate_emissions(a.model, x, batch_size=b)
            used = f"gpu batch {b}" if "CUDAExecutionProvider" in ALIGN_PROVIDERS else f"cpu batch {b}"
            break
        except Exception as e:
            last = repr(e)[:200]
    if em is None:                                     # last resort: the same model on the CPU, slow but finished
        cpu = _ort_session(a.model_path, providers=["CPUExecutionProvider"])
        em, _ = cfa.generate_emissions(cpu, x, batch_size=4)
        used = f"cpu fallback after: {last}"
    emissions_s = round(time.time() - t, 2)
    step = len(x) / 16000 / em.shape[0]
    lines = list(opts.get("lines", []))
    times = list(opts.get("times") or [None] * len(lines))
    win = {"before_s": 1.5, "after_s": 0.3, "last_s": 8.0, **(opts.get("window") or {})}
    parsed = [_lead(l) for l in lines]
    lead = [p[0] if not p[2] else "" for p in parsed]
    t_pass1 = time.time()
    all_words = _words(cfa, a.tokenizer, em, step, " ".join(l for l in lead if l), lang)
    pass1_s = round(time.time() - t_pass1, 2)
    t_pl = time.time()
    per_line = [len(cfa.preprocess_text(l, romanize=True, language=lang, split_size="word")[1]) // 2 if l else 0 for l in lead]
    per_line_s = round(time.time() - t_pl, 2)
    by_line, i = [], 0
    for n in per_line:
        by_line.append(all_words[i:i + n]); i += n
    count_ok = i == len(all_words)
    if not count_ok:
        by_line = None
    diffs = [w[0]["t0"] - tt for w, tt in zip(by_line or [], times) if w and tt is not None]
    offset = float(np.median(diffs)) if diffs else 0.0
    windowed, dur = [], len(x) / 16000
    t_pass2 = time.time()
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
            "emissions_s": emissions_s, "pass1_s": pass1_s, "pass2_s": round(time.time() - t_pass2, 2),
            "read_s": read_s, "resample_s": resample_s, "per_line_s": per_line_s,
            "providers": ALIGN_PROVIDERS, "emissions_on": used}


def _f0(wav_path, opts):
    """Pitch of the (separated) voice with RMVPE, 16 kHz, hop 10 ms, chunks of 3200 frames with 200 frames of context.
    Returns the raw decoding per frame: cents10 = weighted mean of the 9 bins around the salience peak (cents re 10 Hz,
    f0 = 10 * 2 ** (cents10 / 1200)) and smax = the peak salience; thresholds (voicing, f0 range) stay with the caller."""
    t = time.time()
    m = _load_rmvpe()
    sr, hop_ms, n_fft = 16000, float(opts.get("hop_ms", 10)), 1024
    hop = int(sr * hop_ms / 1000)
    y, sr0 = sf.read(wav_path, dtype="float32", always_2d=True)
    y16 = _resample16(y.mean(1), sr0)
    mel_b = torch.from_numpy(_mel_basis()).float().to(DEV)
    x = torch.from_numpy(y16).to(DEV)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=torch.hann_window(n_fft, device=DEV),
                      center=True, return_complex=True).abs()
    mel = torch.log(torch.clamp(mel_b @ spec, min=1e-5))
    T = mel.shape[-1]
    kus, ctx = int(opts.get("chunk_frames", 3200)), int(opts.get("context_frames", 200))
    sal = np.zeros((T, 360), dtype=np.float32)
    with torch.no_grad():
        for a0_ in range(0, T, kus):
            b_ = min(T, a0_ + kus)
            a0, b0 = max(0, a0_ - ctx), min(T, b_ + ctx)
            ch = mel[:, a0:b0]
            n = ch.shape[-1]
            pad = 32 * ((n - 1) // 32 + 1) - n
            if pad:
                ch = torch.nn.functional.pad(ch, (0, pad), mode="constant")
            out = m(ch[None])[0, :n].float().cpu().numpy()
            sal[a0_:b_] = out[a0_ - a0:a0_ - a0 + (b_ - a0_)]
    cm = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
    s = np.pad(sal, ((0, 0), (4, 4)))
    cen = np.argmax(s, axis=1)
    idx = cen[:, None] + np.arange(-4, 5)[None]
    w = np.take_along_axis(s, idx, axis=1)
    cents10 = (w * cm[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
    return {"hop_s": hop / sr, "cents10": [round(float(v), 1) for v in cents10], "smax": [round(float(v), 4) for v in s.max(1)],
            "sec": round(time.time() - t, 2)}


def _deliver(paths, job_input, max_mb):
    """All files at once (parallel requests); returns {stem: record} plus upload_s."""
    t = time.time()
    with ThreadPoolExecutor(max(1, len(paths))) as ex:
        out = dict(ex.map(lambda kp: (kp[0], _deliver_one(kp[1], job_input, max_mb)), paths.items()))
    out["upload_s"] = round(time.time() - t, 2)
    return out


def _deliver_one(p, job_input, max_mb):
    up = job_input.get("upload") or {}
    data = open(p, "rb").read()
    rec = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if up.get("url"):                                   # raw body, one request per file: Authorization: Bearer <token>, X-File: <name>
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
    return rec


JOBS = {"n": 0}


def handler(job):
    t0 = time.time()
    JOBS["n"] += 1
    inp = job.get("input") or {}
    tasks = inp.get("tasks") or ["separate", "align"]
    res = {"device": DEV, "gpu": GPU, "startup_s": STARTUP_S, "load_s": LOAD_S, "warm_s": WARM_S, "timing": {},
           "worker": {"job_n": JOBS["n"], "ready_for_s": round(t0 - _T_START - STARTUP_S, 1)}}
    with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(1) as bg:
        wav, res["timing"]["download_s"], dl = _audio(inp, d)
        res["timing"].update(dl)
        paths, sending = {}, None
        if "separate" in tasks:
            t = time.time()
            paths, sr, info = _separate(wav, d, inp.get("separate") or {})
            res["separate"] = {**info, "sec": round(time.time() - t, 2)}
            # results travel to the caller while the GPU aligns and measures pitch (they only read the local files)
            sending = bg.submit(_deliver, paths, inp, float(inp.get("return_inline_max_mb", 18)))
        if "align" in tasks:
            t = time.time()
            a = inp.get("align") or {}
            src = paths.get("vocals") if a.get("audio", "vocals") == "vocals" and paths.get("vocals") else wav
            res["align"] = _align(src, a)
            res["align"]["sec"] = round(time.time() - t, 2)
        if "f0" in tasks:
            f = inp.get("f0") or {}
            src = paths.get("vocals") if f.get("audio", "vocals") == "vocals" and paths.get("vocals") else wav
            res["f0"] = _f0(src, f)
        if sending is not None:
            t = time.time()
            res["separate"].update(sending.result())
            res["timing"]["upload_wait_s"] = round(time.time() - t, 2)
    res["timing"]["total_s"] = round(time.time() - t0, 2)
    return res


runpod.serverless.start({"handler": handler})

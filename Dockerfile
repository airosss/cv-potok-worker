# Audio separation (demucs htdemucs) + forced alignment (ctc-forced-aligner, MMS) serverless worker for RunPod.
# Generic worker: no keys, no project paths inside; everything job-specific comes in the job input (see handler.py).

# stage 1: ctc-forced-aligner is published only as source with a C++ extension: compile a wheel here, ship it without compilers
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime AS wheels
ENV PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip wheel --no-deps -w /wheels ctc-forced-aligner==1.0.2

# stage 2: runtime
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TORCH_HOME=/models/torch DEBIAN_FRONTEND=noninteractive
# the CUDA/cuDNN libraries of this image live in the conda env: onnxruntime's GPU provider must find them to run on the GPU
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install demucs==4.0.1 runpod soundfile librosa soxr requests unidecode einops pyyaml packaging rotary-embedding-torch==0.6.5 beartype==0.18.5
COPY --from=wheels /wheels /wheels
RUN pip install --no-deps /wheels/*.whl && rm -rf /wheels
RUN pip uninstall -y onnxruntime
# onnxruntime-gpu is pinned to a CUDA 12 / cuDNN 9 build: the newest PyPI wheels need CUDA 13 libraries, which this image does not
# have, and onnxruntime then falls back to the CPU silently (measured: 48 s of alignment on the CPU for a 4-min song)
RUN pip install onnxruntime-gpu==1.20.2
# torch brings its own CUDA 12 / cuDNN 9 libraries inside the python packages: register their folders for the dynamic loader
RUN find /opt/conda \( -name "libcudart.so.12*" -o -name "libcudnn.so.9*" -o -name "libcublasLt.so.12*" -o -name "libcublas.so.12*" \
         -o -name "libcufft.so.11*" -o -name "libcurand.so.10*" -o -name "libcudnn_adv.so.9*" -o -name "libnvrtc.so.12*" -o -name "libnvJitLink.so.12*" \) -exec dirname {} \; | sort -u \
         > /etc/ld.so.conf.d/zz-cuda-from-python.conf && cat /etc/ld.so.conf.d/zz-cuda-from-python.conf && ldconfig
# fail the build if the CUDA provider of onnxruntime cannot resolve its libraries (no GPU needed to check this)
RUN P=$(python -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi', 'libonnxruntime_providers_cuda.so'))") && \
    ldd "$P" | tee /tmp/ldd.txt && ! grep -q "not found" /tmp/ldd.txt
# models baked into the image: a cold start downloads nothing
RUN mkdir -p /models/torch /root/ctc_forced_aligner && python -c "import demucs.pretrained as p; p.get_model('htdemucs'); p.get_model('htdemucs_ft')" && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx', '/root/ctc_forced_aligner/model.onnx')"
# RMVPE pitch network: architecture code from the RMVPE / RVC projects (MIT), the public RVC weights checked by checksum
COPY rmvpe_arch /models/rmvpe/rmvpe_arch
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt', '/models/rmvpe/rmvpe.pt')" && \
    echo "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193  /models/rmvpe/rmvpe.pt" | sha256sum -c -
# BS-RoFormer vocals model (viperx ep 317): code from python-audio-separator 0.48.0 (MIT), public UVR weights checked by checksum,
# built and loaded on the CPU here so a wrong file or an incompatible code change fails the build, not a job
COPY roformer_arch /models/roformer/roformer_arch
RUN python -c "import urllib.request as u; u.urlretrieve('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt', '/models/roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt'); u.urlretrieve('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml', '/models/roformer/model_bs_roformer_ep_317_sdr_12.9755.yaml')" && \
    printf '%s  %s\n' 5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa /models/roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
                     2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f /models/roformer/model_bs_roformer_ep_317_sdr_12.9755.yaml | sha256sum -c - && \
    cd /models/roformer && python -c "from roformer_arch.load import build; m, c = build('model_bs_roformer_ep_317_sdr_12.9755.ckpt', 'model_bs_roformer_ep_317_sdr_12.9755.yaml'); print('roformer ok', sum(p.numel() for p in m.parameters()))"
# the mel filter bank of the pitch step, computed once here: at run time librosa is not imported at all (its first call in a
# fresh worker paid ~14 s of numba compilation)
RUN python -c "import librosa, numpy as np; np.save('/models/rmvpe/mel_16k_1024_128_30_8000_htk.npy', librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True))"
COPY handler.py /handler.py
COPY fastaudio.py /fastaudio.py
# the librosa-free resampler must give exactly librosa's numbers (odd lengths, 44.1 / 48 / 22.05 kHz), the baked mel must be librosa's
RUN cd / && python -c "import numpy as np, librosa; from fastaudio import resample16 as r; \
y = np.random.default_rng(0).standard_normal(44101 * 3).astype(np.float32) * 0.1; \
assert all(np.array_equal(r(y[:n], sr), librosa.resample(y[:n], orig_sr=sr, target_sr=16000).astype(np.float32)) for sr in (44100, 48000, 22050) for n in (len(y), len(y) - 7)); \
assert np.array_equal(np.load('/models/rmvpe/mel_16k_1024_128_30_8000_htk.npy'), librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)); \
print('resample and mel identical to librosa')"
# last step: fail the build if any import or baked model is missing (a masked install error once shipped an image without the aligner)
RUN python -c "import demucs, runpod, soundfile, librosa, unidecode, onnxruntime, ctc_forced_aligner, os; assert os.path.getsize('/root/ctc_forced_aligner/model.onnx') > 1e9; assert os.path.getsize('/models/rmvpe/rmvpe.pt') > 1e8; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers(); print('imports ok', onnxruntime.get_available_providers())"
CMD ["python", "-u", "/handler.py"]

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
RUN pip install demucs==4.0.1 runpod soundfile librosa requests unidecode
COPY --from=wheels /wheels /wheels
RUN pip install --no-deps /wheels/*.whl && rm -rf /wheels
RUN pip uninstall -y onnxruntime
# onnxruntime-gpu is pinned to a CUDA 12 / cuDNN 9 build: the newest PyPI wheels need CUDA 13 libraries, which this image does not
# have, and onnxruntime then falls back to the CPU silently (measured: 48 s of alignment on the CPU for a 4-min song)
RUN pip install onnxruntime-gpu==1.20.2
# fail the build if the CUDA provider of onnxruntime cannot resolve its libraries (no GPU needed to check this)
RUN P=$(python -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi', 'libonnxruntime_providers_cuda.so'))") && \
    ldd "$P" | tee /tmp/ldd.txt && ! grep -q "not found" /tmp/ldd.txt
# models baked into the image: a cold start downloads nothing
RUN mkdir -p /models/torch /root/ctc_forced_aligner && python -c "import demucs.pretrained as p; p.get_model('htdemucs')" && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx', '/root/ctc_forced_aligner/model.onnx')"
COPY handler.py /handler.py
# last step: fail the build if any import or baked model is missing (a masked install error once shipped an image without the aligner)
RUN python -c "import demucs, runpod, soundfile, librosa, unidecode, onnxruntime, ctc_forced_aligner, os; assert os.path.getsize('/root/ctc_forced_aligner/model.onnx') > 1e9; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers(); print('imports ok', onnxruntime.get_available_providers())"
CMD ["python", "-u", "/handler.py"]

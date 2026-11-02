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
# torch brings its own CUDA 12 / cuDNN 9 libraries inside the python packages: register their folders for the dynamic loader
RUN find /opt/conda \( -name "libcudart.so.12*" -o -name "libcudnn.so.9*" -o -name "libcublasLt.so.12*" -o -name "libcublas.so.12*" \
         -o -name "libcufft.so.11*" -o -name "libcurand.so.10*" -o -name "libcudnn_adv.so.9*" -o -name "libnvrtc.so.12*" -o -name "libnvJitLink.so.12*" \) -exec dirname {} \; | sort -u \
         > /etc/ld.so.conf.d/zz-cuda-from-python.conf && cat /etc/ld.so.conf.d/zz-cuda-from-python.conf && ldconfig
# fail the build if the CUDA provider of onnxruntime cannot resolve its libraries (no GPU needed to check this)
RUN P=$(python -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi', 'libonnxruntime_providers_cuda.so'))") && \
    ldd "$P" | tee /tmp/ldd.txt && ! grep -q "not found" /tmp/ldd.txt
# models baked into the image: a cold start downloads nothing
RUN mkdir -p /models/torch /root/ctc_forced_aligner && python -c "import demucs.pretrained as p; p.get_model('htdemucs')" && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx', '/root/ctc_forced_aligner/model.onnx')"
# RMVPE pitch network: architecture code from the RMVPE / RVC projects (MIT), the public RVC weights checked by checksum
COPY rmvpe_arch /models/rmvpe/rmvpe_arch
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt', '/models/rmvpe/rmvpe.pt')" && \
    echo "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193  /models/rmvpe/rmvpe.pt" | sha256sum -c -
COPY handler.py /handler.py
# last step: fail the build if any import or baked model is missing (a masked install error once shipped an image without the aligner)
RUN python -c "import demucs, runpod, soundfile, librosa, unidecode, onnxruntime, ctc_forced_aligner, os; assert os.path.getsize('/root/ctc_forced_aligner/model.onnx') > 1e9; assert os.path.getsize('/models/rmvpe/rmvpe.pt') > 1e8; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers(); print('imports ok', onnxruntime.get_available_providers())"
CMD ["python", "-u", "/handler.py"]

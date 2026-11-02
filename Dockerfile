# Audio separation (demucs htdemucs) + forced alignment (ctc-forced-aligner, MMS) serverless worker for RunPod.
# Generic worker: no keys, no project paths inside; everything job-specific comes in the job input (see handler.py).
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TORCH_HOME=/models/torch DEBIAN_FRONTEND=noninteractive
# build-essential: ctc-forced-aligner is published only as source with a C++ extension, so pip compiles it
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install demucs==4.0.1 runpod soundfile librosa requests unidecode
RUN pip install --no-deps ctc-forced-aligner==1.0.2
RUN pip uninstall -y onnxruntime || true
RUN pip install onnxruntime-gpu
# fail the build loudly if anything the handler imports is missing (a masked install error once shipped an image without the aligner)
RUN python -c "import demucs, runpod, soundfile, librosa, unidecode, onnxruntime, ctc_forced_aligner; print('imports ok', onnxruntime.get_available_providers())"
# models baked into the image: a cold start downloads nothing
RUN mkdir -p /models/torch /root/ctc_forced_aligner && python -c "import demucs.pretrained as p; p.get_model('htdemucs')" && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx', '/root/ctc_forced_aligner/model.onnx')"
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]

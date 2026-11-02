# audio-sep-align-worker

RunPod serverless worker: source separation (demucs `htdemucs`, vocals / no_vocals) and forced alignment of known lyrics
(ctc-forced-aligner, MMS). Models are baked into the image, so a cold start downloads nothing. Generic: the job input carries
the audio (URL or base64), the text lines and, optionally, an upload URL + token for the results. See `handler.py` for the schema.

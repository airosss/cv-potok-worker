# audio-sep-align-worker

RunPod serverless worker: source separation (vocals / no_vocals), forced alignment of known lyrics (ctc-forced-aligner, MMS)
and pitch of the separated voice (RMVPE). Models are baked into the image, so a cold start downloads nothing. Generic: the job
input carries the audio (URL or base64), the text lines and, optionally, an upload URL + token for the results. See
`handler.py` for the schema.

Separation models (`separate.model`):
- `htdemucs` (default), `htdemucs_ft` — demucs 4; `seed` fixes the random shift offsets (repeatable output).
- `bs_roformer_1297` — BS-RoFormer vocals model (viperx, epoch 317); the minus is the mix minus the vocals.
  `num_overlap` (default from its config, 4) and `batch` set speed vs. smoothness.

Environment: `PRELOAD` = comma-separated separation models loaded and warmed at worker start (default `htdemucs`);
the others load on first use.

Third-party code (MIT): `rmvpe_arch/` from the RMVPE / RVC projects; `roformer_arch/` from python-audio-separator 0.48.0,
which carries lucidrains/BS-RoFormer as adapted by ZFTurbo/Music-Source-Separation-Training (see `roformer_arch/LICENSE`).
Weights are downloaded at build time from their public locations and checked by sha256.

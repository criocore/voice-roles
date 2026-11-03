# Security and privacy

voice-roles is a local tool. There is no server, no account, no telemetry and no cloud inference. What it does with your data:

- **The recording never leaves the machine.** Whisper, the diarization models and the pitch tracker all run through CTranslate2 and onnxruntime on your CPU. The only network access is downloading models: Whisper from Hugging Face on the first run (faster-whisper does it), the two diarization models from the sherpa-onnx GitHub releases (`get_models.py`). Both can be done once on a connected machine and copied over.
- **The review page is a static file.** `*.review.html` contains the transcript as embedded JSON and runs no network code. It reads the recording from the same folder through the browser's own file access and stores unsaved edits in the browser's localStorage for that file. Saving uses the browser's save dialog; nothing is uploaded.
- **Voice profiles are biometric-like data.** `voices/<name>.npy` holds speaker embeddings (192 numbers per sample) that identify a person's voice. They are excluded from git by `.gitignore`; keep them out of shared folders, and delete a profile when the person asks. The same applies to `*.auto.json` and `*.review.json`, which hold the full transcript.
- **No shell interpolation of user text.** ffmpeg is called with an argument list, never through a shell string; file names and transcript text are never passed to a shell.
- **Model files are code-like inputs.** ONNX models are loaded by onnxruntime; take them only from the URLs in `get_models.py` or from sherpa-onnx releases.

## Reporting a vulnerability

Open a GitHub issue with the label `security`, or, if the problem exposes other people's recordings or voices, write to the maintainer through the contact in the GitHub profile first and give a few days before public disclosure. Please include the version (`git rev-parse HEAD`), the OS and a minimal reproduction.

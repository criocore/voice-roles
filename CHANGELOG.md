# Changelog

## 0.1.0 (2026-11-03)

First public release. Everything was built and measured on one machine (Windows 11, Ryzen 7 5800H, 8 cores, no GPU) on Russian meeting recordings; reports from other machines and languages are the next step.

- Transcription with faster-whisper (CTranslate2 int8, CPU), `.txt` and `.srt` with timestamps.
- `--speakers`: who spoke when through sherpa-onnx (pyannote-segmentation-3.0 int8 + TitaNet-small embeddings), with automatic speaker count: cluster at a high threshold, merge near-identical voices, give short "ghost" clusters to the nearest voice. Words get speakers by largest time overlap. Output `.speakers.txt`.
- Intonation per turn through SwiftF0: pitch in semitones from the speaker's own median, final-300-ms slope → contour (rise, fall, level), loudness z-score, pauses. Output `.prosody.md`.
- `.review.html`: a self-contained page for a human operator. Rename people, reassign, split (right-click a word), merge and delete turns, edit words, fix contours, flag turns, listen to any turn from the recording next to the page. Edits autosave in the browser; "Save .review.json" writes the corrected transcript.
- `apply_review.py`: rebuilds the outputs from the operator's file, stores voice profiles under real names in `voices/` (used to label people on later recordings without `--names`), collects contour labels and tunes the contour thresholds once 30 labels exist.
- `bench_quality.py`: automatic result vs the operator's corrections: speaker count, word-level speaker accuracy with best mapping, time error (DER-like), name accuracy from voice profiles, missed and extra turn boundaries, word edits, contour agreement with a confusion matrix, stage timings. `--sweep` re-diarizes with other thresholds against the same truth. Results are appended to `calibration/bench.md`.
- `get_models.py`: downloads the two diarization models (45 MB) into `models/diar`.
- Measured on an 18-minute four-person meeting: text (medium) 634 s, roles 146 s (RTF 0.13), intonation 6 s; cross-speaker cosine ≤ 0.51, same speaker across cuts 0.88 to 0.96; fixed `--clusters 4` gave 82% word agreement against the automatic merge pipeline's result.

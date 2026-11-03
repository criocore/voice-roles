# Contributing

Thanks for looking. The most useful contributions right now are measurements, not code: the whole tool was built and checked on one machine (Windows 11, Ryzen 7 5800H) and on a handful of Russian recordings.

## Reports that help

- **A benchmark table from your machine.** Run `bench_quality.bat` on a recording you corrected on the review page and paste the table into an issue with your CPU, OS, recording length, number of people and the model (`small`, `medium`, `large-v3`). The "Measured" section of the README grows from such reports.
- **A voice-profile result across sessions.** Did `voices/` recognise the same people on a different day or microphone? Which `--voice-sim` worked? Numbers from the "имена по голосу верны" line and the per-speaker table are enough.
- **Contour threshold tuning.** After `apply_review` tuned `calibration/contour_thresholds.json` on 30 and more labels, report the before/after agreement it printed.
- **macOS and Linux.** The Python code has nothing Windows-specific, but only the `.bat` launchers exist. A `.sh` twin and a report that `get_models.py`, `transcribe.py --speakers` and the review page work there would close that gap.
- **Other languages.** Whisper and the diarization models are multilingual; the pitch logic is language-free. The review page strings and the contour names are Russian. Reports from other languages, or a small i18n of the page, are welcome.

## Code

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt pytest
venv\Scripts\python -m pytest -q tests
```

- Keep it CPU-only, keyless and offline after the model download: no torch, no cloud calls, no telemetry.
- Pure logic (word assignment, turns, contour labels, profile matching, benchmark metrics) lives in functions without model calls and is covered by `tests/test_core.py`. Add a test when you change one of those.
- Models are called only through sherpa-onnx, faster-whisper and swift-f0; new models should come with a licence note in the README.
- Comments and docstrings in the code are Russian (the tool's audience); commit messages and these documents are English.
- CI runs `py_compile` and the unit tests on Ubuntu, Windows and macOS with Python 3.11 and 3.12.

## Pull requests

One topic per pull request. Say what you ran live (OS, CPU, recording) and what you did not. Code is under MIT; by contributing you agree to those terms and to the [code of conduct](CODE_OF_CONDUCT.md).

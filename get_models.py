"""Скачивает модели для ролей (sherpa-onnx) в models/diar. Whisper качается сам при первом прогоне.

    get_models.bat
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

DIAR = Path(__file__).resolve().parent / "models" / "diar"
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
FILES = {
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2": BASE + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "nemo_en_titanet_small.onnx": BASE + "speaker-recongition-models/nemo_en_titanet_small.onnx",  # опечатка в адресе у авторов
}


def fetch(url: str, dst: Path) -> None:
    def hook(n, size, total):
        if total > 0:
            print(f"\r  {dst.name}: {min(100, n * size * 100 // total)}%", end="", flush=True)
    urllib.request.urlretrieve(url, dst, hook)
    print()


def main() -> int:
    DIAR.mkdir(parents=True, exist_ok=True)
    seg = DIAR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx"
    emb = DIAR / "nemo_en_titanet_small.onnx"
    if seg.exists() and emb.exists():
        print("Модели уже на месте:", DIAR)
        return 0
    if not seg.exists():
        arc = DIAR / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        fetch(FILES[arc.name], arc)
        with tarfile.open(arc, "r:bz2") as t:
            t.extractall(DIAR, filter="data")
        arc.unlink()
    if not emb.exists():
        fetch(FILES[emb.name], emb)
    ok = seg.exists() and emb.exists()
    print("Готово" if ok else "Что-то не скачалось, проверь интернет и запусти ещё раз", DIAR)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

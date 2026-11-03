"""Локальная расшифровка лекций: видео/аудио -> .txt (с метками времени) + .srt.

Работает офлайн (faster-whisper на CPU), ничего никуда не отправляет.
Результат кладётся рядом с исходным файлом. Уже расшифрованные файлы пропускаются.

Примеры:
    transcribe.bat "C:\\path\\lecture.mp4"
    transcribe.bat "C:\\path\\папка_с_лекциями"
    transcribe.bat lecture.mp4 --model small --force
    transcribe.bat meeting.mp4 --speakers --names "Анна,Борис,Вера"

С --speakers рядом появляются ещё три файла: .speakers.txt (кто что сказал), .prosody.md
(интонация по репликам) и .review.html (страница правок для оператора: открыть в браузере,
исправить имена, говорящих, слова и контуры, сохранить .review.json и перетащить его на
apply_review.bat). Число людей определяется само, --clusters N задаёт его жёстко. Имена берутся
из voices/ по голосу (после первой правки оператором) или из --names по порядку появления.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4a", ".mp3", ".wav", ".ogg", ".flac", ".aac"}
MODELS_DIR = Path(__file__).resolve().parent / "models"


def ts(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}" if srt else f"{h:02}:{m:02}:{s:02}"


def extract_audio(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def collect(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in MEDIA_EXT)
        elif p.is_file():
            files.append(p)
        else:
            print(f"Не найдено: {p}")
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="Локальная расшифровка лекций")
    ap.add_argument("paths", nargs="+", help="файлы или папки")
    ap.add_argument("--model", default="medium", help="tiny/base/small/medium/large-v3 (по умолчанию medium)")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--force", action="store_true", help="перерасшифровать, даже если .txt уже есть")
    ap.add_argument("--speakers", action="store_true", help="роли (кто говорил) и интонация, файлы .speakers.txt и .prosody.md")
    ap.add_argument("--names", default="", help="имена по порядку первого появления: Анна,Борис,Вера")
    ap.add_argument("--clusters", type=int, default=-1, help="число людей, если известно (иначе определяется само)")
    ap.add_argument("--threshold", type=float, default=0.9, help="порог кластеризации голосов, выше = меньше людей")
    ap.add_argument("--threads", type=int, default=8, help="потоков CPU для диаризации")
    ap.add_argument("--voice-sim", type=float, default=0.6, help="порог сходства с голосами из voices/ для подписи по имени")
    args = ap.parse_args()

    def done(f: Path) -> bool:
        if not f.with_suffix(".txt").exists():
            return False
        return not args.speakers or f.with_suffix(".speakers.txt").exists()

    files = collect(args.paths)
    todo = [f for f in files if args.force or not done(f)]
    for f in files:
        if f not in todo:
            print(f"Пропуск (уже готово): {f.name}")
    if not todo:
        return 0

    from faster_whisper import WhisperModel
    print(f"Загрузка модели {args.model}... (первый раз она скачивается)")
    model = WhisperModel(args.model, device="cpu", compute_type="int8", download_root=str(MODELS_DIR))

    for src in todo:
        print(f"\n=== {src.name}")
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            extract_audio(src, wav)
            segments, info = model.transcribe(
                str(wav),
                language=args.lang,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,  # меньше «зацикливаний» на длинных записях
                word_timestamps=args.speakers,
            )
            txt_lines, srt_blocks, words = [], [], []
            for i, seg in enumerate(segments, 1):
                text = seg.text.strip()
                if not text:
                    continue
                txt_lines.append(f"[{ts(seg.start)}] {text}")
                srt_blocks.append(f"{i}\n{ts(seg.start, True)} --> {ts(seg.end, True)}\n{text}\n")
                for w in seg.words or []:
                    words.append({"start": w.start, "end": w.end, "word": w.word})
                print(f"\r  {ts(seg.end)} / {ts(info.duration)}", end="", flush=True)

            src.with_suffix(".txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
            src.with_suffix(".srt").write_text("\n".join(srt_blocks), encoding="utf-8")
            print(f"\n  Текст за {time.time() - t0:.0f} с -> {src.with_suffix('.txt')}")

            if args.speakers and words:
                import speakers
                t1 = time.time()
                names = [n.strip() for n in args.names.split(",") if n.strip()]
                r = speakers.run(wav, words, names, src.stem, args.threads, args.clusters, args.threshold,
                                 media=src.name, voice_sim=args.voice_sim,
                                 params={"model": args.model, "names_given": bool(names), "text_sec": round(time.time() - t0)})
                src.with_suffix(".speakers.txt").write_text(r["txt"], encoding="utf-8")
                src.with_suffix(".prosody.md").write_text(r["md"], encoding="utf-8")
                src.with_suffix(".review.html").write_text(r["html"], encoding="utf-8")
                # нетронутый автоматический результат: с ним bench_quality.py сравнивает правки оператора
                src.with_suffix(".auto.json").write_text(json.dumps(r["data"], ensure_ascii=False), encoding="utf-8")
                by_voice = f", по голосу узнаны: {', '.join(r['named'].values())}" if r["named"] else ""
                print(f"  Роли за {time.time() - t1:.0f} с, людей {r['n']}{by_voice} -> {src.with_suffix('.speakers.txt')}")
                print(f"  Правки оператора: открой {src.with_suffix('.review.html').name}, сохрани .review.json, "
                      f"перетащи его на apply_review.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())

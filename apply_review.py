"""Применяет правки оператора из .review.json (сохранён со страницы .review.html).

    apply_review.bat "C:\\path\\meeting.review.json"      (или сам файл записи, json ищется рядом)

Что делает:
1. Пересобирает .speakers.txt, .prosody.md и .review.html по исправленным данным.
2. Запоминает голоса людей с настоящими именами (не "Спикер N") в voices/<Имя>.npy.
   В следующих прогонах transcribe.bat --speakers подписывает их по голосу, без --names.
3. Складывает метки контуров интонации в calibration/contours.json и, когда меток >= 30,
   подбирает пороги (calibration/contour_thresholds.json), которые лучше сходятся с оператором.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import speakers as sp

MIN_LABELS = 30
MIN_VOICE_SEC = 3.0
SAME_PERSON = 0.55  # ниже этого новый голос под старым именем считаем ошибкой и не сохраняем


def find_json(arg: Path) -> Path:
    if arg.name.endswith(".review.json"):
        return arg
    cand = arg.with_suffix(".review.json")
    if cand.exists():
        return cand
    raise SystemExit(f"Нет файла правок рядом: {cand}")


def rebuild(data: dict, out_stem: Path) -> list[dict]:
    turns, base, names = sp.import_turns(data)
    txt, md = sp.render(turns, base, names, data["title"])
    out = lambda ext: out_stem.with_name(out_stem.name + ext)  # noqa: E731
    out(".speakers.txt").write_text(txt, encoding="utf-8")
    out(".prosody.md").write_text(md, encoding="utf-8")
    clean = dict(data)
    clean.pop("saved", None)
    clean.pop("edits", None)
    out(".review.html").write_text(sp.review_html(clean), encoding="utf-8")
    print(f"Пересобраны {out_stem.name}.speakers.txt / .prosody.md / .review.html, реплик {len(turns)}, людей {len(names)}")
    return turns


def update_voices(data: dict, media: Path, threads: int) -> None:
    names = [s["name"].strip() for s in data["speakers"]]
    spans: dict[int, list] = {}
    for tr in data["turns"]:
        if tr["words"]:
            spans.setdefault(tr["spk"], []).append((tr["words"][0]["s"], tr["words"][-1]["e"]))
    todo = {k: v for k, v in spans.items() if names[k] and not sp.GENERIC.match(names[k])
            and sum(b - a for a, b in v) >= MIN_VOICE_SEC}
    if not todo:
        print("Голоса: настоящих имён нет, профили не обновлены (имена задаются в карточках сверху страницы)")
        return
    if not media.exists():
        print(f"Голоса: запись {media} не найдена, профили не обновлены")
        return
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        samples, sr = sf.read(str(wav), dtype="float32")
    ext = sp.extractor(threads)
    old = sp.load_profiles()
    for k, v in todo.items():
        e = sp.span_embedding(ext, samples, sr, v)
        if e is None:
            continue
        name = names[k]
        if name in old:
            score = float((old[name] @ e).max())
            if score < SAME_PERSON:
                print(f"Голоса: {name}: новый голос не похож на сохранённый (косинус {score:.2f}), "
                      f"не записан. Другой человек с тем же именем? Удали voices/{name}.npy, если профиль устарел")
                continue
            n = sp.save_profile(name, e)
            print(f"Голоса: {name}: добавлен образец (сходство со старыми {score:.2f}), всего {n}")
        else:
            sp.save_profile(name, e)
            print(f"Голоса: {name}: новый профиль, {sum(b - a for a, b in v):.0f} с речи")


def update_contours(data: dict, stem: str) -> None:
    sp.CALIB.mkdir(exist_ok=True)
    store = sp.CALIB / "contours.json"
    db = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
    rows = [{"k": tr["slope"], "end": tr["end_semi"], "label": tr.get("contour_fix") or tr["contour"],
             "fixed": bool(tr.get("contour_fix"))}
            for tr in data["turns"] if tr.get("slope") is not None and tr.get("end_semi") is not None]
    db[stem] = rows
    store.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
    allrows = [r for v in db.values() for r in v]
    fixed = sum(r["fixed"] for r in allrows)
    print(f"Контуры: {len(rows)} меток из этого файла ({sum(r['fixed'] for r in rows)} исправлено), всего {len(allrows)}")
    if len(allrows) < MIN_LABELS or fixed == 0:
        return
    th0 = sp.contour_thresholds()
    acc = lambda th: np.mean([sp.contour_label(r["k"], r["end"], th) == r["label"] for r in allrows])  # noqa: E731
    best, best_th = acc(th0), th0
    for k_hi in np.arange(3, 16.1, 1.0):
        for k_lo in np.arange(0.5, k_hi, 0.5):
            for end in np.arange(0.0, 3.1, 0.25):
                th = {"k_hi": float(k_hi), "k_lo": float(k_lo), "end": float(end)}
                a = acc(th)
                if a > best + 1e-9:
                    best, best_th = a, th
    if best_th is not th0:
        (sp.CALIB / "contour_thresholds.json").write_text(json.dumps(best_th), encoding="utf-8")
        print(f"Контуры: пороги подстроены {th0} -> {best_th}, совпадение с оператором {acc(th0):.0%} -> {best:.0%}")
    else:
        print(f"Контуры: текущие пороги {th0} лучшие, совпадение с оператором {best:.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Применить правки оператора к расшифровке")
    ap.add_argument("path", help=".review.json или файл записи, рядом с которым он лежит")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-voices", action="store_true", help="не трогать профили голосов")
    args = ap.parse_args()

    js = find_json(Path(args.path))
    data = json.loads(js.read_text(encoding="utf-8"))
    stem = js.with_name(js.name[: -len(".review.json")])
    media = js.with_name(data.get("media") or "")
    print(f"Правки: {js.name}, сохранены {data.get('saved', '?')}, правок на странице {data.get('edits', '?')}")
    rebuild(data, stem)
    if not args.no_voices:
        update_voices(data, media, args.threads)
    update_contours(data, stem.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

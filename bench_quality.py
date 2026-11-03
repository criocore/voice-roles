"""Метрики качества: автоматический результат (.auto.json) против правок оператора (.review.json).

    bench_quality.bat "C:\\path\\meeting.review.json"
    bench_quality.bat "C:\\path\\meeting.review.json" --sweep 0.7,0.8,0.9,0.95
    bench_quality.bat "C:\\path\\meeting.review.json" --sweep-clusters 3,4,5

Правда = то, что сохранил оператор со страницы .review.html. Сначала правь, потом меряй.
Каждый замер дописывается в calibration/bench.md, так виден прогресс калибровки по датам.
--sweep заново диаризует запись с разными порогами (или числом людей) и сравнивает с правдой:
так подбирается --threshold / --clusters для своих записей.
"""
import argparse
import itertools
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf

import speakers as sp


# ---------------------------------------------------------------- сравнение двух разметок
def words_of(data: dict) -> dict:
    out = {}
    for ti, tr in enumerate(data["turns"]):
        for j, w in enumerate(tr["words"]):
            out[(round(w["s"], 2), round(w["e"], 2))] = {"spk": tr["spk"], "turn": ti, "first": j == 0,
                                                         "dur": max(0.0, w["e"] - w["s"]), "w": w["w"].strip()}
    return out


def best_mapping(overlap: dict, auto_keys: list, truth_keys: list) -> dict:
    """auto spk -> truth spk, максимум суммарной длительности совпадений. Перебор до 8 человек, дальше жадно."""
    if len(auto_keys) <= 8 and len(truth_keys) <= 8:
        pool = truth_keys + [None] * max(0, len(auto_keys) - len(truth_keys))
        best, best_map = -1.0, {}
        for perm in itertools.permutations(pool, len(auto_keys)):
            score = sum(overlap.get((a, t), 0.0) for a, t in zip(auto_keys, perm) if t is not None)
            if score > best:
                best, best_map = score, {a: t for a, t in zip(auto_keys, perm) if t is not None}
        return best_map
    pairs = sorted(overlap.items(), key=lambda kv: -kv[1])
    used_a, used_t, m = set(), set(), {}
    for (a, t), _ in pairs:
        if a not in used_a and t not in used_t:
            m[a] = t
            used_a.add(a)
            used_t.add(t)
    return m


def label_of(tr: dict) -> str:
    return tr.get("contour_fix") or tr.get("contour") or "?"


def compare(auto: dict, truth: dict) -> dict:
    A, T = words_of(auto), words_of(truth)
    common = [k for k in T if k in A]
    overlap, cnt_t = {}, {}
    for k in common:
        a, t = A[k]["spk"], T[k]["spk"]
        overlap[(a, t)] = overlap.get((a, t), 0.0) + T[k]["dur"]
        cnt_t[t] = cnt_t.get(t, 0) + 1
    auto_keys = sorted({A[k]["spk"] for k in common})
    truth_keys = sorted({T[k]["spk"] for k in common})
    mapping = best_mapping(overlap, auto_keys, truth_keys)

    ok_words = sum(1 for k in common if mapping.get(A[k]["spk"]) == T[k]["spk"])
    total_dur = sum(T[k]["dur"] for k in common) or 1e-9
    bad_dur = sum(T[k]["dur"] for k in common if mapping.get(A[k]["spk"]) != T[k]["spk"])

    per_spk = []
    aname = lambda a: auto["speakers"][a]["name"] if a < len(auto["speakers"]) else "?"  # noqa: E731
    for t in truth_keys:
        tname = truth["speakers"][t]["name"]
        src = [a for a, tt in mapping.items() if tt == t]
        a = src[0] if src else None
        good = sum(1 for k in common if T[k]["spk"] == t and a is not None and A[k]["spk"] == a)
        per_spk.append({"truth": tname, "auto": aname(a) if a is not None else "(никто)",
                        "words": cnt_t.get(t, 0), "recall": good / max(1, cnt_t.get(t, 0)),
                        "name_ok": a is not None and aname(a) == tname})
    named = [p for p in per_spk if p["auto"] != "(никто)" and not sp.GENERIC.match(p["auto"])]

    truth_starts = {k for k in T if T[k]["first"]}
    auto_starts = {k for k in A if A[k]["first"]}
    conf, n_fixed, n_cmp, n_ok = {}, 0, 0, 0
    for tr in truth["turns"]:
        if not tr["words"]:
            continue
        key = (round(tr["words"][0]["s"], 2), round(tr["words"][0]["e"], 2))
        if tr.get("contour_fix"):
            n_fixed += 1
        if key not in A or not A[key]["first"]:
            continue
        al, tl = auto["turns"][A[key]["turn"]].get("contour", "?"), label_of(tr)
        if al == "?" or tl == "?":
            continue
        n_cmp += 1
        n_ok += al == tl
        conf[(al, tl)] = conf.get((al, tl), 0) + 1

    p = auto.get("params", {})
    return {
        "n_auto": len(auto_keys), "n_truth": len(truth_keys),
        "words_common": len(common), "words_edited": sum(1 for k in common if A[k]["w"] != T[k]["w"]),
        "words_dropped": len(A) - len(common),
        "word_acc": ok_words / max(1, len(common)), "time_err": bad_dur / total_dur,
        "per_spk": per_spk, "name_acc": (sum(p["name_ok"] for p in named) / len(named)) if named else None,
        "n_named": len(named),
        "turns_truth": len(truth["turns"]), "turns_auto": len(auto["turns"]),
        "missed_splits": len(truth_starts - auto_starts), "extra_splits": len(auto_starts - truth_starts),
        "contour_cmp": n_cmp, "contour_acc": (n_ok / n_cmp) if n_cmp else None, "contour_fixed": n_fixed, "conf": conf,
        "params": p,
    }


def pct(v):
    return "" if v is None else f"{v:.0%}"


def report(m: dict, stem: str) -> str:
    p = m["params"]
    rows = [f"## {datetime.now():%Y-%m-%d %H:%M} {stem}", "",
            f"Параметры: " + ", ".join(f"{k}={v}" for k, v in p.items()) if p else "Параметры: нет", "",
            "| метрика | значение | что значит |", "|---|---|---|",
            f"| людей: авто / по оператору | {m['n_auto']} / {m['n_truth']} | должно совпасть |",
            f"| слова с верным говорящим | {pct(m['word_acc'])} | доля слов, у которых автомат угадал человека |",
            f"| ошибка по времени речи | {pct(m['time_err'])} | доля секунд, отданных не тому человеку (аналог DER) |",
            f"| имена по голосу верны | {pct(m['name_acc'])} из {m['n_named']} | только для людей, узнанных из voices/ |",
            f"| реплики: авто / по оператору | {m['turns_auto']} / {m['turns_truth']} | |",
            f"| пропущенные границы | {m['missed_splits']} | оператор разрезал реплику, автомат не заметил смену |",
            f"| лишние границы | {m['extra_splits']} | автомат разрезал там, где оператор склеил |",
            f"| слова исправлены / удалены | {m['words_edited']} / {m['words_dropped']} | качество Whisper, не ролей |",
            f"| контур совпал | {pct(m['contour_acc'])} из {m['contour_cmp']} | исправлено оператором: {m['contour_fixed']} |"]
    if p.get("audio_sec"):
        a = p["audio_sec"]
        rows.append(f"| скорость | текст {p.get('text_sec', '?')} с, роли {p.get('diar_sec', '?')} с, интонация "
                    f"{p.get('prosody_sec', '?')} с | на {a:.0f} с записи, RTF ролей "
                    f"{(p.get('diar_sec') or 0) / a:.2f} |")
    rows += ["", "| человек (оператор) | кем назвал автомат | слов | найдено | имя верно |", "|---|---|---|---|---|"]
    for s in m["per_spk"]:
        rows.append(f"| {s['truth']} | {s['auto']} | {s['words']} | {pct(s['recall'])} | {'да' if s['name_ok'] else 'нет'} |")
    if m["conf"]:
        labels = ["подъём", "ровно", "падение"]
        rows += ["", "Контуры, строки = автомат, столбцы = оператор:", "",
                 "| авто \\ оператор | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for al in labels:
            rows.append(f"| {al} | " + " | ".join(str(m["conf"].get((al, tl), 0)) for tl in labels) + " |")
    if m["contour_fixed"] == 0:
        rows += ["", "Оператор не правил контуры, совпадение 100% ничего не говорит о качестве интонации."]
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------- перебор параметров диаризации
def sweep(truth: dict, media: Path, thresholds: list[float], clusters: list[int], threads: int) -> str:
    words = [{"start": w["s"], "end": w["e"], "word": w["w"]} for tr in truth["turns"] for w in tr["words"]]
    words.sort(key=lambda w: w["start"])
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        samples, sr = sf.read(str(wav), dtype="float32")
    rows = ["| вариант | людей (надо %d) | слова верно | ошибка по времени | границ пропущено / лишних | с |"
            % len({tr["spk"] for tr in truth["turns"]}), "|---|---|---|---|---|---|"]
    runs = [("threshold", t, -1) for t in thresholds] + [("clusters", c, c) for c in clusters]
    for kind, val, ncl in runs:
        t0 = time.time()
        segs = sp.diarize(samples, sr, threads, ncl, val if kind == "threshold" else 0.9)
        turns = sp.to_turns(sp.assign_words(words, segs))
        for tr in turns:
            tr.update(contour="?", final_slope=None, final_end=None, semi_mean=None, semi_range=None, loud_z=None, pauses=0)
        m = compare(sp.export(turns, {}, [], "", ""), truth)
        line = (f"| {kind}={val} | {m['n_auto']} | {pct(m['word_acc'])} | {pct(m['time_err'])} | "
                f"{m['missed_splits']} / {m['extra_splits']} | {time.time() - t0:.0f} |")
        print(line, flush=True)
        rows.append(line)
    return "\n".join(rows) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Метрики качества ролей и интонации против правок оператора")
    ap.add_argument("review", help="файл .review.json, сохранённый со страницы правок")
    ap.add_argument("--auto", help="файл .auto.json (по умолчанию ищется рядом)")
    ap.add_argument("--sweep", default="", help="пороги кластеризации через запятую, например 0.7,0.8,0.9,0.95")
    ap.add_argument("--sweep-clusters", default="", help="число людей через запятую, например 3,4,5")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-log", action="store_true", help="не дописывать calibration/bench.md")
    args = ap.parse_args()

    js = Path(args.review)
    truth = json.loads(js.read_text(encoding="utf-8"))
    stem = js.name[: -len(".review.json")] if js.name.endswith(".review.json") else js.stem
    out = []
    auto_p = Path(args.auto) if args.auto else js.with_name(stem + ".auto.json")
    if auto_p.exists():
        auto = json.loads(auto_p.read_text(encoding="utf-8"))
        out.append(report(compare(auto, truth), stem))
    else:
        print(f"Нет {auto_p.name}: его пишет transcribe.bat --speakers, сравнение авто/оператор пропущено")
    if args.sweep or args.sweep_clusters:
        media = js.with_name(truth.get("media") or "")
        if not media.exists():
            raise SystemExit(f"Запись не найдена рядом: {media}")
        th = [float(x) for x in args.sweep.split(",") if x.strip()]
        cl = [int(x) for x in args.sweep_clusters.split(",") if x.strip()]
        print(f"Перебор на {media.name}: каждый вариант = полная диаризация, это минуты")
        out.append(f"### перебор {datetime.now():%Y-%m-%d %H:%M} {stem}\n\n" + sweep(truth, media, th, cl, args.threads))
    text = "\n".join(out)
    print("\n" + text)
    if not args.no_log and text.strip():
        sp.CALIB.mkdir(exist_ok=True)
        with open(sp.CALIB / "bench.md", "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Записано в {sp.CALIB / 'bench.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

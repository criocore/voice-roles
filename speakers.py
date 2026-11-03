"""Роли (кто говорил) и интонация поверх faster-whisper. Всё на CPU через onnxruntime, без torch.

Диаризация: sherpa-onnx (сегментация pyannote-3.0 int8 + эмбеддер TitaNet-small), число людей
определяется само: порог кластеризации, потом короткие "призраки" (< min_total с речи) уходят к
ближайшему голосу по эмбеддингу, потом почти одинаковые голоса (косинус >= sim) сливаются.
Слова Whisper получают спикера по наибольшему перекрытию во времени.
Интонация: SwiftF0 даёт F0 и громкость по кадрам; высота считается в полутонах от медианы
конкретного человека, контур по наклону последних 300 мс реплики.

Калибровка оператором: страница .review.html (см. review_template.html) и apply_review.py.
Голоса, подтверждённые оператором, лежат в voices/<Имя>.npy и подписывают людей в следующих
прогонах; пороги контура берутся из calibration/contour_thresholds.json, если он есть.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models" / "diar"
SEG = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx"
EMB = MODELS / "nemo_en_titanet_small.onnx"
EMB_MAX_SEC = 60  # ONNX TitaNet не принимает вход длиннее ~2 минут
VOICES = HERE / "voices"
CALIB = HERE / "calibration"
TEMPLATE = HERE / "review_template.html"
GENERIC = re.compile(r"^Спикер \d+$")
DEFAULT_CONTOUR = {"k_hi": 8.0, "k_lo": 3.0, "end": 1.0}


def ts(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"


# ---------------------------------------------------------------- эмбеддинги голоса
def extractor(threads: int = 8):
    import sherpa_onnx

    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB), num_threads=threads))


def embed(ext, sr: int, chunks: list[np.ndarray]) -> np.ndarray:
    st = ext.create_stream()
    st.accept_waveform(sr, np.concatenate(chunks))
    st.input_finished()
    e = np.asarray(ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def span_embedding(ext, samples: np.ndarray, sr: int, spans: list[tuple[float, float]]) -> np.ndarray | None:
    """Один вектор голоса по отрезкам (start, end): самые длинные сначала, не больше EMB_MAX_SEC."""
    chunks, acc = [], 0.0
    for a, b in sorted(spans, key=lambda p: p[0] - p[1]):
        if b - a < 0.2:
            continue
        chunks.append(samples[int(a * sr): int(b * sr)])
        acc += b - a
        if acc > EMB_MAX_SEC:
            break
    return embed(ext, sr, chunks) if acc >= 1.0 else None


def load_profiles() -> dict[str, np.ndarray]:
    """Имя -> матрица (n, d) эмбеддингов, подтверждённых оператором."""
    out = {}
    if VOICES.is_dir():
        for p in sorted(VOICES.glob("*.npy")):
            out[p.stem] = np.atleast_2d(np.load(p))
    return out


def save_profile(name: str, emb: np.ndarray, keep: int = 8) -> int:
    VOICES.mkdir(exist_ok=True)
    p = VOICES / f"{name}.npy"
    old = np.atleast_2d(np.load(p)) if p.exists() else np.zeros((0, emb.shape[0]), dtype=np.float32)
    new = np.vstack([old, emb[None, :]])[-keep:]
    np.save(p, new)
    return len(new)


def match_profiles(refs: dict[int, np.ndarray], profiles: dict[str, np.ndarray], min_sim: float) -> dict[int, str]:
    """Жадное сопоставление спикер -> имя по лучшему косинусу; каждый не чаще одного раза."""
    cands = sorted(((float((prof @ e).max()), k, name) for k, e in refs.items() for name, prof in profiles.items()),
                   reverse=True)
    used_k, used_n, out = set(), set(), {}
    for score, k, name in cands:
        if score < min_sim:
            break
        if k in used_k or name in used_n:
            continue
        out[k] = name
        used_k.add(k)
        used_n.add(name)
    return out


# ---------------------------------------------------------------- диаризация
def diarize(samples: np.ndarray, sr: int, threads: int = 8, clusters: int = -1, threshold: float = 0.9,
            min_total: float = 15.0, sim: float = 0.8) -> list[dict]:
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=clusters, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    res = sherpa_onnx.OfflineSpeakerDiarization(cfg).process(samples).sort_by_start_time()
    segs = [{"start": float(s.start), "end": float(s.end), "spk": int(s.speaker)} for s in res]
    if clusters < 0:
        segs = _merge_clusters(segs, samples, sr, threads, min_total, sim)
    return _renumber(segs)


def _renumber(segs):
    """Номера по порядку первого появления. Считаем только реплики от 1 с, чтобы обрывок
    на наложении голосов не выдвигал человека вперёд и не путал порядок для --names."""
    order = {}
    for s in sorted(segs, key=lambda s: s["start"]):
        if s["end"] - s["start"] >= 1.0:
            order.setdefault(s["spk"], len(order))
    for s in sorted(segs, key=lambda s: s["start"]):
        order.setdefault(s["spk"], len(order))
    for s in segs:
        s["spk"] = order[s["spk"]]
    return segs


def speaker_embeddings(ext, samples, sr, segs, keys=None) -> dict[int, np.ndarray]:
    keys = sorted({s["spk"] for s in segs}) if keys is None else keys
    out = {}
    for k in keys:
        e = span_embedding(ext, samples, sr, [(s["start"], s["end"]) for s in segs if s["spk"] == k])
        if e is not None:
            out[k] = e
    return out


def _merge_clusters(segs, samples, sr, threads, min_total, sim):
    total: dict[int, float] = {}
    for s in segs:
        total[s["spk"]] = total.get(s["spk"], 0.0) + s["end"] - s["start"]
    big = [k for k, v in total.items() if v >= min_total]
    if not big:
        return segs
    ext = extractor(threads)
    ref = speaker_embeddings(ext, samples, sr, segs, big)

    # похожие крупные голоса -> один человек
    while len(big) > 1:
        score, a, b = max((float(ref[x] @ ref[y]), x, y) for i, x in enumerate(big) for y in big[i + 1:])
        if score < sim:
            break
        keep, drop = (a, b) if total[a] >= total[b] else (b, a)
        for s in segs:
            if s["spk"] == drop:
                s["spk"] = keep
        total[keep] += total.pop(drop)
        big.remove(drop)
        ref[keep] = speaker_embeddings(ext, samples, sr, segs, [keep])[keep]

    # призраки -> ближайший голос по эмбеддингу (совсем короткие: по времени)
    big_segs = [s for s in segs if s["spk"] in big]
    for s in segs:
        if s["spk"] in big:
            continue
        if s["end"] - s["start"] >= 0.2:
            e = embed(ext, sr, [samples[int(s["start"] * sr): int(s["end"] * sr)]])
            s["spk"] = max(big, key=lambda k: float(e @ ref[k]))
        else:
            near = min(big_segs, key=lambda x: min(abs(x["start"] - s["start"]), abs(x["end"] - s["end"])))
            s["spk"] = near["spk"]
    return segs


# ---------------------------------------------------------------- слова -> спикеры -> реплики
def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_words(words: list[dict], segs: list[dict]) -> list[dict]:
    out = []
    for w in words:
        score: dict[int, float] = {}
        for s in segs:
            o = _overlap(w["start"], w["end"], s["start"], s["end"])
            if o > 0:
                score[s["spk"]] = score.get(s["spk"], 0.0) + o
        if score:
            spk, sure = max(score, key=score.get), len(score) == 1
        else:
            mid = (w["start"] + w["end"]) / 2
            s = min(segs, key=lambda s: min(abs(s["start"] - mid), abs(s["end"] - mid)))
            spk, sure = s["spk"], False
        out.append({**w, "spk": spk, "sure": sure})
    # одиночное слово чужого спикера между словами одного человека считаем ошибкой границы
    for i in range(1, len(out) - 1):
        if out[i - 1]["spk"] == out[i + 1]["spk"] != out[i]["spk"]:
            out[i]["spk"], out[i]["sure"] = out[i - 1]["spk"], False
    return out


def to_turns(words: list[dict], max_gap: float = 1.0) -> list[dict]:
    turns: list[dict] = []
    for w in words:
        if turns and turns[-1]["spk"] == w["spk"] and w["start"] - turns[-1]["end"] <= max_gap:
            turns[-1]["end"] = w["end"]
            turns[-1]["words"].append(w)
        else:
            turns.append({"spk": w["spk"], "start": w["start"], "end": w["end"], "words": [w]})
    return turns


# ---------------------------------------------------------------- интонация
def contour_thresholds() -> dict:
    p = CALIB / "contour_thresholds.json"
    if p.exists():
        return {**DEFAULT_CONTOUR, **json.loads(p.read_text(encoding="utf-8"))}
    return dict(DEFAULT_CONTOUR)


def contour_label(k: float, end: float, th: dict | None = None) -> str:
    th = th or contour_thresholds()
    if k > th["k_hi"] or (k > th["k_lo"] and end > th["end"]):
        return "подъём"
    if k < -th["k_hi"] or (k < -th["k_lo"] and end < -th["end"]):
        return "падение"
    return "ровно"


def prosody(samples: np.ndarray, sr: int, turns: list[dict], threads: int = 8) -> dict:
    from swift_f0 import SwiftF0

    r = SwiftF0(threads=threads).detect(samples, sr, fmin=60, fmax=500)
    t, f0, loud = np.asarray(r.timestamps), np.asarray(r.pitch_hz), np.asarray(r.loudness_db)
    voiced = np.asarray(r.confidence) >= 0.5
    th = contour_thresholds()

    masks: dict[int, list] = {}
    for tr in turns:
        m = (t >= tr["start"]) & (t <= tr["end"]) & voiced
        tr["_m"] = m
        masks.setdefault(tr["spk"], []).append(m)
    base = {}
    for spk, ms in masks.items():
        m = np.logical_or.reduce(ms)
        base[spk] = None if m.sum() < 10 else {
            "f0_med": float(np.median(f0[m])), "loud_med": float(np.median(loud[m])),
            "loud_sd": float(np.std(loud[m]) + 1e-6)}

    for tr in turns:
        m, b = tr.pop("_m"), base.get(tr["spk"])
        tr.update(semi_mean=None, semi_range=None, final_slope=None, final_end=None, loud_z=None, contour="?")
        ws = tr["words"]
        gaps = [ws[i + 1]["start"] - ws[i]["end"] for i in range(len(ws) - 1)]
        tr["pauses"] = sum(1 for g in gaps if g > 0.25)
        if b is None or m.sum() < 5:
            continue
        semi = 12 * np.log2(f0[m] / b["f0_med"])
        tr["semi_mean"] = float(semi.mean())
        tr["semi_range"] = float(np.percentile(semi, 95) - np.percentile(semi, 5))
        tr["loud_z"] = float((loud[m].mean() - b["loud_med"]) / b["loud_sd"])
        tt = t[m]
        tail = tt >= tt[-1] - 0.3
        if tail.sum() >= 3:
            k = float(np.polyfit(tt[tail], semi[tail], 1)[0])  # полутоны в секунду
            end = float(semi[tail].mean())
            tr["final_slope"], tr["final_end"] = k, end
            tr["contour"] = contour_label(k, end, th)
    return base


# ---------------------------------------------------------------- вывод
def render(turns: list[dict], base: dict, names: list[str], title: str) -> tuple[str, str]:
    """Возвращает (speakers.txt, prosody.md)."""
    label = lambda k: names[k] if k < len(names) else f"Спикер {k + 1}"  # noqa: E731
    total: dict[int, float] = {}
    for tr in turns:
        total[tr["spk"]] = total.get(tr["spk"], 0.0) + tr["end"] - tr["start"]

    head = [f"# {title}", ""]
    for k in sorted(total):
        b = base.get(k)
        extra = f", голос {b['f0_med']:.0f} Гц" if b else ""
        head.append(f"- {label(k)}: {ts(total[k])} речи{extra}")

    txt = []
    for tr in turns:
        text = "".join(w["word"] for w in tr["words"]).strip()
        txt.append(f"[{ts(tr['start'])}] {label(tr['spk'])}: {text}")

    md = head + ["", "Контур: движение тона в последних 300 мс реплики. Высота: средняя высота реплики в полутонах "
                 "от медианы этого человека. Громкость: z-оценка относительно его нормы. Знак ? у имени: "
                 "столько слов на границе смены говорящего привязаны неуверенно.", "",
                 "| время | кто | контур | наклон, пт/с | высота, пт | громк. z | паузы | текст |",
                 "|---|---|---|---|---|---|---|---|"]
    for tr in turns:
        text = "".join(w["word"] for w in tr["words"]).strip().replace("|", "/")
        unsure = sum(1 for w in tr["words"] if not w.get("sure", True))
        who = label(tr["spk"]) + (f" ({unsure}?)" if unsure else "")
        f = lambda v, fmt="{:+.1f}": fmt.format(v) if v is not None else ""  # noqa: E731
        md.append(f"| {ts(tr['start'])} | {who} | {tr['contour']} | {f(tr['final_slope'])} | "
                  f"{f(tr['semi_mean'])} | {f(tr['loud_z'])} | {tr['pauses']} | {text} |")
    return "\n".join(head + ["", *txt]) + "\n", "\n".join(md) + "\n"


def export(turns: list[dict], base: dict, names: list[str], title: str, media: str, params: dict | None = None) -> dict:
    """Данные для страницы правок, apply_review.py и bench_quality.py."""
    n = max([tr["spk"] for tr in turns] + [len(names) - 1]) + 1
    speakers = []
    for k in range(n):
        b = base.get(k)
        speakers.append({"name": names[k] if k < len(names) else f"Спикер {k + 1}",
                         "f0": round(b["f0_med"]) if b else None})
    rnd = lambda v, d=2: None if v is None else round(v, d)  # noqa: E731
    out_turns = []
    for tr in turns:
        out_turns.append({
            "spk": tr["spk"], "start": rnd(tr["start"]), "end": rnd(tr["end"]),
            "contour": tr["contour"], "slope": rnd(tr["final_slope"], 1), "end_semi": rnd(tr["final_end"]),
            "semi": rnd(tr["semi_mean"], 1), "loud_z": rnd(tr["loud_z"]), "pauses": tr["pauses"],
            "contour_fix": None, "note": "",
            "words": [{"w": w["word"], "s": rnd(w["start"]), "e": rnd(w["end"]), "sure": bool(w.get("sure", True))}
                      for w in tr["words"]],
        })
    return {"title": title, "media": media, "version": 1, "params": params or {}, "speakers": speakers,
            "turns": out_turns}


def import_turns(data: dict) -> tuple[list[dict], dict, list[str]]:
    """Обратно из review.json: (turns для render, base, names). Правки оператора уже учтены."""
    turns = []
    for tr in data["turns"]:
        if not tr["words"]:
            continue
        words = [{"word": w["w"], "start": w["s"], "end": w["e"], "sure": w.get("sure", True)} for w in tr["words"]]
        turns.append({"spk": tr["spk"], "start": words[0]["start"], "end": words[-1]["end"], "words": words,
                      "contour": tr.get("contour_fix") or tr.get("contour", "?"),
                      "final_slope": tr.get("slope"), "final_end": tr.get("end_semi"), "semi_mean": tr.get("semi"),
                      "loud_z": tr.get("loud_z"), "pauses": tr.get("pauses", 0)})
    base = {k: ({"f0_med": s["f0"]} if s.get("f0") else None) for k, s in enumerate(data["speakers"])}
    names = [s["name"].strip() or f"Спикер {k + 1}" for k, s in enumerate(data["speakers"])]
    return turns, base, names


def review_html(data: dict) -> str:
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", blob, 1)


def run(wav: Path, words: list[dict], names: list[str], title: str, threads: int = 8, clusters: int = -1,
        threshold: float = 0.9, media: str = "", voice_sim: float = 0.6, params: dict | None = None) -> dict:
    """Полный проход: wav 16 кГц моно + слова Whisper -> {"txt", "md", "html", "data", "n", "named"}.
    Если --names не задан, имена берутся из voices/ по сходству голоса (>= voice_sim)."""
    import time

    samples, sr = sf.read(str(wav), dtype="float32")
    t0 = time.time()
    segs = diarize(samples, sr, threads, clusters, threshold)
    t_diar = time.time() - t0
    n = len({s["spk"] for s in segs})
    named = {}
    profiles = load_profiles()
    if not names and profiles:
        refs = speaker_embeddings(extractor(threads), samples, sr, segs)
        named = match_profiles(refs, profiles, voice_sim)
        names = [named.get(k, f"Спикер {k + 1}") for k in range(n)]
    turns = to_turns(assign_words(words, segs))
    t1 = time.time()
    base = prosody(samples, sr, turns, threads)
    p = {**(params or {}), "threshold": threshold, "clusters": clusters, "voice_sim": voice_sim, "threads": threads,
         "names_given": bool(params and params.get("names_given")), "audio_sec": round(len(samples) / sr, 1),
         "diar_sec": round(t_diar, 1), "prosody_sec": round(time.time() - t1, 1)}
    txt, md = render(turns, base, names, title)
    data = export(turns, base, names, title, media, p)
    return {"txt": txt, "md": md, "html": review_html(data), "data": data, "n": n, "named": named}

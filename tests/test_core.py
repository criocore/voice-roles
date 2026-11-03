"""Unit tests for the pure parts of the pipeline: no models, no audio, no network."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bench_quality as bq  # noqa: E402
import speakers as sp  # noqa: E402


def words(spec):
    """spec: list of (start, end, text)."""
    return [{"start": s, "end": e, "word": " " + w} for s, e, w in spec]


def test_assign_words_by_overlap_and_fixes_single_stray_word():
    segs = [{"start": 0.0, "end": 2.0, "spk": 0}, {"start": 2.0, "end": 4.0, "spk": 1}]
    ws = words([(0.1, 0.5, "a"), (0.6, 1.0, "b"), (1.9, 2.3, "c"), (2.5, 3.0, "d"), (3.1, 3.5, "e")])
    out = sp.assign_words(ws, segs)
    assert [w["spk"] for w in out] == [0, 0, 1, 1, 1]
    assert out[2]["sure"] is False  # overlaps both speakers
    # a lone word of speaker 1 between two words of speaker 0 is treated as a boundary error
    segs2 = [{"start": 0.0, "end": 0.55, "spk": 0}, {"start": 0.55, "end": 1.05, "spk": 1},
             {"start": 1.05, "end": 3.0, "spk": 0}]
    out2 = sp.assign_words(words([(0.1, 0.5, "a"), (0.6, 1.0, "b"), (1.1, 1.5, "c")]), segs2)
    assert [w["spk"] for w in out2] == [0, 0, 0]
    assert out2[1]["sure"] is False


def test_assign_words_outside_segments_takes_nearest():
    segs = [{"start": 0.0, "end": 1.0, "spk": 0}, {"start": 5.0, "end": 6.0, "spk": 1}]
    out = sp.assign_words(words([(4.6, 4.8, "x")]), segs)
    assert out[0]["spk"] == 1 and out[0]["sure"] is False


def test_to_turns_merges_same_speaker_within_gap():
    ws = [{"start": 0, "end": 1, "word": " a", "spk": 0}, {"start": 1.5, "end": 2, "word": " b", "spk": 0},
          {"start": 3.5, "end": 4, "word": " c", "spk": 0}, {"start": 4, "end": 5, "word": " d", "spk": 1}]
    turns = sp.to_turns(ws, max_gap=1.0)
    assert [(t["spk"], len(t["words"])) for t in turns] == [(0, 2), (0, 1), (1, 1)]
    assert turns[0]["end"] == 2


def test_renumber_orders_by_first_long_segment():
    segs = [{"start": 0.0, "end": 0.4, "spk": 7},  # short fragment must not win the first slot
            {"start": 1.0, "end": 5.0, "spk": 3}, {"start": 6.0, "end": 9.0, "spk": 7}]
    out = sp._renumber([dict(s) for s in segs])
    assert [s["spk"] for s in out] == [1, 0, 1]


def test_contour_label_thresholds():
    th = {"k_hi": 8.0, "k_lo": 3.0, "end": 1.0}
    assert sp.contour_label(9, 0, th) == "подъём"
    assert sp.contour_label(4, 2, th) == "подъём"
    assert sp.contour_label(4, 0, th) == "ровно"
    assert sp.contour_label(-9, 0, th) == "падение"
    assert sp.contour_label(-4, -2, th) == "падение"
    assert sp.contour_label(0, 0, th) == "ровно"


def test_match_profiles_is_greedy_and_unique():
    e = {0: np.array([1.0, 0.0]), 1: np.array([0.9, 0.1]), 2: np.array([0.0, 1.0])}
    prof = {"Анна": np.array([[1.0, 0.0]]), "Вера": np.array([[0.0, 1.0], [0.1, 0.9]])}
    m = sp.match_profiles(e, prof, min_sim=0.6)
    assert m == {0: "Анна", 2: "Вера"}  # speaker 1 is close to Анна but she is taken
    assert sp.match_profiles(e, prof, min_sim=0.999) == {0: "Анна", 2: "Вера"}
    assert sp.match_profiles({0: np.array([0.7, 0.7])}, prof, min_sim=0.8) == {}


def _turns():
    ws = sp.assign_words(words([(0.0, 0.5, "Привет"), (0.6, 1.0, "всем"), (2.0, 2.4, "Добрый"), (2.5, 3.0, "день")]),
                         [{"start": 0.0, "end": 1.2, "spk": 0}, {"start": 1.8, "end": 3.2, "spk": 1}])
    turns = sp.to_turns(ws)
    for t in turns:
        t.update(contour="ровно", final_slope=0.5, final_end=0.1, semi_mean=0.2, semi_range=1.0, loud_z=0.0, pauses=0)
    return turns


def test_render_and_export_roundtrip():
    turns = _turns()
    base = {0: {"f0_med": 120.0}, 1: {"f0_med": 200.0}}
    txt, md = sp.render(turns, base, ["Анна"], "Тест")
    assert "[00:00:00] Анна: Привет всем" in txt and "Спикер 2: Добрый день" in txt
    assert "| 00:00:02 | Спикер 2 | ровно |" in md
    data = sp.export(turns, base, ["Анна"], "Тест", "test.mp4", {"threshold": 0.9})
    assert data["speakers"] == [{"name": "Анна", "f0": 120}, {"name": "Спикер 2", "f0": 200}]
    back, base2, names = sp.import_turns(data)
    assert names == ["Анна", "Спикер 2"] and base2[1] == {"f0_med": 200}
    assert "".join(w["word"] for w in back[1]["words"]).strip() == "Добрый день"
    html = sp.review_html(data)
    assert '"title": "Тест"' in html and "/*DATA*/null" not in html


def test_review_html_escapes_script_close():
    turns = _turns()
    turns[0]["words"][0]["word"] = " </script><b>"
    html = sp.review_html(sp.export(turns, {}, [], "x", ""))
    assert "</script><b>" not in html.split("const EMBEDDED")[1].split("\n")[0]


def test_bench_compare_counts_edits():
    turns = _turns()
    auto = sp.export(turns, {}, [], "x", "")
    truth = json.loads(json.dumps(auto))
    truth["speakers"] = [{"name": "Анна", "f0": None}, {"name": "Борис", "f0": None}]
    truth["turns"][1]["spk"] = 0                     # operator reassigned the second turn
    truth["turns"][0]["contour_fix"] = "подъём"      # and fixed one contour
    m = bq.compare(auto, truth)
    assert m["n_auto"] == 2 and m["n_truth"] == 1
    assert m["word_acc"] == 0.5 and m["words_edited"] == 0
    assert m["contour_cmp"] == 2 and m["contour_acc"] == 0.5 and m["contour_fixed"] == 1
    # a split by the operator shows up as a missed boundary
    t2 = json.loads(json.dumps(auto))
    tr = t2["turns"][1]
    t2["turns"].append({**tr, "words": tr["words"][1:], "spk": 1})
    tr["words"] = tr["words"][:1]
    m2 = bq.compare(auto, t2)
    assert m2["missed_splits"] == 1 and m2["extra_splits"] == 0 and m2["word_acc"] == 1.0
    assert "| слова с верным говорящим | 100% |" in bq.report(m2, "x")

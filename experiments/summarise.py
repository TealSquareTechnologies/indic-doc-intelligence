"""Prints the summary numbers used in the report and writes latency_summary.json."""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LANGUAGES, RESULTS_DIR, load_json, save_json  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

LANGS = list(LANGUAGES)
CONDITIONS = ["clean", "small", "noisy", "skewed", "jpeg", "photo"]


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else float("nan")


def median(values):
    values = list(values)
    return statistics.median(values) if values else float("nan")


def exists(name):
    return (RESULTS_DIR / name).exists()


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def exp1():
    if not exists("exp1_ocr.json"):
        return
    fixed = load_json("exp1_ocr.json")
    shipped = load_json("exp1_ocr_shipped.json") if exists("exp1_ocr_shipped.json") else []
    section("Experiment 1: CER (%) by language x condition, preparation on (deskew fixed)")
    print(f"{'lang':6}" + "".join(f"{c:>9}" for c in CONDITIONS) + f"{'mean':>9}")
    for code in LANGS:
        vals = [100 * mean(r["cer"] for r in fixed if r["code"] == code and r["condition"] == c and r["preprocess"])
                for c in CONDITIONS]
        print(f"{code:6}" + "".join(f"{v:9.1f}" for v in vals) + f"{mean(vals):9.1f}")
    print("\nWER (%) same grid")
    for code in LANGS:
        vals = [100 * mean(r["wer"] for r in fixed if r["code"] == code and r["condition"] == c and r["preprocess"])
                for c in CONDITIONS]
        print(f"{code:6}" + "".join(f"{v:9.1f}" for v in vals) + f"{mean(vals):9.1f}")
    print("\nMean CER (%) by condition: no prep / prep shipped / prep fixed")
    for c in CONDITIONS:
        no = 100 * mean(r["cer"] for r in shipped if r["condition"] == c and not r["preprocess"]) if shipped else float("nan")
        sh = 100 * mean(r["cer"] for r in shipped if r["condition"] == c and r["preprocess"]) if shipped else float("nan")
        fx = 100 * mean(r["cer"] for r in fixed if r["condition"] == c and r["preprocess"])
        print(f"{c:8} {no:7.1f} {sh:7.1f} {fx:7.1f}")
    print("\nOverall mean CER: no prep %.1f, shipped %.1f, fixed %.1f" % (
        100 * mean(r["cer"] for r in shipped if not r["preprocess"]) if shipped else float("nan"),
        100 * mean(r["cer"] for r in shipped if r["preprocess"]) if shipped else float("nan"),
        100 * mean(r["cer"] for r in fixed if r["preprocess"])))
    print("\nSeconds per page: no prep %.2f, prep %.2f (fixed run)" % (
        mean(r["seconds"] for r in shipped if not r["preprocess"]) if shipped else float("nan"),
        mean(r["seconds"] for r in fixed if r["preprocess"])))
    import numpy as np
    xs = [r["confidence"] for r in fixed if r["preprocess"]]
    ys = [r["cer"] for r in fixed if r["preprocess"]]
    print("Confidence vs CER Pearson r = %.3f" % np.corrcoef(xs, ys)[0, 1])
    # Pages where confidence is high but CER is bad
    bad = [(r["code"], r["condition"], r["confidence"], r["cer"]) for r in fixed if r["preprocess"] and r["confidence"] > 0.8 and r["cer"] > 0.2]
    print("High-confidence (>0.8) pages with CER > 20%:", len(bad), bad[:5])
    print("Latin vs Indic clean CER: eng %.2f, indic mean %.2f" % (
        100 * mean(r["cer"] for r in fixed if r["code"] == "eng" and r["condition"] == "clean" and r["preprocess"]),
        100 * mean(r["cer"] for r in fixed if r["code"] != "eng" and r["condition"] == "clean" and r["preprocess"])))


def exp2():
    if exists("exp2_detection_a.json"):
        section("Experiment 2a: detection on ground truth")
        for r in load_json("exp2_detection_a.json"):
            print(r["code"], r["top1_accuracy"], r["secondary_scripts"])
    if exists("exp2_detection_b.json"):
        section("Experiment 2b: detection on recognised text (prep on)")
        rows = load_json("exp2_detection_b.json")
        for c in CONDITIONS:
            sub = [r for r in rows if r["condition"] == c]
            print(f"{c:8} top1 correct {100*mean(r['correct'] for r in sub):5.1f}%  mean top1 share {mean(r['top1_share'] for r in sub):.3f}")
        wrong = [(r["code"], r["condition"], r["top1"], round(r["cer"], 2)) for r in rows if not r["correct"]]
        print("wrong:", wrong)
    if exists("exp2_detection_c.json"):
        section("Experiment 2c: mixed scripts")
        rows = load_json("exp2_detection_c.json")
        for ratio in sorted({r["ratio"] for r in rows}):
            sub = [r for r in rows if r["ratio"] == ratio]
            both = mean(len(r["detected"]) == 2 for r in sub if 0 < ratio < 1)
            print(f"ratio {ratio:.2f} reported secondary share {mean(r['share_secondary'] for r in sub):.3f}  both reported {both}")
    if exists("exp2_languages_d.json"):
        section("Experiment 2d: language strings (clean pages)")
        rows = load_json("exp2_languages_d.json")
        for cfg in ["oracle", "default", "all_ten", "two_pass", "two_pass_all"]:
            sub = [r for r in rows if r["config"] == cfg]
            print(f"{cfg:9} CER {100*mean(r['cer'] for r in sub):6.1f}%  seconds {mean(r['seconds'] for r in sub):5.2f}")
        print("per-language default-string CER:")
        for code in LANGS:
            sub = [r for r in rows if r["code"] == code]
            print(f"  {code}: " + "  ".join(f"{cfg} {100*mean(r['cer'] for r in sub if r['config']==cfg):.1f}" for cfg in ["oracle", "default", "all_ten", "two_pass", "two_pass_all"])
                  + "  suggested=" + str({r["language_string"] for r in sub if r["config"] == "two_pass"}))


def exp3():
    if not exists("exp3_correction.json"):
        return
    section("Experiment 3: correction")
    rows = load_json("exp3_correction.json")
    for c in ["clean", "noisy", "small"]:
        sub = [r for r in rows if r["condition"] == c]
        print(f"{c:6} CER before {100*mean(r['cer_before'] for r in sub):5.1f} after {100*mean(r['cer_after'] for r in sub):5.1f}"
              f"  improved on {sum(r['cer_after'] < r['cer_before'] for r in sub)}/{len(sub)} pages"
              f"  seconds {mean(r['seconds'] for r in sub):.1f}")
    print("per language (all conditions): before -> after")
    for code in LANGS:
        sub = [r for r in rows if r["code"] == code]
        print(f"  {code}: {100*mean(r['cer_before'] for r in sub):5.1f} -> {100*mean(r['cer_after'] for r in sub):5.1f}"
              f"  length ratio {mean(r['chars_out']/max(r['chars_in'],1) for r in sub):.2f}")
    print("applied:", sum(r["applied"] for r in rows), "/", len(rows))


def exp4():
    if not exists("exp4_translation.json"):
        return
    section("Experiment 4: translation")
    rows = load_json("exp4_translation.json")
    from sacrebleu.metrics import BLEU, CHRF
    for direction in sorted({r["direction"] for r in rows}):
        sub = [r for r in rows if r["direction"] == direction and r["applied"]]
        hyps = [r["hypothesis"] for r in sub]
        refs = [[r["reference"] for r in sub]]
        bleu = BLEU(tokenize="none" if not direction.endswith("eng") else "13a").corpus_score(hyps, refs).score
        chrf = CHRF().corpus_score(hyps, refs).score
        print(f"{direction:9} n={len(sub):3} sentence chrF mean {mean(r['chrf'] for r in sub):5.1f}  corpus chrF {chrf:5.1f}  BLEU {bleu:5.1f}  seconds {mean(r['seconds'] for r in sub):.1f}"
              f"  empty/unapplied {sum(not r['applied'] or not r['hypothesis'] for r in rows if r['direction']==direction)}")
    low = sorted(rows, key=lambda r: r["chrf"])[:3]
    for r in low:
        print("LOW", r["direction"], round(r["chrf"], 1), "|", r["hypothesis"][:120], "|| REF:", r["reference"][:120])


def exp5():
    if exists("exp5_a_chunking.json"):
        section("Experiment 5a: chunking")
        for r in load_json("exp5_a_chunking.json"):
            print(r)
    if exists("exp5_b_crosslingual.json"):
        section("Experiment 5b: cross-lingual")
        for r in load_json("exp5_b_crosslingual.json"):
            print(r)
    if exists("exp5_c_scores.json"):
        section("Experiment 5c: refusal scores")
        rows = load_json("exp5_c_scores.json")
        for g in ["in_corpus", "similar_news", "general"]:
            s = [r["best_score"] for r in rows if r["group"] == g]
            print(f"{g:13} n={len(s)} mean {mean(s):.3f} median {median(s):.3f} min {min(s):.3f} max {max(s):.3f}"
                  f"  below 0.20: {sum(v < 0.2 for v in s)}  below 0.30: {sum(v < 0.3 for v in s)}  below 0.40: {sum(v < 0.4 for v in s)}")
    if exists("exp5_d_answers.json"):
        section("Experiment 5d: answers")
        rows = load_json("exp5_d_answers.json")
        for g in ["in_corpus", "similar_news", "general"]:
            sub = [r for r in rows if r["group"] == g]
            print(f"{g:13} n={len(sub)} refused {sum(r['refused'] for r in sub)}  contains_gold {sum(r['contains_gold'] for r in sub)}"
                  f"  judge_yes {sum(bool(r['judge_correct']) for r in sub)}  cited {sum(r['cited'] for r in sub)}  seconds {mean(r['seconds'] for r in sub):.1f}")
        wrong = [r for r in rows if r["group"] == "in_corpus" and not r["refused"] and not r["judge_correct"] and not r["contains_gold"]]
        for r in wrong[:6]:
            print("WRONG:", r["question"], "| gold:", r["gold"], "| got:", r["answer"][:100])
        refused_in = [r for r in rows if r["group"] == "in_corpus" and r["refused"]]
        for r in refused_in[:6]:
            print("REFUSED:", r["question"], "| score", r["best_score"], "| got:", r["answer"][:80])
        leaked = [r for r in rows if r["group"] != "in_corpus" and not r["refused"]]
        for r in leaked[:8]:
            print("LEAK:", r["group"], r["question"], "| score", r["best_score"], "| got:", r["answer"][:100])


def exp6():
    if not exists("exp6_photos.json"):
        return
    section("Experiment 6: photos")
    for r in load_json("exp6_photos.json"):
        print(r["photo"], "prep", r["preprocess"], "conf", r["confidence"], "detected", r["detected"],
              "chars", len(r["raw_text"]), "seconds", r["seconds"], "notes", r["notes"])
        print("  RAW:", r["raw_text"][:300].replace("\n", " / "))
        print("  CORRECTED:", r["corrected_text"][:300].replace("\n", " / "))
        print("  TRANSLATED:", (r["translated_text"] or "")[:400].replace("\n", " / "))


def latency():
    stages = {}
    if exists("exp1_ocr.json"):
        fixed = load_json("exp1_ocr.json")
        stages["Recognition, no preparation"] = mean(r["seconds"] for r in load_json("exp1_ocr_shipped.json") if not r["preprocess"]) if exists("exp1_ocr_shipped.json") else float("nan")
        stages["Recognition with preparation"] = mean(r["seconds"] for r in fixed if r["preprocess"])
    if exists("exp3_correction.json"):
        stages["Correction (model, one page)"] = mean(r["seconds"] for r in load_json("exp3_correction.json"))
    if exists("exp4_translation.json"):
        stages["Translation (model, one sentence)"] = mean(r["seconds"] for r in load_json("exp4_translation.json"))
    if exists("exp5_d_answers.json"):
        rows = load_json("exp5_d_answers.json")
        stages["Answer, retrieval only (refused)"] = mean(r["seconds"] for r in rows if r["refused"] and r["best_score"] < 0.2) if any(r["refused"] and r["best_score"] < 0.2 for r in rows) else float("nan")
        stages["Answer with the model"] = mean(r["seconds"] for r in rows if not r["refused"])
    if exists("exp_stage_timing.json"):
        stages.update(load_json("exp_stage_timing.json"))
    stages = {k: v for k, v in stages.items() if v == v}
    save_json("latency_summary.json", stages)
    section("Latency summary (seconds)")
    for k, v in stages.items():
        print(f"{k:42} {v:8.3f}")


if __name__ == "__main__":
    exp1(); exp2(); exp3(); exp4(); exp5(); exp6(); latency()

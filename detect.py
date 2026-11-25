"""CLI: python detect.py "your prompt here" [--json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from features import all_features, heuristic_features, LM_FEATURE_NAMES

HERE = Path(__file__).parent
DEFAULT_MODEL = HERE / "detector.joblib"
MIN_WORDS = 15


def load_detector(path: str | Path = DEFAULT_MODEL) -> dict:
    bundle = joblib.load(path)
    bundle["uses_lm"] = any(n in LM_FEATURE_NAMES for n in bundle["feature_names"])
    bundle["scorer"] = None
    if bundle["uses_lm"]:
        from features import LMScorer
        bundle["scorer"] = LMScorer()
    return bundle


def explain(f: dict) -> list[tuple[str, str]]:
    """Return (direction, reason) pairs from simple, readable rules."""
    out: list[tuple[str, str]] = []
    S, H = "synthetic", "human"
    if f["stock_phrase_count"] >= 1:
        out.append((S, f"{f['stock_phrase_count']} stock phrase(s) typical of LLM prompts"))
    if f["bullet_lines"] >= 2:
        out.append((S, f"{f['bullet_lines']} bulleted/numbered lines (structured)"))
    if f["em_dash_rate"] > 0:
        out.append((S, "uses em dashes"))
    if f["colon_rate"] > 0.03:
        out.append((S, "frequent colons (structured formatting)"))
    if f["avg_word_len"] > 5.2:
        out.append((S, f"long average word length ({f['avg_word_len']:.1f})"))
    if f["informal_count"] >= 1:
        out.append((H, f"{f['informal_count']} informal word(s) (pls/u/thx/idk...)"))
    if f["starts_lowercase"]:
        out.append((H, "starts with a lowercase letter"))
    if f["no_end_punct"]:
        out.append((H, "no terminal punctuation"))
    if f["burstiness"] > 0.6:
        out.append((H, f"uneven sentence lengths (burstiness {f['burstiness']:.2f})"))
    if "perplexity" in f:
        if f["perplexity"] and f["perplexity"] < 20:
            out.append((S, f"low GPT-2 perplexity ({f['perplexity']:.1f})"))
        elif f["perplexity"] > 80:
            out.append((H, f"high GPT-2 perplexity ({f['perplexity']:.1f})"))
        if f["top10_frac"] > 0.7:
            out.append((S, f"{f['top10_frac']:.0%} of tokens in GPT-2's top-10 predictions"))
    return out


def detect(text: str, bundle: dict) -> dict:
    feats = all_features(text, bundle["scorer"])
    x = np.array([[feats[n] for n in bundle["feature_names"]]], dtype=float)
    p = float(bundle["model"].predict_proba(x)[0, 1])
    n_words = int(heuristic_features(text)["length_words"])
    if n_words < MIN_WORDS:
        verdict = "uncertain (too short)"
    elif p > 0.7:
        verdict = "synthetic"
    elif p < 0.3:
        verdict = "human"
    else:
        verdict = "uncertain"
    reasons = explain(feats)
    # Show the reasons supporting the verdict first.
    lean = "synthetic" if p >= 0.5 else "human"
    reasons.sort(key=lambda r: r[0] != lean)
    return {
        "probability_synthetic": round(p, 4),
        "verdict": verdict,
        "word_count": n_words,
        "signals": [{"points_to": d, "reason": r} for d, r in reasons],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify a prompt as human or synthetic.")
    ap.add_argument("prompt")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}. Run train.py first.", file=sys.stderr)
        return 1
    res = detect(args.prompt, load_detector(args.model))
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"probability_synthetic: {res['probability_synthetic']:.3f}")
    print(f"verdict: {res['verdict']}")
    print("top signals:")
    if not res["signals"]:
        print("  (none)")
    for s in res["signals"][:6]:
        print(f"  [{s['points_to']}] {s['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

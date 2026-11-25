"""Train the detector.

Usage:
    python train.py            # heuristics + GPT-2 features
    python train.py --no-lm    # heuristics only (fast, no torch needed)

Data files hold one prompt per line. A literal "\\n" inside a line is turned into a
real newline so multi-line (e.g. bulleted) prompts fit on one line.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from features import HEURISTIC_FEATURE_NAMES, LM_FEATURE_NAMES, LMScorer, all_features

HERE = Path(__file__).parent


def load_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip().replace("\\n", "\n") for ln in f if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lm", action="store_true", help="heuristic features only")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "detector.joblib"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    human = load_lines(data_dir / "human.txt")
    synth = load_lines(data_dir / "synthetic.txt")
    texts = human + synth
    y = np.array([0] * len(human) + [1] * len(synth))
    print(f"Loaded {len(human)} human and {len(synth)} synthetic prompts.")

    scorer = None if args.no_lm else LMScorer()
    names = list(HEURISTIC_FEATURE_NAMES) + ([] if args.no_lm else LM_FEATURE_NAMES)

    rows = []
    for i, t in enumerate(texts, 1):
        f = all_features(t, scorer)
        rows.append([f[n] for n in names])
        if scorer and i % 25 == 0:
            print(f"  scored {i}/{len(texts)}")
    X = np.array(rows, dtype=float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    model = GradientBoostingClassifier(random_state=args.seed)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("\nHoldout (20%, stratified):")
    print(classification_report(y_te, pred, target_names=["human", "synthetic"], zero_division=0))

    joblib.dump({"model": model, "feature_names": names}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

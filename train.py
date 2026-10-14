import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.dataset import load_dataset
from src.models import get_models
from src.features import GROUPS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", default="data/adress/audio")
    p.add_argument("--labels",    default="data/adress/labels.csv")
    p.add_argument("--n_folds",   type=int, default=5)
    p.add_argument("--out_dir",   default="results")
    p.add_argument("--cv", choices=["group", "naive"], default="group",
                   help="group: all recordings of a speaker stay in one fold (default). "
                        "naive: split by recording, leaks speakers across folds.")
    p.add_argument("--n_jobs",    type=int, default=1, help="parallel workers for feature extraction")
    return p.parse_args()


def cv_score(X, y, clf, n_folds, groups=None):
    if groups is None:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    acc, f1, auc = [], [], []
    for tr, te in cv.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        prob = pipe.predict_proba(X[te])[:, 1]
        acc.append(accuracy_score(y[te], pred))
        f1.append(f1_score(y[te], pred, zero_division=0))
        auc.append(roc_auc_score(y[te], prob))
    return np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(auc), np.std(auc)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X_groups, y, _, speakers = load_dataset(args.audio_dir, args.labels,
                                            return_speakers=True, n_jobs=args.n_jobs)
    groups = speakers if args.cv == "group" else None
    print(f"{len(y)} samples  |  AD {y.sum()}  HC {(y==0).sum()}  |  "
          f"{len(set(speakers))} speakers  |  cv={args.cv}\n")

    records = []

    # feature ablation — SVM fixed
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    for group in GROUPS:
        acc, acc_s, f1, f1_s, auc, auc_s = cv_score(X_groups[group], y, svm, args.n_folds, groups)
        records.append({"exp": "ablation", "variant": group, "model": "SVM",
                        "acc": acc, "acc_std": acc_s, "f1": f1, "f1_std": f1_s, "auc": auc, "auc_std": auc_s})
        print(f"[ablation] {group:8s}  acc {acc:.3f}±{acc_s:.3f}  f1 {f1:.3f}  auc {auc:.3f}")

    print()

    # model comparison — all features
    for name, clf in get_models().items():
        acc, acc_s, f1, f1_s, auc, auc_s = cv_score(X_groups["all"], y, clf, args.n_folds, groups)
        records.append({"exp": "models", "variant": "all", "model": name,
                        "acc": acc, "acc_std": acc_s, "f1": f1, "f1_std": f1_s, "auc": auc, "auc_std": auc_s})
        print(f"[models]   {name:8s}  acc {acc:.3f}±{acc_s:.3f}  f1 {f1:.3f}  auc {auc:.3f}")

    out = os.path.join(args.out_dir, f"scores_{args.cv}.csv" if args.cv != "group" else "scores.csv")
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.features import extract, GROUPS

LABEL_MAP = {"AD": 1, "HC": 0}


def load_dataset(audio_dir: str, labels_csv: str):
    df = pd.read_csv(labels_csv)
    rows = {g: [] for g in GROUPS}
    y, ids = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting features"):
        wav_path = os.path.join(audio_dir, row["uttid"])
        if not os.path.exists(wav_path):
            print(f"  skip: {wav_path}")
            continue
        feats = extract(wav_path)
        for g in GROUPS:
            rows[g].append(feats[g])
        y.append(LABEL_MAP[row["label"]])
        ids.append(row["uttid"])

    X_groups = {g: np.array(rows[g]) for g in GROUPS}
    return X_groups, np.array(y), ids

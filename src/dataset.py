import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from src.features import extract, GROUPS

LABEL_MAP = {"AD": 1, "HC": 0}


def load_dataset(audio_dir: str, labels_csv: str, return_speakers: bool = False, n_jobs: int = 1):
    """Extract features for every row of labels.csv.

    labels.csv needs `uttid,label`; an optional `speaker` column is returned
    when return_speakers=True so that CV can keep one speaker's recordings together.
    """
    df = pd.read_csv(labels_csv)
    keep = []
    for i, row in df.iterrows():
        wav_path = os.path.join(audio_dir, row["uttid"])
        if os.path.exists(wav_path):
            keep.append(i)
        else:
            print(f"  skip: {wav_path}")
    df = df.loc[keep].reset_index(drop=True)

    paths = [os.path.join(audio_dir, u) for u in df["uttid"]]
    feats = Parallel(n_jobs=n_jobs)(
        delayed(extract)(p) for p in tqdm(paths, desc="extracting features"))

    X_groups = {g: np.array([f[g] for f in feats]) for g in GROUPS}
    y = df["label"].map(LABEL_MAP).to_numpy()
    ids = df["uttid"].tolist()

    if return_speakers:
        speakers = df["speaker"].astype(str).to_numpy() if "speaker" in df else np.array(ids)
        return X_groups, y, ids, speakers
    return X_groups, y, ids

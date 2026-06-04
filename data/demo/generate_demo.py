"""
Generate synthetic audio files for pipeline testing.
These are NOT real speech — just noise signals to verify the code runs.
"""
import struct
import numpy as np
import os
import csv


def _write_wav(path, signal, sr):
    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16)
    n_samples = len(pcm)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n_samples * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n_samples * 2))
        f.write(pcm.tobytes())


SR = 16000
DURATION = 8
N_AD = 10
N_HC = 10
SEED = 42
rng = np.random.default_rng(SEED)

out_dir = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(out_dir, exist_ok=True)

rows = []
for i in range(N_AD):
    # AD: slower, lower-energy signal
    t = np.linspace(0, DURATION, SR * DURATION)
    sig = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.1 * rng.standard_normal(len(t))
    fname = f"AD_{i:03d}.wav"
    _write_wav(os.path.join(out_dir, fname), sig, SR)
    rows.append({"uttid": fname, "label": "AD"})

for i in range(N_HC):
    # HC: richer, higher-energy signal
    t = np.linspace(0, DURATION, SR * DURATION)
    sig = 0.6 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t) + 0.05 * rng.standard_normal(len(t))
    fname = f"HC_{i:03d}.wav"
    _write_wav(os.path.join(out_dir, fname), sig, SR)
    rows.append({"uttid": fname, "label": "HC"})

labels_path = os.path.join(os.path.dirname(__file__), "labels.csv")
with open(labels_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["uttid", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} files → {out_dir}")
print(f"Labels → {labels_path}")

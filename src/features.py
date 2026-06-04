import numpy as np
import librosa

N_MFCC = 40
SR = 16000
HOP = 512
FMIN = 75
FMAX = 400

GROUPS = ("mfcc", "pitch", "energy", "all")


def extract(wav_path: str) -> dict[str, np.ndarray]:
    y, _ = librosa.load(wav_path, sr=SR, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP)
    delta = librosa.feature.delta(mfcc)
    mfcc_vec = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
    ])

    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, hop_length=HOP)
    f0_v = f0[voiced] if voiced is not None and voiced.any() else np.array([0.0])
    pitch_vec = np.array([f0_v.mean(), f0_v.std(), f0_v.min(), f0_v.max(), f0_v.max() - f0_v.min()])

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    speech_rate = float(voiced.mean()) if voiced is not None and voiced.size > 0 else 0.0
    energy_vec = np.array([rms.mean(), rms.std(), speech_rate])

    feats = {
        "mfcc":   mfcc_vec.astype(np.float32),
        "pitch":  pitch_vec.astype(np.float32),
        "energy": energy_vec.astype(np.float32),
    }
    feats["all"] = np.concatenate([feats["mfcc"], feats["pitch"], feats["energy"]])
    return feats

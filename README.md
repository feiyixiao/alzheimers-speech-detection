# Cognitive Decline Detection from Speech

Binary classification of Alzheimer's Disease (AD) vs. Healthy Control (HC) from picture-description speech recordings, using handcrafted acoustic features and four classical classifiers.

This project is part of a literature review on automated cognitive decline screening (Cookie Theft task, DementiaBank corpus).

---

## Method

```
raw .wav
    ↓
Acoustic feature extraction (librosa)
    MFCC × 40 (mean + std)
    Delta MFCC (mean + std)
    Pitch F0 (mean, std, min, max, range)
    RMS energy (mean, std)
    Speech rate (voiced frame ratio)
    → 168-dim feature vector per utterance
    ↓
StandardScaler
    ↓
4 classifiers compared:
    Logistic Regression (baseline)
    SVM with RBF kernel
    Random Forest
    MLP
    ↓
5-fold speaker-grouped cross-validation → Accuracy / F1 / AUC
```

This approach follows the feature-based tradition established by Fraser et al. (2016), who achieved 81.9% accuracy on DementiaBank using handcrafted linguistic and acoustic features.

---

## Data

This project is designed for the **ADReSS 2020** dataset (Luz et al., 2020), a balanced subset of the DementiaBank Pitt Corpus (156 sessions, Cookie Theft picture description task).

**Accessing the data:**
1. Go to [talkbank.org](https://talkbank.org/dementia/index.html)
2. Register as a consortium member (free, academic use)
3. Note: student access requires a faculty advisor to submit the membership request
4. Once approved, place audio files and a `labels.csv` in `data/adress/`

**`labels.csv` format:**
```
uttid,label
S001.wav,AD
S002.wav,HC
...
```

**Demo data** (no application needed):
```bash
python data/demo/generate_demo.py  # creates 20 synthetic .wav files
python train.py                    # runs on demo data by default
```
Note: demo scores will be artificially perfect (synthetic data is trivially separable).

---

## Quickstart

```bash
pip install -r requirements.txt

# Demo (synthetic audio — just verifies the pipeline runs)
python data/demo/generate_demo.py
python train.py

# Real data (Pitt Cookie Theft copy, speaker-grouped CV)
python data/make_pitt_labels.py --root /path/to/pitt_cookie
python train.py --audio_dir /path/to/pitt_cookie --labels data/adress/labels.csv --n_jobs 8
```

Results are saved to `results/scores.csv` (`--cv naive` writes `scores_naive.csv`).

---

## Results (Pitt Cookie Theft, speaker-grouped 5-fold CV)

**Data actually used:** the full Pitt Cookie Theft set from DementiaBank, *not* the ADReSS 2020 subset. 498 recordings from 267 speakers (AD 255 recordings / 168 speakers, HC 243 / 99). AD = ProbableAD + PossibleAD; MCI, Vascular, Memory and unlabelled dementia recordings are excluded. Labels are built by `data/make_pitt_labels.py`.

**CV:** `StratifiedGroupKFold`, speaker ID (filename prefix, e.g. `001` from `001-0`) as group, so all visits of one person are in the same fold. Mean over 5 folds, all 168 features.

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | 0.652 ± 0.047 | 0.658 | 0.708 ± 0.048 |
| SVM (RBF) | 0.672 ± 0.048 | 0.656 | 0.733 ± 0.050 |
| Random Forest | 0.674 ± 0.060 | 0.665 | 0.731 ± 0.045 |
| MLP | 0.637 ± 0.022 | 0.650 | 0.698 ± 0.045 |

Differences between models are within one standard deviation, so no model is clearly best.

Feature ablation (SVM): MFCC 0.679 acc / 0.731 AUC, pitch 0.644 / 0.679, energy + speech rate 0.516 / 0.559, all 0.672 / 0.733.

**Speaker leakage check.** Splitting by recording instead (`--cv naive`, old behaviour) gives slightly higher numbers (SVM 0.687 acc / 0.751 AUC; RF 0.683 / 0.754), because visits of the same speaker end up in both train and test. Grouped CV is the number to report.

**Caveats**
- Features are computed on the whole recording, so the investigator's voice is included, not only the participant's (the `.cha` timestamps could be used to cut it out).
- Not comparable one-to-one with Fraser et al. (2016, 81.9%, which also used linguistic features) or ADReSS 2020 systems (~83%, official train/test split with age and gender matching). Here, age and gender are not controlled.
- Single run, `random_state=42`, no hyper-parameter tuning.

---

## Project Structure

```
├── src/
│   ├── features.py     # librosa feature extraction
│   ├── dataset.py      # load audio + labels (+ speaker ids) → numpy arrays
│   └── models.py       # four sklearn classifiers
├── train.py            # training + speaker-grouped cross-validation
├── notebooks/
│   └── explore.ipynb   # feature visualisation + results
├── data/
│   ├── demo/           # synthetic audio for pipeline testing
│   └── adress/         # real data goes here (not in repo)
└── assets/             # literature review materials
    ├── poster_cognitive_decline.tex
    └── paper_comparison_table.html
```

---

## Key References

- Becker et al. (1994) — DementiaBank / Pitt Corpus
- Fraser et al. (2016) — Linguistic features, 81.9% accuracy
- Luz et al. (2020) — ADReSS Challenge (INTERSPEECH)
- Rohanian et al. (2021) — Multimodal BiLSTM, 84%

---

## Author

Feixiang Xiao — M.Sc. Computational Linguistics, University of Stuttgart

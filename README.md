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
    Pitch F0 (mean, std, min, max)
    RMS energy (mean, std)
    Speech rate (voiced frame ratio)
    → 167-dim feature vector per utterance
    ↓
StandardScaler
    ↓
4 classifiers compared:
    Logistic Regression (baseline)
    SVM with RBF kernel
    Random Forest
    MLP
    ↓
5-fold cross-validation → Accuracy / F1 / AUC
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

# Real data (after obtaining ADReSS)
python train.py --audio_dir data/adress/audio --labels data/adress/labels.csv
```

Results are saved to `results/scores.csv`.

---

## Results (ADReSS 2020, 5-fold CV)

*To be filled in once DementiaBank access is obtained.*

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| SVM (RBF) | — | — | — |
| Random Forest | — | — | — |
| MLP | — | — | — |

For reference: Fraser et al. (2016) reported 81.9% accuracy; top ADReSS 2020 systems reached ~83%.

---

## Project Structure

```
├── src/
│   ├── features.py     # librosa feature extraction
│   ├── dataset.py      # load audio + labels → numpy arrays
│   └── models.py       # four sklearn classifiers
├── train.py            # training + cross-validation
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

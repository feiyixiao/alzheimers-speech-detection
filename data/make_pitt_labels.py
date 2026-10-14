"""Build labels.csv from a local Pitt Cookie Theft copy (DementiaBank).

Expected layout:  <root>/{Dementia,Control}/cookie/<speaker>-<visit>.{mp3,cha}
The diagnosis comes from the PAR @ID line of each .cha file.

  AD = ProbableAD + PossibleAD,  HC = Control.
  MCI / Vascular / Memory / Other / blank are excluded (use --include_other_dementia to count them as AD).
"""
import argparse, glob, os, re
import pandas as pd

AD_DIAG = {"ProbableAD", "PossibleAD"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="data/adress/labels.csv")
    p.add_argument("--include_other_dementia", action="store_true")
    a = p.parse_args()

    rows = []
    for folder in ("Dementia", "Control"):
        for cha in sorted(glob.glob(os.path.join(a.root, folder, "cookie", "*.cha"))):
            base = os.path.basename(cha)[:-4]
            if not os.path.exists(cha[:-4] + ".mp3"):
                continue
            txt = open(cha, encoding="utf8", errors="ignore").read()
            m = re.search(r"@ID:\s+eng\|Pitt\|PAR\|[^|]*\|[^|]*\|([^|]*)\|", txt)
            diag = m.group(1) if m else ""
            if folder == "Control":
                label = "HC"
            elif diag in AD_DIAG or a.include_other_dementia:
                label = "AD"
            else:
                continue
            rows.append({"uttid": f"{folder}/cookie/{base}.mp3", "label": label,
                         "speaker": base.split("-")[0], "diagnosis": diag})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby("label").agg(recordings=("uttid", "size"), speakers=("speaker", "nunique")))
    print("speakers with both labels:", sorted(df.groupby("speaker")["label"].nunique().loc[lambda s: s > 1].index))


if __name__ == "__main__":
    main()

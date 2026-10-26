from pathlib import Path
import pandas as pd

src = Path("data/processed/peer/localization")
dst = Path("data/processed/peer_dedup/localization")
dst.mkdir(parents=True, exist_ok=True)

def norm(frame):
    return frame["sequence"].astype(str).str.upper().str.strip()

train, valid, test = (pd.read_csv(src / f"{s}.csv") for s in ("train", "valid", "test"))

test_seqs = set(norm(test))
train_clean = train[~norm(train).isin(test_seqs)]
valid_clean = valid[~norm(valid).isin(test_seqs | set(norm(train_clean)))]

for name, before, after in (("train", train, train_clean), ("valid", valid, valid_clean), ("test", test, test)):
    after.to_csv(dst / f"{name}.csv", index=False)
    print(f"{name}: {len(before)} -> {len(after)} (removed {len(before) - len(after)})")
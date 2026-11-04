"""Build a combined ("master") dataset from all prepared PEER tasks.
Run from the repository root:
    python Code/scripts/build_master_dataset.py

Leakage rule: each unique sequence goes to the strictest split it appears in
across all tasks (test > valid > train). So no task's test sequence is ever in
master train/valid, and no task's valid sequence is ever in master train.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
SOURCES = {
    "solubility": PROCESSED / "peer" / "solubility",
    "localization": PROCESSED / "peer_dedup" / "localization",
    "fold": PROCESSED / "peer" / "fold",
}
OUTPUT_DIR = PROCESSED / "peer" / "master"
SPLITS = ("train", "valid", "test")
PRIORITY = {"train": 0, "valid": 1, "test": 2}

# 1. Collect every (sequence, task, split, label) row from all tasks.
frames = []
for task, folder in SOURCES.items():
    for split in SPLITS:
        df = pd.read_csv(folder / f"{split}.csv")
        frames.append(pd.DataFrame({
            "sequence": df["sequence"].astype(str).str.upper().str.strip(),
            "task": task,
            "task_split": split,
            "task_label": df["label"],
        }))
rows = pd.concat(frames, ignore_index=True)

# 2. Give each unique sequence the strictest split it appears in.
rows["priority"] = rows["task_split"].map(PRIORITY)
names = {value: key for key, value in PRIORITY.items()}
master_split = rows.groupby("sequence")["priority"].max().map(names)

# 3. One row per unique sequence, keeping its label in each task it came from.
labels = rows.drop_duplicates(["sequence", "task"]).pivot(
    index="sequence", columns="task", values="task_label"
)
source_tasks = rows.groupby("sequence")["task"].agg(lambda t: ";".join(sorted(set(t))))
master = pd.DataFrame({"split": master_split, "source_tasks": source_tasks}).join(labels)
master = master.rename_axis("sequence").reset_index()
assert master["sequence"].is_unique

# 4. Write train/valid/test in the repo's standard format, plus extra columns.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for split in SPLITS:
    part = master[master["split"] == split].reset_index(drop=True)
    out = pd.DataFrame({
        "idx": range(len(part)),
        "sequence": part["sequence"],
        "label": 0,  # placeholder: the data loader requires a label column
        "split": split,
        "dataset": "master",
        "source_tasks": part["source_tasks"],
    })
    for task in SOURCES:
        out[f"{task}_label"] = part[task].astype("Int64")
    out.to_csv(OUTPUT_DIR / f"{split}.csv", index=False)
    print(f"master {split}: {len(out)} sequences")

# 5. Report sequences that moved to a stricter split because of cross-task overlap.
rows["master_split"] = rows["sequence"].map(master_split)
moved = rows[rows["task_split"] != rows["master_split"]]
print("\nMoved because of overlap between tasks (task, original split -> master split):")
print(moved.groupby(["task", "task_split", "master_split"]).size().to_string() if len(moved) else "  none")
print(f"\nWrote {OUTPUT_DIR}")
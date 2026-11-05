"""Load a prepared PEER task with the repo's data loader and print a quick summary.

Run from the repository root (or anywhere):
    python Code/scripts/load_peer_dataset.py --task fold
    python Code/scripts/load_peer_dataset.py --task localization
    python Code/scripts/load_peer_dataset.py --task solubility
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Code.src.utils.dataloader import create_dataloader  # noqa: E402

TASKS = ("solubility", "localization", "fold", "master")
DEFAULT_DATA_DIR = "data/processed/peer"

parser = argparse.ArgumentParser(description="Load a prepared PEER task and print a summary.")
parser.add_argument("--task", choices=TASKS, default="solubility")
parser.add_argument("--data_dir", default=None, help="Override the default folder for the task.")
args = parser.parse_args()

data_dir = PROJECT_ROOT / (args.data_dir or DEFAULT_DATA_DIR)
print(f"Task: {args.task} | data: {data_dir}")

for split in ("train", "valid", "test"):
    loader = create_dataloader(
        task=args.task,
        split=split,
        data_dir=data_dir,
        encoding="raw",
        batch_size=4,
        use_cache=False,
    )
    batch = next(iter(loader))
    labels = batch["label"]
    labels = labels.tolist() if hasattr(labels, "tolist") else labels
    print(
        f"  {split}: {len(loader.dataset)} sequences | "
        f"fields: {list(batch.keys())} | first labels: {labels}"
    )
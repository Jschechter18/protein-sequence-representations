from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
config_dir = repo_root / "external" / "PEER_Benchmark" / "config" / "single_task" / "BERT"

if not config_dir.is_dir():
    raise SystemExit(f"PEER configs not found: {config_dir}")

tasks = sorted(path.stem.removesuffix("_BERT") for path in config_dir.glob("*.yaml"))

for task in tasks:
    print(task)
print(f"\n{len(tasks)} tasks")
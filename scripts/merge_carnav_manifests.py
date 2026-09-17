"""Concatenate manifest_<shard>.jsonl files (written by parallel
generate_carnav.py processes) into the single manifest.jsonl that
model/dataset.py's ClipDataset expects."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="dataset/carnav_pilot")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / args.out_dir
    shard_files = sorted(out_dir.glob("manifest_*.jsonl"))
    if not shard_files:
        print(f"no manifest_*.jsonl shards found in {out_dir}")
        return

    lines = []
    for f in shard_files:
        lines.extend(f.read_text().splitlines())

    (out_dir / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    print(f"merged {len(shard_files)} shards, {len(lines)} episodes -> {out_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()

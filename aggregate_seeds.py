import json
import glob
import argparse
from collections import defaultdict
import numpy as np
import csv

def load_seed_file(path, key):
    with open(path, "r") as f:
        data = json.load(f)
    return {
        int(epoch): v["mean_iou"]
        for epoch, v in data[key].items()
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob for seed json files")
    ap.add_argument("--dataset", required=True, help="e.g. bdd or cityscapes")
    ap.add_argument("--ema", action="store_true", help="Use EMA variant")
    ap.add_argument("--out", required=True, help="Output CSV")
    args = ap.parse_args()

    key = args.dataset + ("-ema" if args.ema else "")
    files = sorted(glob.glob(args.glob))
    assert len(files) > 0, "No files found"

    per_epoch = defaultdict(list)

    for fpath in files:
        values = load_seed_file(fpath, key)
        for epoch, miou in values.items():
            per_epoch[epoch].append(miou)

    rows = []
    for epoch in sorted(per_epoch):
        vals = np.array(per_epoch[epoch])
        rows.append([
            epoch,
            vals.mean(),
            vals.std(ddof=0),
            len(vals),
        ])

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mean_miou", "std_miou", "num_seeds"])
        writer.writerows(rows)

    print(f"Wrote {args.out} ({len(rows)} epochs, key='{key}')")

if __name__ == "__main__":
    main()
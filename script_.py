#!/usr/bin/env python3
import argparse
import json
from typing import Dict, Any, List, Tuple, Optional


def _epoch_items(ds: Dict[str, Any]) -> List[Tuple[int, float]]:
    """
    Returns list of (epoch_int, mean_iou) sorted by epoch.
    Expects ds like: {"1": {"mean_iou": ...}, "2": {...}, ...}
    """
    items: List[Tuple[int, float]] = []
    for k, v in ds.items():
        try:
            epoch = int(k)
        except ValueError:
            continue
        if isinstance(v, dict) and "mean_iou" in v:
            items.append((epoch, float(v["mean_iou"])))
    items.sort(key=lambda x: x[0])
    return items


def mean_from_epoch(items: List[Tuple[int, float]], start_epoch: int) -> Optional[float]:
    vals = [miou for ep, miou in items if ep >= start_epoch]
    if not vals:
        return None
    return sum(vals) / len(vals)


def min_max(items: List[Tuple[int, float]]) -> Tuple[Tuple[int, float], Tuple[int, float]]:
    if not items:
        raise ValueError("Empty epoch list")
    min_item = min(items, key=lambda x: x[1])  # (epoch, miou)
    max_item = max(items, key=lambda x: x[1])
    return min_item, max_item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to metrics JSON (top-level dict of datasets).")
    ap.add_argument("--start-epoch", type=int, default=5, help="Compute mean_iou average from this epoch (inclusive).")
    ap.add_argument(
        "--minmax-datasets",
        nargs="*",
        default=["bdd", "bdd-ema", "cityscapes", "cityscapes-ema"],
        help="Datasets for which to print min/max mean_iou.",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Per-dataset mean from epoch X
    print(f"== Mean mIoU from epoch >= {args.start_epoch} ==")
    for ds_name in sorted(data.keys()):
        items = _epoch_items(data[ds_name])
        avg = mean_from_epoch(items, args.start_epoch)
        if avg is None:
            print(f"{ds_name:16s}  (no epochs >= {args.start_epoch})")
        else:
            print(f"{ds_name:16s}  avg={avg:.3f}  (n={sum(1 for ep, _ in items if ep >= args.start_epoch)})")

        # Min/max for selected datasets (respect start-epoch)
    print("\n== Min/Max mean_iou from epoch >= {} ==".format(args.start_epoch))
    for ds_name in args.minmax_datasets:
        if ds_name not in data:
            print(f"{ds_name:16s}  (not found)")
            continue

        items = _epoch_items(data[ds_name])
        items = [(ep, v) for ep, v in items if ep >= args.start_epoch]

        if not items:
            print(f"{ds_name:16s}  (no epochs >= {args.start_epoch})")
            continue

        (ep_min, v_min), (ep_max, v_max) = min_max(items)
        print(
            f"{ds_name:16s}  "
            f"min={v_min:.3f} @ep{ep_min:02d}   "
            f"max={v_max:.3f} @ep{ep_max:02d}"
        )

if __name__ == "__main__":
    main()
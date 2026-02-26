#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, math
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd


def _epochs(ds: Dict[str, Any]) -> List[int]:
    out = []
    for k in ds.keys():
        try:
            out.append(int(k))
        except Exception:
            pass
    return sorted(out)


def _miou(log: dict, key: str, epoch: int) -> float | None:
    ds = log.get(key)
    if ds is None:
        return None
    e = ds.get(str(epoch))
    if e is None:
        return None
    v = e.get("mean_iou")
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


@dataclass(frozen=True)
class Selection:
    seed: int
    epoch: int
    synthia: float
    synthia_ema: float
    score: float


def select_checkpoint(
    log: dict,
    seed: int,
    min_epoch: int,
    mode: str = "max_min",  # "max_min" or "lexicographic"
    synthia_key: str = "synthia",
    synthia_ema_key: str = "synthia-ema",
) -> Selection:
    # common epochs for both keys
    e1 = set(_epochs(log.get(synthia_key, {})))
    e2 = set(_epochs(log.get(synthia_ema_key, {})))
    epochs = sorted([e for e in (e1 & e2) if e >= min_epoch])
    if not epochs:
        raise ValueError(f"Seed {seed}: no epochs >= {min_epoch} common to {synthia_key} and {synthia_ema_key}")

    best: Selection | None = None
    for e in epochs:
        s = _miou(log, synthia_key, e)
        se = _miou(log, synthia_ema_key, e)
        if s is None or se is None:
            continue

        if mode == "max_min":
            score = min(s, se)
            cand = Selection(seed=seed, epoch=e, synthia=s, synthia_ema=se, score=score)
            if best is None or cand.score > best.score or (cand.score == best.score and cand.epoch > best.epoch):
                best = cand

        elif mode == "lexicographic":
            # maximize synthia, tie-break by synthia-ema, then later epoch
            score = float("nan")
            cand = Selection(seed=seed, epoch=e, synthia=s, synthia_ema=se, score=score)
            if best is None:
                best = cand
            else:
                if (cand.synthia > best.synthia) or \
                   (cand.synthia == best.synthia and cand.synthia_ema > best.synthia_ema) or \
                   (cand.synthia == best.synthia and cand.synthia_ema == best.synthia_ema and cand.epoch > best.epoch):
                    best = cand
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if best is None:
        raise ValueError(f"Seed {seed}: selection failed (missing miou values?)")
    return best


def range_delta(log: dict, key: str, min_epoch: int) -> dict:
    ds = log.get(key, {})
    epochs = [e for e in _epochs(ds) if e >= min_epoch]
    vals = []
    for e in epochs:
        v = _miou(log, key, e)
        if v is not None and not math.isnan(v):
            vals.append(v)

    if len(vals) == 0:
        return {"min": np.nan, "max": np.nan, "delta": np.nan, "n_epochs": 0}

    arr = np.array(vals, dtype=np.float64)
    return {"min": float(arr.min()), "max": float(arr.max()), "delta": float(arr.max() - arr.min()), "n_epochs": int(arr.size)}


def agg_stats(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "var": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size >= 2 else 0.0,
        "var": float(x.var(ddof=1)) if x.size >= 2 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="path with ...seed_{seed}.json placeholder")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--min-epoch", type=int, default=5)
    ap.add_argument("--selection-mode", choices=["max_min", "lexicographic"], default="max_min")
    ap.add_argument(
        "--targets",
        default="cityscapes,cityscapes-ema,bdd,bdd-ema",
        help="Target dataset keys to analyze for stability/performance",
    )
    ap.add_argument("--outdir", default="dg_target_out")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    targets = [k.strip() for k in args.targets.split(",") if k.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    # load logs
    logs: Dict[int, dict] = {}
    missing = []
    for seed in seeds:
        path = args.pattern.format(seed=seed)
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path, "r") as f:
            logs[seed] = json.load(f)
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    # 1) selection by source (synthia & synthia-ema)
    selections: List[Selection] = []
    for seed in sorted(logs.keys()):
        selections.append(select_checkpoint(logs[seed], seed, args.min_epoch, args.selection_mode))

    sel_df = pd.DataFrame([s.__dict__ for s in selections]).sort_values("seed").reset_index(drop=True)
    sel_df.to_csv(os.path.join(args.outdir, "selected_checkpoints_per_seed.csv"), index=False)

    # 2) stability on target domains (range deltas)
    stab_rows = []
    for seed in sorted(logs.keys()):
        log = logs[seed]
        for key in targets:
            r = range_delta(log, key, args.min_epoch)
            stab_rows.append(
                {
                    "seed": seed,
                    "dataset": key,
                    "min_epoch": args.min_epoch,
                    **r,
                }
            )
    stab_df = pd.DataFrame(stab_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    stab_df.to_csv(os.path.join(args.outdir, "target_stability_range_per_seed.csv"), index=False)

    # aggregate stability
    agg_rows = []
    for key in targets:
        deltas = stab_df.loc[stab_df["dataset"] == key, "delta"].to_numpy(dtype=np.float64)
        st = agg_stats(deltas)
        agg_rows.append({"dataset": key, "metric": "delta_range", **st})
    stab_agg_df = pd.DataFrame(agg_rows).sort_values(["dataset"]).reset_index(drop=True)
    stab_agg_df.to_csv(os.path.join(args.outdir, "target_stability_range_aggregate.csv"), index=False)

    # 3) target performance at selected checkpoints
    perf_rows = []
    for s in selections:
        for key in targets:
            v = _miou(logs[s.seed], key, s.epoch)
            perf_rows.append(
                {
                    "seed": s.seed,
                    "selected_epoch": s.epoch,
                    "dataset": key,
                    "miou": v if v is not None else np.nan,
                }
            )
    perf_df = pd.DataFrame(perf_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    perf_df.to_csv(os.path.join(args.outdir, "target_performance_at_selected_checkpoints.csv"), index=False)

    # aggregate perf
    perf_agg_rows = []
    for key in targets:
        vals = perf_df.loc[perf_df["dataset"] == key, "miou"].to_numpy(dtype=np.float64)
        st = agg_stats(vals)
        perf_agg_rows.append({"dataset": key, "metric": "miou_at_selected_ckpt", **st})
    perf_agg_df = pd.DataFrame(perf_agg_rows).sort_values(["dataset"]).reset_index(drop=True)
    perf_agg_df.to_csv(os.path.join(args.outdir, "target_performance_aggregate.csv"), index=False)

    # concise print
    print("=== Selected checkpoints (by Synthia + Synthia-EMA) ===")
    print(sel_df.to_string(index=False))
    print("\n=== Target stability (range delta) aggregate across seeds ===")
    print(stab_agg_df.to_string(index=False))
    print("\n=== Target performance at selected checkpoints aggregate across seeds ===")
    print(perf_agg_df.to_string(index=False))

    print("\nWrote CSVs to:", args.outdir)


if __name__ == "__main__":
    main()
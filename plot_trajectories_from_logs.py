"""Target-domain trajectories from training logs (mean across seeds).

Reads the JSON training logs directly, e.g.
    {"cityscapes": {"1": {"mean_iou": ...}, ...}, "cityscapes-ema": {...},
     "bdd": {...}, "bdd-ema": {...}, ...}

Layout: rows = target domains, columns = online | EMA.
Lines: mean target-domain mIoU across seeds per configuration.
The single-encoder model (RGB) is drawn as a dashed black reference.
Epochs outside the primary evaluation window are lightly shaded.

Run:
    python plot_trajectories_from_logs.py --log-dir ./logs_diss --output-dir ./figures
    python plot_trajectories_from_logs.py ... --check   # also print temporal metrics

--check recomputes temporal mean / SD / range per seed and window and aggregates
them across seeds (population SD, as in the thesis), so the printed values can be
compared against the tables.
"""
from pathlib import Path
import argparse
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

# ------------------------------------------------------------------ configuration
TITLE_TAG = 'synthia_dual_static'          # used in output file name
SOURCE = 'SYNTHIA'
SEEDS = [0, 1, 2]

# (label, filename pattern with {seed}, colour, linestyle)
CONFIGS = [
    ('RGB',           'synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_{seed}_new.json',  '#1F1F1F', (0, (5, 3))),
    ('RGB + Edge',    'v2_edge_branched_ade_seed{seed}.json',  '#E69F00', '-'),
    ('RGB + FFT',     'v2_fft_branched_ade_seed{seed}.json',   '#56B4E9', '-'),
    ('RGB + DCT',     'v2_dct_branched_ade_seed{seed}.json',   '#009E73', '-'),
    ('RGB + HSV',     'synthia_static_hsv_fixed_seed{seed}.json', '#CC79A7', '-'),
    ('RGB + Wavelet', 'v2_wavelets_branched_ade_seed{seed}.json', '#D55E00', '-'),
]

# (display name, log key)
TARGETS = [('Cityscapes', 'cityscapes'), ('BDD100K', 'bdd')]
PARAMS = [('Online', ''), ('EMA', '-ema')]

PRIMARY_WINDOW_START = 6
WINDOWS = [(6, 30), (11, 30), (16, 30)]
# ------------------------------------------------------------------------------


def load_curve(path, key):
    with open(path) as f:
        log = json.load(f)
    if key not in log:
        raise KeyError(f'{path}: key "{key}" not found (available: {list(log)})')
    series = log[key]
    epochs = sorted(int(e) for e in series)
    return np.array(epochs), np.array([series[str(e)]['mean_iou'] for e in epochs])


def load_config(log_dir, pattern, key):
    """Load all seeds; keep only epochs present in every seed (with a warning)."""
    loaded = []
    for seed in SEEDS:
        path = log_dir / pattern.format(seed=seed)
        epochs, values = load_curve(path, key)
        loaded.append((path.name, dict(zip(epochs.tolist(), values.tolist()))))
    common = sorted(set.intersection(*(set(d) for _, d in loaded)))
    for name, d in loaded:
        extra, missing = sorted(set(d) - set(common)), sorted(set(range(1, max(d) + 1)) - set(d))
        if extra or missing:
            print(f'WARNING {name} [{key}]: '
                  f'{len(d)} epochs (min {min(d)}, max {max(d)}); '
                  f'missing {missing or "-"}; dropped (not in all seeds) {extra or "-"}',
                  file=sys.stderr)
    curves = np.array([[d[e] for e in common] for _, d in loaded])
    return np.array(common), curves              # shape: (n_seeds, n_epochs)


def temporal_metrics(epochs, curves, start, end):
    mask = (epochs >= start) & (epochs <= end)
    w = curves[:, mask]
    mu = w.mean(axis=1)
    sd = w.std(axis=1, ddof=0)
    rng = w.max(axis=1) - w.min(axis=1)
    agg = lambda x: (x.mean(), x.std(ddof=0))
    return agg(mu), agg(sd), agg(rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--log-dir', type=Path, default=Path('./logs_diss'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load everything first so errors appear before plotting
    data = {}
    for label, pattern, _, _ in CONFIGS:
        for _, tkey in TARGETS:
            for _, psuf in PARAMS:
                data[(label, tkey, psuf)] = load_config(args.log_dir, pattern, tkey + psuf)

    if args.check:
        for tname, tkey in TARGETS:
            for s, e in WINDOWS:
                print(f'\n=== {tname} · epochs {s}-{e} ===')
                print(f'{"Configuration":16s} {"Params":7s} {"Mean":>16s} {"Std":>16s} {"Delta":>16s}')
                for label, *_ in CONFIGS:
                    for pname, psuf in PARAMS:
                        ep, cv = data[(label, tkey, psuf)]
                        (m, ms), (sd, sds), (r, rs) = temporal_metrics(ep, cv, s, e)
                        print(f'{label:16s} {pname:7s} {m:7.3f} ± {ms:5.3f}  '
                              f'{sd:6.3f} ± {sds:5.3f}  {r:6.3f} ± {rs:5.3f}')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#87919B', 'axes.linewidth': .7,
        'text.color': '#222222', 'axes.labelcolor': '#222222',
    })
    fig, axs = plt.subplots(len(TARGETS), len(PARAMS), figsize=(9.6, 3.3 * len(TARGETS) + .9),
                            sharex=True, squeeze=False)
    fig.subplots_adjust(left=.085, right=.99, bottom=.1, top=.86, wspace=.12, hspace=.32)

    for r, (tname, tkey) in enumerate(TARGETS):
        # shared y-limits per row (online and EMA comparable)
        row_vals = np.concatenate([data[(l, tkey, p)][1].mean(axis=0)
                                   for l, *_ in CONFIGS for _, p in PARAMS])
        lo, hi = np.floor(row_vals.min() - .5), np.ceil(row_vals.max() + .5)
        for c, (pname, psuf) in enumerate(PARAMS):
            ax = axs[r, c]
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E2E6EA', linewidth=.7)
            ax.axvspan(.5, PRIMARY_WINDOW_START - .5, color='#8A939C', alpha=.08, linewidth=0)
            for label, _, color, ls in CONFIGS:
                epochs, curves = data[(label, tkey, psuf)]
                lw = 1.6 if label == 'RGB' else 1.4
                ax.plot(epochs, curves.mean(axis=0), color=color, linestyle=ls,
                        linewidth=lw, zorder=3 if label == 'RGB' else 2)
            ax.set_xlim(.5, epochs.max() + .5)
            ax.set_ylim(lo, hi)
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(1 if hi - lo <= 8 else 2))
            ax.tick_params(length=3, color='#87919B')
            ax.set_title(f'({chr(97 + r * len(PARAMS) + c)}) {tname} · {pname}', loc='left', pad=8)
            if c == 0:
                ax.set_ylabel('Target mIoU (%)')
            if r == len(TARGETS) - 1:
                ax.set_xlabel('Epoch')

    handles = [Line2D([], [], color=col, linestyle=ls, linewidth=1.6, label=lab)
               for lab, _, col, ls in CONFIGS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .995),
               ncol=len(CONFIGS), frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.text(.54, .005, f'{SOURCE} · Mean across {len(SEEDS)} seeds · '
             f'Shaded: epochs before the primary evaluation window',
             ha='center', fontsize=8.5, color='#4D555D')

    stem = args.output_dir / f'trajectories_{TITLE_TAG}'
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'\n{stem.with_suffix(".pdf")}')


if __name__ == '__main__':
    sys.exit(main())
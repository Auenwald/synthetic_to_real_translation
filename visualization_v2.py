"""
Results pipeline: single encoder vs. dual-encoder variants (one architecture, one source).

From the JSON training logs, this script produces in one run:
  figures/<TAG>_temporal_online.pdf       dot plot, online only (isolated architecture effect)
  figures/<TAG>_temporal_online_ema.pdf   dot plot, online + EMA (combined effect)
  figures/<TAG>_trajectories.pdf          mean trajectories (appendix)
  tables/<TAG>_windows_cityscapes.tex     all windows, online + EMA   (appendix)
  tables/<TAG>_windows_bdd.tex            all windows, online + EMA   (appendix)
  tables/<TAG>_source_selected.tex        source-selected checkpoints (appendix)
  tables/<TAG>_figures.tex                LaTeX figure environments for the three figures
and prints a console summary (same format as the original analysis script).

Conventions (identical to the original analysis script):
  - temporal SD and aggregation across seeds use population SD (ddof=0)
  - missing epochs inside a window are skipped (a warning is printed)
  - checkpoint selection: epoch with the highest source-validation mIoU per seed

Usage:
  1. Edit the EXPERIMENT and CONFIGS blocks below (paths per seed, source keys).
  2. python results_pipeline.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

# =============================================================================
# EXPERIMENT SETTINGS  (edit here)
# =============================================================================
EXPERIMENT = {
    "tag": "synthia_dual_adaptive",                 # prefix for files and LaTeX labels
    "source": "SYNTHIA",                            # display name of the source dataset
    "arch_short": "adaptive bidirectional",         # used in captions
    "fig_dir": "./figures",
    "table_dir": "./tables",
    "results_section_label": "sec:results_single_ema",  # for the seed remark in captions
    "ema_config": r"$\lambda=0.995$, $u=20$",
}

LOG_DIR = "./logs_diss"
SEEDS = [0, 1, 2]


def seed_paths(pattern):
    """Helper: build per-seed paths from a pattern containing {seed}."""
    return [os.path.join(LOG_DIR, pattern.format(seed=s)) for s in SEEDS]


# One entry per configuration. The first entry is the single-encoder reference.
# 'source_key' is the log key of the source validation (online); '-ema' is appended for EMA.
# Colours: keep them identical across all figures of the thesis.
CONFIGS = [
    {"label": "RGB", "single": True, "color": "#1F1F1F",
     "source_key": "synthia",
     "paths": seed_paths("synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_{seed}_new.json")},
    {"label": "RGB + Edge", "color": "#E69F00", "source_key": "synthiabranched",
     "paths": seed_paths("v3_synthia_branched_edge_ade_seed{seed}.json")},
    {"label": "RGB + FFT", "color": "#56B4E9", "source_key": "synthiabranched",
     "paths": seed_paths("v3_synthia_branched_fft_ade_seed{seed}.json")},
    {"label": "RGB + DCT", "color": "#009E73", "source_key": "synthiabranched",
     "paths": seed_paths("synthia_v3_dct_branched_ade_seed{seed}.json")},
    {"label": "RGB + HSV", "color": "#CC79A7", "source_key": "synthiabranched",
     "paths": seed_paths("synthia_bidirectional_hsv_fixed_seed{seed}.json")},
    {"label": "RGB + Wavelet", "color": "#D55E00", "source_key": "synthiabranched",
     "paths": seed_paths("synthia_v3_wavelets_branched_ade_seed{seed}.json")},
]

TARGETS = [("Cityscapes", "cityscapes", "cityscapes"), ("BDD100K", "bdd", "bdd")]  # (display, log key, file suffix)
PARAMS = [("Online", ""), ("EMA", "-ema")]
WINDOWS = [(6, 30), (11, 30), (16, 30)]
PRIMARY = WINDOWS[0]
EXPECTED_LAST_EPOCH = 30
STD_DDOF = 0
BOLD_BEST = True          # bold best value per target domain and window in the tables
# =============================================================================


NUM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 10: "ten"}


def n_seeds_word():
    return NUM_WORDS.get(len(SEEDS), str(len(SEEDS)))


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr)


def load_logs(cfg):
    logs = []
    for p in cfg["paths"]:
        with open(p) as f:
            logs.append(json.load(f))
    return logs


def series(log, key):
    if key not in log:
        raise KeyError(f'key "{key}" not in log (available: {list(log)})')
    return {int(e): float(v["mean_iou"]) for e, v in log[key].items()}


def pstd(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=STD_DDOF)) if len(x) > STD_DDOF else 0.0


def check_epochs(cfg, logs):
    for path, log in zip(cfg["paths"], logs):
        for _, tkey, _ in TARGETS:
            for _, suf in PARAMS:
                s = series(log, tkey + suf)
                missing = sorted(set(range(1, EXPECTED_LAST_EPOCH + 1)) - set(s))
                if missing:
                    warn(f"{Path(path).name} [{tkey + suf}]: missing epochs {missing}")


def temporal(logs, key, start, end):
    per_seed = []
    for log in logs:
        s = series(log, key)
        vals = np.array([s[e] for e in range(start, end + 1) if e in s])
        per_seed.append((vals.mean(), pstd(vals), vals.max() - vals.min()))
    a = np.array(per_seed)
    return {"mu": (a[:, 0].mean(), pstd(a[:, 0])),
            "sd": (a[:, 1].mean(), pstd(a[:, 1])),
            "rng": (a[:, 2].mean(), pstd(a[:, 2]))}


def source_selected(logs, source_key, suf):
    epochs, results = [], {t[1]: [] for t in TARGETS}
    for log in logs:
        src = series(log, source_key + suf)
        best = max(src, key=src.get)
        epochs.append(best)
        for _, tkey, _ in TARGETS:
            results[tkey].append(series(log, tkey + suf)[best])
    return epochs, {k: (float(np.mean(v)), pstd(v)) for k, v in results.items()}


def curve_mean(logs, key):
    common = sorted(set.intersection(*(set(series(l, key)) for l in logs)))
    return np.array(common), np.array([np.mean([series(l, key)[e] for l in logs]) for e in common])


# ----------------------------------------------------------------------------- compute
def compute_all():
    res = {"temporal": {}, "selected": {}, "curves": {}}
    for cfg in CONFIGS:
        logs = load_logs(cfg)
        check_epochs(cfg, logs)
        lab = cfg["label"]
        for _, tkey, _ in TARGETS:
            for pname, suf in PARAMS:
                for w in WINDOWS:
                    res["temporal"][(lab, tkey, pname, w)] = temporal(logs, tkey + suf, *w)
                res["curves"][(lab, tkey, pname)] = curve_mean(logs, tkey + suf)
        for pname, suf in PARAMS:
            ep, vals = source_selected(logs, cfg["source_key"], suf)
            if max(ep) < EXPECTED_LAST_EPOCH or any(e < EXPECTED_LAST_EPOCH - 5 for e in ep):
                pass
            res["selected"][(lab, pname)] = (ep, vals)
    return res


def print_summary(res):
    for dname, tkey, _ in TARGETS:
        for w in WINDOWS:
            print(f"\n=== {dname} · epochs {w[0]}-{w[1]} ===")
            print(f'{"Configuration":16s} {"Params":7s} {"Mean":>17s} {"Std":>15s} {"Delta":>15s}')
            for cfg in CONFIGS:
                for pname, _ in PARAMS:
                    t = res["temporal"][(cfg["label"], tkey, pname, w)]
                    print(f'{cfg["label"]:16s} {pname:7s} '
                          f'{t["mu"][0]:7.3f} +- {t["mu"][1]:5.3f}  '
                          f'{t["sd"][0]:6.3f} +- {t["sd"][1]:5.3f}  '
                          f'{t["rng"][0]:6.3f} +- {t["rng"][1]:5.3f}')
    print("\n=== Source-selected checkpoints ===")
    for cfg in CONFIGS:
        for pname, _ in PARAMS:
            ep, vals = res["selected"][(cfg["label"], pname)]
            parts = "  ".join(f"{d}: {vals[k][0]:.3f} +- {vals[k][1]:.3f}" for d, k, _ in TARGETS)
            print(f'{cfg["label"]:16s} {pname:7s} epochs {ep}  {parts}')


# ----------------------------------------------------------------------------- tables
def fmt(m, s, bold=False):
    body = f"{m:.3f} \\pm {s:.3f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def best_flags(res, tkey, w):
    """Return dict (label, param, metric) -> bold?  (ties on rounded values are all bold)"""
    flags = {}
    for metric, better in (("mu", max), ("sd", min), ("rng", min)):
        vals = {(c["label"], p): round(res["temporal"][(c["label"], tkey, p, w)][metric][0], 3)
                for c in CONFIGS for p, _ in PARAMS}
        target = better(vals.values())
        for k, v in vals.items():
            flags[(*k, metric)] = BOLD_BEST and v == target
    return flags


def table_windows(res, dname, tkey, fsuf):
    tag, src, arch = EXPERIMENT["tag"], EXPERIMENT["source"], EXPERIMENT["arch_short"]
    n = n_seeds_word()
    rows_per_window = 2 * len(CONFIGS)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption[{src} results: {arch} dual encoder on {dname} across evaluation windows]{{%",
        rf"\textbf{{Temporal performance and checkpoint stability of the {arch}",
        rf"dual-encoder variants on {dname}.}}",
        rf"Training is performed on {src}. The temporal metrics are computed",
        rf"independently for each of {n} seeds and subsequently aggregated across",
        r"seeds. Reported values denote mean $\pm$ standard deviation across seeds.",
        r"\emph{RGB} denotes the single-encoder model; \emph{RGB + view} denotes the",
        r"dual-encoder model with the corresponding auxiliary view."
        + (r" Bold values indicate the best result for each evaluation window." if BOLD_BEST else ""),
        rf"EMA uses the reference configuration ({EXPERIMENT['ema_config']}).}}",
        rf"\label{{tab:{tag}_windows_{fsuf}}}",
        r"\small",
        r"\begin{tabular}{lllccc}",
        r"\toprule",
        r"$\mathcal{W}_{\mathrm{ep}}$ & Configuration & Parameters",
        r"& $\overline{\mu}_{\mathcal{W}_{\mathrm{ep}}}$",
        r"& $\overline{\sigma}_{\mathcal{W}_{\mathrm{ep}}}$",
        r"& $\overline{\Delta}_{\mathcal{W}_{\mathrm{ep}}}$ \\",
        r"\midrule",
    ]
    for wi, w in enumerate(WINDOWS):
        flags = best_flags(res, tkey, w)
        for ci, cfg in enumerate(CONFIGS):
            for pi, (pname, _) in enumerate(PARAMS):
                t = res["temporal"][(cfg["label"], tkey, pname, w)]
                wcol = rf"\multirow{{{rows_per_window}}}{{*}}{{{w[0]}--{w[1]}}}" if ci == 0 and pi == 0 else ""
                ccol = rf"\multirow{{2}}{{*}}{{{cfg['label']}}}" if pi == 0 else ""
                cells = " & ".join(fmt(*t[m], flags[(cfg["label"], pname, m)]) for m in ("mu", "sd", "rng"))
                lines.append(f"{wcol} & {ccol} & {pname} & {cells} \\\\")
            if ci < len(CONFIGS) - 1:
                lines.append(r"\addlinespace[2pt]")
        lines.append(r"\midrule" if wi < len(WINDOWS) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_source_selected(res):
    tag, src, arch = EXPERIMENT["tag"], EXPERIMENT["source"], EXPERIMENT["arch_short"]
    n = n_seeds_word()
    tcols = " & ".join(d for d, _, _ in TARGETS)
    best = {}
    for d, tkey, _ in TARGETS:
        vals = {(c["label"], p): round(res["selected"][(c["label"], p)][1][tkey][0], 3)
                for c in CONFIGS for p, _ in PARAMS}
        top = max(vals.values())
        for k, v in vals.items():
            best[(*k, tkey)] = BOLD_BEST and v == top
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption[{src} results: {arch} dual encoder at source-selected checkpoints]{{%",
        rf"\textbf{{Target-domain performance of the {arch} dual-encoder",
        r"variants at source-selected checkpoints.}",
        rf"Training is performed on {src}. Online and EMA checkpoints are selected",
        rf"independently for each of {n} seeds based exclusively on {src}",
        r"validation performance. Selected epochs are listed per seed. Reported values",
        r"denote mean target-domain mIoU $\pm$ standard deviation across seeds."
        + (r" Bold values indicate the best result for each target domain." if BOLD_BEST else ""),
        rf"EMA uses the reference configuration ({EXPERIMENT['ema_config']}).}}",
        rf"\label{{tab:{tag}_source_selected}}",
        r"\small",
        r"\begin{tabular}{llc" + "c" * len(TARGETS) + "}",
        r"\toprule",
        rf"Configuration & Parameters & Selected epochs & {tcols} \\",
        r"\midrule",
    ]
    for ci, cfg in enumerate(CONFIGS):
        for pi, (pname, _) in enumerate(PARAMS):
            ep, vals = res["selected"][(cfg["label"], pname)]
            ccol = rf"\multirow{{2}}{{*}}{{{cfg['label']}}}" if pi == 0 else ""
            eps = "$" + ", ".join(map(str, ep)) + "$"
            cells = " & ".join(fmt(*vals[tkey], best[(cfg["label"], pname, tkey)]) for _, tkey, _ in TARGETS)
            lines.append(f"{ccol} & {pname} & {eps} & {cells} \\\\")
        if ci < len(CONFIGS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures
def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 10.5, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#87919B", "axes.linewidth": .7,
        "text.color": "#222222", "axes.labelcolor": "#222222",
    })


def dot_plot(res, show_ema, path):
    C_ON, C_EMA, OFF = "#646C76", "#2878A5", (.17 if show_ema else 0)
    titles = [r"Temporal mean $\overline{\mu}_{\mathcal{W}_{\mathrm{ep}}}$",
              r"Temporal standard deviation $\overline{\sigma}_{\mathcal{W}_{\mathrm{ep}}}$",
              r"Temporal range $\overline{\Delta}_{\mathcal{W}_{\mathrm{ep}}}$"]
    ylabels = ["mIoU (%)", "mIoU points", "mIoU points"]
    metrics = ["mu", "sd", "rng"]
    eb = dict(markersize=6, capsize=3, elinewidth=1.25, capthick=1.1,
              markeredgecolor="white", markeredgewidth=.6, zorder=3)
    labels = [("RGB\n(single)" if c.get("single") else "RGB\n+ " + c["label"].split("+ ")[1]) for c in CONFIGS]
    x = np.arange(len(CONFIGS))
    params = PARAMS if show_ema else PARAMS[:1]

    fig, axs = plt.subplots(3, len(TARGETS), figsize=(9.6, 8.4), squeeze=False)
    fig.subplots_adjust(left=.09, right=.99, bottom=.09, top=.875, wspace=.2, hspace=.62)
    for col, (dname, tkey, _) in enumerate(TARGETS):
        for row, m in enumerate(metrics):
            ax = axs[row, col]
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E2E6EA", linewidth=.7)
            ax.axvspan(-.45, .45, color="#8A939C", alpha=.08, linewidth=0)
            ax.axvline(.5, color="#C9CFD5", linewidth=.8, linestyle=(0, (3, 3)), zorder=1)
            ref = res["temporal"][(CONFIGS[0]["label"], tkey, "Online", PRIMARY)][m][0]
            ax.axhline(ref, color=C_ON, linewidth=.8, linestyle=(0, (1, 2)), alpha=.8, zorder=1)
            vals_for_lim = []
            for i, cfg in enumerate(CONFIGS):
                for pname, _ in params:
                    mm, ss = res["temporal"][(cfg["label"], tkey, pname, PRIMARY)][m]
                    vals_for_lim += [mm - ss, mm + ss]
                    xo = x[i] - OFF if pname == "Online" else x[i] + OFF
                    ax.errorbar(xo, mm, yerr=ss, fmt="s" if pname == "Online" else "o",
                                color=C_ON if pname == "Online" else C_EMA, **eb)
            if m == "mu":
                lo, hi = np.floor(min(vals_for_lim) - .5), np.ceil(max(vals_for_lim) + .5)
                ax.set_ylim(lo, hi)
                ax.yaxis.set_major_locator(MultipleLocator(1))
            elif m == "sd":
                ax.set_ylim(0, max(1.9, np.ceil(max(vals_for_lim) * 2) / 2 + .1))
                ax.yaxis.set_major_locator(MultipleLocator(.5))
            else:
                ax.set_ylim(0, max(8.4, np.ceil(max(vals_for_lim)) + .4))
                ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.set_xticks(x, labels)
            ax.get_xticklabels()[0].set_fontweight("bold")
            ax.set_xlim(-.6, len(CONFIGS) - .4)
            if col == 0:
                ax.set_ylabel(ylabels[row])
            ax.set_title(f"({chr(97 + row * len(TARGETS) + col)}) {titles[row]}", loc="left", pad=8)
            ax.tick_params(length=3, color="#87919B")
        pos = axs[0, col].get_position()
        fig.text((pos.x0 + pos.x1) / 2, .915, dname, ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    handles = [Line2D([], [], marker="s", linestyle="", color=C_ON, label="Online")]
    if show_ema:
        handles.append(Line2D([], [], marker="o", linestyle="", color=C_EMA, label="EMA"))
    handles.append(Line2D([], [], color=C_ON, linewidth=.8, linestyle=(0, (1, 2)),
                          label="Single encoder, online"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .995),
               ncol=len(handles), frameon=False, handletextpad=.5, columnspacing=2.2)
    fig.text(.54, .005, f"{EXPERIMENT['source']} · Epochs {PRIMARY[0]}–{PRIMARY[1]} · "
             f"Mean ± standard deviation across {len(SEEDS)} seeds",
             ha="center", fontsize=9, color="#4D555D")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def trajectory_plot(res, path):
    fig, axs = plt.subplots(len(TARGETS), len(PARAMS), figsize=(9.6, 3.3 * len(TARGETS) + .9),
                            sharex=True, squeeze=False)
    fig.subplots_adjust(left=.085, right=.99, bottom=.1, top=.86, wspace=.12, hspace=.32)
    for r, (dname, tkey, _) in enumerate(TARGETS):
        allv = np.concatenate([res["curves"][(c["label"], tkey, p)][1] for c in CONFIGS for p, _ in PARAMS])
        lo, hi = np.floor(allv.min() - .5), np.ceil(allv.max() + .5)
        for c, (pname, _) in enumerate(PARAMS):
            ax = axs[r, c]
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E2E6EA", linewidth=.7)
            ax.axvspan(.5, PRIMARY[0] - .5, color="#8A939C", alpha=.08, linewidth=0)
            for cfg in CONFIGS:
                ep, mv = res["curves"][(cfg["label"], tkey, pname)]
                single = cfg.get("single", False)
                ax.plot(ep, mv, color=cfg["color"], linestyle=(0, (5, 3)) if single else "-",
                        linewidth=1.6 if single else 1.4, zorder=3 if single else 2)
            ax.set_xlim(.5, EXPECTED_LAST_EPOCH + .5)
            ax.set_ylim(lo, hi)
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(1 if hi - lo <= 8 else 2))
            ax.tick_params(length=3, color="#87919B")
            ax.set_title(f"({chr(97 + r * len(PARAMS) + c)}) {dname} · {pname}", loc="left", pad=8)
            if c == 0:
                ax.set_ylabel("Target mIoU (%)")
            if r == len(TARGETS) - 1:
                ax.set_xlabel("Epoch")
    handles = [Line2D([], [], color=c["color"], linewidth=1.6,
                      linestyle=(0, (5, 3)) if c.get("single") else "-", label=c["label"]) for c in CONFIGS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .995),
               ncol=len(CONFIGS), frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.text(.54, .005, f"{EXPERIMENT['source']} · Mean across {len(SEEDS)} seeds · "
             "Shaded: epochs before the primary evaluation window",
             ha="center", fontsize=8.5, color="#4D555D")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_snippets():
    tag, src, arch = EXPERIMENT["tag"], EXPERIMENT["source"], EXPERIMENT["arch_short"]
    n, sec = n_seeds_word(), EXPERIMENT["results_section_label"]
    refs = (rf"Tables~\ref{{tab:{tag}_windows_cityscapes}}" "\n"
            rf"    and~\ref{{tab:{tag}_windows_bdd}} and" "\n"
            rf"    Figure~\ref{{fig:{tag}_trajectories}}")
    return rf"""% ---------------------------------------------------------------- main text: architecture only
\begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\textwidth]{{figures/{tag}_temporal_online.pdf}}
    \caption[{src} results: {arch} dual encoder without EMA]{{%
    \textbf{{Temporal target-domain performance of the {arch}
    dual-encoder model compared with the single-encoder model for {src}
    training.}}
    Only online parameters are shown; metrics are computed over epochs
    ${PRIMARY[0]}$--${PRIMARY[1]}$. \emph{{RGB + view}} denotes the dual-encoder model with the
    corresponding auxiliary view. The dotted line indicates the
    single-encoder performance. Markers indicate the mean across {n}
    seeds; vertical bars indicate the corresponding standard deviation.
    Single-encoder results use the same {n} seeds and therefore differ
    slightly from Section~\ref{{{sec}}}. Complete results and
    trajectories are provided in
    {refs}.}}
    \label{{fig:{tag}_online}}
\end{{figure}}

% ---------------------------------------------------------------- main text: combined with EMA
\begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\textwidth]{{figures/{tag}_temporal_online_ema.pdf}}
    \caption[{src} results: {arch} dual encoder with and without EMA]{{%
    \textbf{{Combined effect of dual-encoder processing and EMA for the
    {arch} variant and {src} training.}}
    Online (squares) and EMA parameters (circles) are shown for the
    single-encoder model and each dual-encoder variant; metrics are computed
    over epochs ${PRIMARY[0]}$--${PRIMARY[1]}$. \emph{{RGB + view}} denotes the dual-encoder model
    with the corresponding auxiliary view. The dotted line indicates the
    online performance of the single-encoder model. Markers indicate the mean
    across {n} seeds; vertical bars indicate the corresponding standard
    deviation. EMA uses the reference configuration ({EXPERIMENT['ema_config']}).
    Complete results and trajectories are provided in
    {refs}.}}
    \label{{fig:{tag}_online_ema}}
\end{{figure}}

% ---------------------------------------------------------------- appendix: trajectories
\begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\textwidth]{{figures/{tag}_trajectories.pdf}}
    \caption[{src} results: trajectories of the {arch} dual-encoder variants]{{%
    \textbf{{Target-domain performance trajectories of the {arch}
    dual-encoder variants for the {src}-trained models.}}
    Lines show the mean target-domain mIoU across {n} seeds for online
    (left) and EMA parameters (right) on Cityscapes (top) and BDD100K
    (bottom). The single-encoder model (\emph{{RGB}}) is shown as a dashed
    reference; \emph{{RGB + view}} denotes the dual-encoder model with the
    corresponding auxiliary view. The shaded region marks the epochs
    preceding the primary evaluation window. EMA uses the reference
    configuration ({EXPERIMENT['ema_config']}). The corresponding temporal
    metrics are reported in
    Tables~\ref{{tab:{tag}_windows_cityscapes}}
    and~\ref{{tab:{tag}_windows_bdd}}.}}
    \label{{fig:{tag}_trajectories}}
\end{{figure}}
"""


# ----------------------------------------------------------------------------- main
def main():
    fig_dir, tab_dir = Path(EXPERIMENT["fig_dir"]), Path(EXPERIMENT["table_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    tag = EXPERIMENT["tag"]

    res = compute_all()
    print_summary(res)

    style()
    dot_plot(res, False, fig_dir / f"{tag}_temporal_online")
    dot_plot(res, True, fig_dir / f"{tag}_temporal_online_ema")
    trajectory_plot(res, fig_dir / f"{tag}_trajectories")

    for dname, tkey, fsuf in TARGETS:
        (tab_dir / f"{tag}_windows_{fsuf}.tex").write_text(table_windows(res, dname, tkey, fsuf))
    (tab_dir / f"{tag}_source_selected.tex").write_text(table_source_selected(res))
    (tab_dir / f"{tag}_figures.tex").write_text(figure_snippets())

    print(f"\nFigures: {fig_dir.resolve()}\nTables:  {tab_dir.resolve()}")


if __name__ == "__main__":
    main()
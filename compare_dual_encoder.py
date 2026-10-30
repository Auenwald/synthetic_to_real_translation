#!/usr/bin/env python3
"""Compare static/unidirectional and adaptive/bidirectional SYNTHIA logs.

Dependencies: numpy, matplotlib. Run from the directory containing logs_diss:
    python compare_dual_encoder_logs.py
See README_dual_encoder_comparison.md for metrics and interpretation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PATTERNS = {
    'HSV': {
        'Static': 'synthia_static_hsv_fixed_seed{seed}.json',
        'Bidirectional': 'synthia_bidirectional_hsv_fixed_seed{seed}.json',
    },
    'FFT': {
        'Static': 'v2_fft_branched_ade_seed{seed}.json',
        'Bidirectional': 'v3_synthia_branched_fft_ade_seed{seed}.json',
    },
}
COLORS = {'Static': '#245B84', 'Bidirectional': '#C56B24'}
LABELS = {'Static': 'Static unidirectional', 'Bidirectional': 'Adaptive bidirectional'}
DOMAINS = ('Source', 'Cityscapes', 'BDD100K')
MEASURES = (
    'source_selected_epoch', 'source_selected_score', 'target_at_source_selected',
    'target_peak_epoch', 'target_peak', 'target_final', 'peak_minus_selected',
    'peak_minus_final', 'selected_minus_peak_epochs', 'source_at_target_peak',
    'source_change_peak_to_selected', 'source_change_peak_to_final',
    'temporal_mean', 'temporal_std', 'temporal_range',
)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key!r}')
        result[key] = value
    return result


def resolve_key(data, domain, state, source_key=None):
    """Infer only known domain prefixes; ambiguous matches are errors."""
    if domain == 'Source' and source_key:
        key = source_key + ('-ema' if state == 'ema' else '')
        if key not in data:
            raise ValueError(f'Missing explicitly requested source key {key!r}')
        return key
    matches = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        normalized = key.lower().replace('_', '-').strip()
        is_ema = normalized.endswith('-ema')
        if is_ema != (state == 'ema'):
            continue
        base = normalized[:-4] if is_ema else normalized
        prefixes = {'Source': ('synthia',), 'Cityscapes': ('cityscapes',),
                    'BDD100K': ('bdd',)}[domain]
        if base.startswith(prefixes):
            matches.append(key)
    if len(matches) != 1:
        raise ValueError(f'{domain}/{state}: expected one matching key, found {matches}. '
                         f'Available keys: {list(data)}. For source use --source-key if needed.')
    return matches[0]


def extract_series(data, key, epochs, metric):
    block = data[key]
    values = []
    for epoch in epochs:
        record = block.get(str(int(epoch)))
        if not isinstance(record, dict) or metric not in record:
            raise ValueError(f'{key}: missing {metric!r} at epoch {epoch}')
        value = record[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'{key}/{epoch}/{metric}: expected finite numeric value, got {value!r}')
        values.append(float(value))
    return np.array(values)


def load_run(path, states, epochs, metric, source_key=None):
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=no_duplicate_keys)
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object at the top level')
        series, keys = {}, {}
        for state in states:
            series[state], keys[state] = {}, {}
            for domain in DOMAINS:
                key = resolve_key(data, domain, state, source_key)
                keys[state][domain] = key
                series[state][domain] = extract_series(data, key, epochs, metric)
        return series, keys
    except (ValueError, OSError) as exc:
        raise ValueError(f'{path}: {exc}') from exc


def argmax_index(values, tie_break):
    matches = np.flatnonzero(values == np.max(values))
    return int(matches[0] if tie_break == 'first' else matches[-1])


def run_metrics(source, target, epochs, windows, tie_break):
    # Selection uses every candidate epoch, independently of temporal windows.
    selected = argmax_index(source, tie_break)
    peak = argmax_index(target, tie_break)
    base = dict(
        source_selected_epoch=int(epochs[selected]), source_selected_score=float(source[selected]),
        target_at_source_selected=float(target[selected]), target_peak_epoch=int(epochs[peak]),
        target_peak=float(target[peak]), target_final=float(target[-1]),
        peak_minus_selected=float(target[peak] - target[selected]),
        peak_minus_final=float(target[peak] - target[-1]),
        selected_minus_peak_epochs=int(epochs[selected] - epochs[peak]),
        source_at_target_peak=float(source[peak]),
        source_change_peak_to_selected=float(source[selected] - source[peak]),
        source_change_peak_to_final=float(source[-1] - source[peak]),
        selected_after_peak_with_target_drop=bool(selected > peak and target[selected] < target[peak]),
        source_rises_to_selection_while_target_drops=bool(
            selected > peak and target[selected] < target[peak] and source[selected] > source[peak]),
    )
    rows = []
    for start, end in windows:
        values = target[(epochs >= start) & (epochs <= end)]
        rows.append(dict(base, window=f'{start}-{end}', n_epochs=len(values),
                         temporal_mean=float(np.mean(values)),
                         temporal_std=float(np.std(values, ddof=0)),
                         temporal_range=float(np.ptp(values))))
    return rows


def mean_sd(values, ddof):
    x = np.asarray(values, dtype=float)
    return float(x.mean()), float(x.std(ddof=ddof))


def aggregate(rows, group_fields, ddof, measures=MEASURES):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_fields), []).append(row)
    result = []
    for key, group in groups.items():
        record = dict(zip(group_fields, key))
        record['n_seeds'] = len(group)
        for metric in measures:
            mean, sd = mean_sd([r[metric] for r in group], ddof)
            record[metric + '_mean'] = mean
            record[metric + '_seed_sd'] = sd
        if 'source_rises_to_selection_while_target_drops' in group[0]:
            record['n_source_rises_to_selection_while_target_drops'] = sum(
                r['source_rises_to_selection_while_target_drops'] for r in group)
        result.append(record)
    return result


def paired_differences(rows):
    """Descriptive within-seed architectural differences, not a significance test."""
    groups = {}
    fields = ('view', 'state', 'target', 'seed', 'window')
    for r in rows:
        groups.setdefault(tuple(r[k] for k in fields), {})[r['architecture']] = r
    result = []
    for key, pair in groups.items():
        record = dict(zip(fields, key))
        for metric in MEASURES:
            record[metric] = pair['Bidirectional'][metric] - pair['Static'][metric]
        result.append(record)
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(runs, views, seeds, states, epochs, out, ddof, tie_break, metric, per_seed_plots=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})
    artifacts = []
    y_label = 'mIoU (%)' if metric == 'mean_iou' else metric
    handles = [Line2D([0], [0], color=COLORS[a], lw=2, label=LABELS[a]) for a in COLORS]
    mark_handles = [Line2D([0], [0], color='#444444', ls='', marker=m, markersize=8, label=l)
                    for m, l in [('D', 'Source-selected checkpoint'),
                                 ('*', 'Retrospective target maximum')]]

    def save(fig, name):
        for ext in ('png', 'pdf'):
            path = out / f'{name}.{ext}'
            fig.savefig(path, dpi=200, facecolor='white')
            artifacts.append(str(path))
        plt.close(fig)

    def style(ax, row):
        ax.set_ylabel(f'{DOMAINS[row]}\n{y_label}')
        ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlim(epochs[0] - .3, epochs[-1] + .3)
        if row == 2:
            ax.set_xlabel('Epoch')

    for state in states:
        # Mean curves: no fictitious common selection or peak epoch.
        fig, axes = plt.subplots(3, len(views), figsize=(6 * len(views), 8.6),
                                 sharex=True, sharey='row', squeeze=False)
        for col, view in enumerate(views):
            axes[0, col].set_title(view, fontweight='bold')
            for row, domain in enumerate(DOMAINS):
                ax = axes[row, col]
                for architecture in COLORS:
                    values = np.stack([runs[(view, architecture, seed)][state][domain] for seed in seeds])
                    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=ddof)
                    ax.plot(epochs, mean, color=COLORS[architecture], lw=1.8)
                    ax.fill_between(epochs, mean-sd, mean+sd, color=COLORS[architecture], alpha=.14, lw=0)
                style(ax, row)
        fig.suptitle(f'SYNTHIA: architectural comparison | {state.upper()}', y=.99, fontsize=14)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .956), ncol=2, frameon=False)
        fig.text(.5, .015, f'Curves: mean across {len(seeds)} seeds. Bands: ±1 SD across seeds (ddof={ddof}); not temporal SD.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .91))
        save(fig, f'overview_{state}')

        if not per_seed_plots:
            continue

        # Each column is one actual run/seed; selection and target maxima remain per-run.
        for view in views:
            fig, axes = plt.subplots(3, len(seeds), figsize=(4.4*len(seeds), 8.6),
                                     sharex=True, sharey='row', squeeze=False)
            for col, seed in enumerate(seeds):
                axes[0, col].set_title(f'Seed {seed}', fontweight='bold')
                for architecture, color in COLORS.items():
                    series = runs[(view, architecture, seed)][state]
                    selected = argmax_index(series['Source'], tie_break)
                    for row, domain in enumerate(DOMAINS):
                        ax, values = axes[row, col], series[domain]
                        ax.plot(epochs, values, color=color, lw=1.6)
                        ax.axvline(epochs[selected], color=color, ls='--', lw=.9, alpha=.55)
                        ax.scatter(epochs[selected], values[selected], marker='D', s=42,
                                   color=color, edgecolors='white', linewidths=.5, zorder=5)
                        if row:
                            peak = argmax_index(values, tie_break)
                            ax.scatter(epochs[peak], values[peak], marker='*', s=105,
                                       color=color, edgecolors='white', linewidths=.5, zorder=6)
                for row in range(3):
                    style(axes[row, col], row)
            fig.suptitle(f'{view} | {state.upper()}: source validation and target performance', y=.99, fontsize=14)
            fig.legend(handles=handles + mark_handles, loc='upper center',
                       bbox_to_anchor=(.5, .957), ncol=2, frameon=False)
            fig.text(.5, .015, 'Dashed lines: source-selected epochs. Target maxima are retrospective references only.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .04, 1, .89))
            save(fig, f'{view.lower()}_per_seed_{state}')
    return artifacts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--logs-dir', type=Path, default=Path('logs_diss'))
    p.add_argument('--output-dir', type=Path, default=Path('comparison_dual_encoders'))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--views', nargs='+', choices=list(PATTERNS), default=list(PATTERNS))
    p.add_argument('--states', nargs='+', choices=['online', 'ema'], default=['online', 'ema'])
    p.add_argument('--windows', nargs='+', default=['6:30', '11:30', '16:30'])
    p.add_argument('--last-epoch', type=int, default=30,
                   help='Candidate epochs are 1..last-epoch; independent of temporal windows.')
    p.add_argument('--seed-ddof', type=int, choices=[0, 1], default=1,
                   help='Across-seed SD convention; temporal SD always uses ddof=0.')
    p.add_argument('--tie-break', choices=['first', 'last'], default='first')
    p.add_argument('--metric', default='mean_iou')
    p.add_argument('--source-key', help='Exact online source-validation key; EMA uses this key + "-ema".')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--per-seed-plots', action='store_true',
                   help='Additionally export individual seed trajectories for the appendix.')
    args = p.parse_args()
    try:
        args.windows = [tuple(map(int, w.split(':'))) for w in args.windows]
        if any(len(w) != 2 or not 1 <= w[0] < w[1] <= args.last_epoch for w in args.windows):
            raise ValueError
    except ValueError:
        p.error('Windows must be START:END with 1 <= START < END <= last-epoch.')
    if len(args.seeds) <= args.seed_ddof:
        p.error('Number of seeds must exceed --seed-ddof.')
    for name in ('seeds', 'views', 'states', 'windows'):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            p.error(f'Duplicate values in --{name} are not allowed.')
    return args


def main():
    args = parse_args()
    epochs = np.arange(1, args.last_epoch + 1)
    paths = {(v, a, seed): args.logs_dir / pattern.format(seed=seed)
             for v in args.views for a, pattern in PATTERNS[v].items() for seed in args.seeds}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError('Missing log files (no seeds are silently omitted):\n  ' + '\n  '.join(missing))
    runs, provenance, rows = {}, [], []
    # Validate all logs before writing results. Missing epochs cause an error, not interpolation.
    for (view, architecture, seed), path in paths.items():
        run, keys = load_run(path, args.states, epochs, args.metric, args.source_key)
        runs[(view, architecture, seed)] = run
        provenance.append(dict(view=view, architecture=architecture, seed=seed,
                               path=str(path.resolve()), resolved_keys=keys))
        for state in args.states:
            for target in DOMAINS[1:]:
                for values in run_metrics(run[state]['Source'], run[state][target], epochs,
                                          args.windows, args.tie_break):
                    rows.append(dict(view=view, architecture=architecture, state=state,
                                     target=target, seed=seed, **values))
    summary = aggregate(rows, ('view', 'architecture', 'state', 'target', 'window'), args.seed_ddof)
    differences = paired_differences(rows)
    diff_summary = aggregate(differences, ('view', 'state', 'target', 'window'), args.seed_ddof)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, records in [('per_seed_metrics.csv', rows), ('summary_across_seeds.csv', summary),
                           ('bidirectional_minus_static_per_seed.csv', differences),
                           ('bidirectional_minus_static_summary.csv', diff_summary)]:
        write_csv(out / name, records)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out / 'analysis_config.json').write_text(json.dumps(dict(arguments=config, inputs=provenance,
        selection='Source argmax over epochs 1..last_epoch, independently for each seed and parameter trajectory',
        target_maximum='Retrospective maximum over epochs 1..last_epoch; never used for selection',
        score_units='Unchanged from logs; no automatic scaling', temporal_sd_ddof=0), indent=2), encoding='utf-8')
    primary = f'{args.windows[0][0]}-{args.windows[0][1]}'
    report = ['# Dual-encoder comparison', '',
              f'Candidate epochs: 1–{args.last_epoch}; primary temporal window: {primary}.',
              f'Seeds: {args.seeds}. Values: mean ± seed SD (ddof={args.seed_ddof}).', '',
              'Target maxima and gaps below are retrospective, computed within each seed before averaging.', '',
              '| View | Architecture | State | Target | Temporal mean | Temporal SD | Range | Target at source selection | Target maximum | Maximum − selected |',
              '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['window'] != primary:
            continue
        def fmt(key):
            return f"{r[key+'_mean']:.3f} ± {r[key+'_seed_sd']:.3f}"
        metrics = ('temporal_mean', 'temporal_std', 'temporal_range',
                   'target_at_source_selected', 'target_peak', 'peak_minus_selected')
        report.append('| ' + ' | '.join([r['view'], r['architecture'], r['state'], r['target']]
                                       + [fmt(k) for k in metrics]) + ' |')
    report += ['', 'The complete CSV files also contain peak/selection epochs, final target scores,',
               'source-score changes between these epochs, all temporal windows, and paired',
               'bidirectional-minus-static differences. No significance tests are performed.', '',
               'A positive source change from target peak to source selection describes endpoint',
               'differences; it does not establish a monotonically increasing source trajectory.',
               'Mean curves and seed SD bands do not measure within-run temporal variability.']
    (out / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    if not args.no_plots:
        plot_results(runs, args.views, args.seeds, args.states, epochs, out,
                     args.seed_ddof, args.tie_break, args.metric, args.per_seed_plots)
    print('\n'.join(report))
    print(f'\nResults saved to: {out.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(f'ERROR: {exc}')

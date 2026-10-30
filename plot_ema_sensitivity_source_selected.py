"""Source-selected target-domain performance for all EMA configurations.
Plots supplied aggregates only (mean +- SD across five seeds).
Run: python plot_ema_sensitivity_source_selected.py [--output-dir PATH]

Layout: one row per source dataset, one column per target domain.
Each panel: No EMA | decay variation (u = 20) | interval variation (lambda = 0.995).
The reference configuration (lambda = 0.995, u = 20) is the same run and
therefore appears in both groups. No EMA = online performance of the reference run.
To add GTA5: add an entry to DATA and YLIM; the figure grows to 2 x 2 automatically.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

REF_DECAY, REF_INTERVAL = '0.995', '20'
TARGETS = ['Cityscapes', 'BDD100K']

# (mean, sd) per target: [Cityscapes, BDD100K]
DATA = {
    'SYNTHIA': {
        'none': [(44.085, 1.463), (38.439, 2.211)],
        'decay': {  # u = 20
            '0.9':   [(44.990, 0.910), (39.270, 1.510)],
            '0.99':  [(45.040, 0.480), (39.620, 1.070)],
            '0.995': [(45.238, 0.606), (39.436, 1.281)],
            '0.999': [(45.790, 0.720), (40.180, 1.000)],
        },
        'interval': {  # lambda = 0.995
            '1':   [(44.651, 0.783), (38.968, 1.508)],
            '20':  [(45.238, 0.606), (39.436, 1.281)],
            '100': [(45.606, 0.641), (40.276, 1.145)],
            '500': [(45.350, 0.754), (39.954, 1.115)],
        },
    },
    # 'GTA5': {...}  # same structure
}

# Identical span (7 mIoU points) in all panels -> comparable effect sizes.
YLIM = {
    'SYNTHIA': {'Cityscapes': (41.0, 48.0), 'BDD100K': (35.0, 42.0)},
    # 'GTA5': {'Cityscapes': (..., ...), 'BDD100K': (..., ...)},
}

C_NONE, C_DECAY, C_INT, C_REF = '#646C76', '#2878A5', '#3A9A7A', '#C05B13'
X_NONE = 0.0
X_DECAY = [1.3, 2.3, 3.3, 4.3]
X_INT = [5.6, 6.6, 7.6, 8.6]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#87919B', 'axes.linewidth': .7,
        'text.color': '#222222', 'axes.labelcolor': '#222222',
    })
    sources = list(DATA.keys())
    nrows = len(sources)
    fig, axs = plt.subplots(nrows, 2, figsize=(10.2, 1.2 + 3.1 * nrows), squeeze=False)
    fig.subplots_adjust(left=.085, right=.99, wspace=.17, hspace=.62,
                        top=1 - .95 / (1.2 + 3.1 * nrows),
                        bottom=.85 / (1.2 + 3.1 * nrows))
    eb = dict(markersize=6, capsize=3.2, elinewidth=1.3, capthick=1.1,
              markeredgecolor='white', markeredgewidth=.6, zorder=3)

    for row, src in enumerate(sources):
        d = DATA[src]
        for col, tgt in enumerate(TARGETS):
            ax = axs[row, col]
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E2E6EA', linewidth=.7)
            for xs in (.65, 4.95):
                ax.axvline(xs, color='#C9CFD5', linewidth=.8, linestyle=(0, (3, 3)), zorder=1)

            m, s = d['none'][col]
            # Online reference level across the panel
            ax.axhline(m, color=C_NONE, linewidth=.8, linestyle=(0, (1, 2)), alpha=.8, zorder=1)
            ax.errorbar(X_NONE, m, yerr=s, fmt='s', color=C_NONE, **eb)

            for xs, key, color, ref in ((X_DECAY, 'decay', C_DECAY, REF_DECAY),
                                        (X_INT, 'interval', C_INT, REF_INTERVAL)):
                for x, (lab, vals) in zip(xs, d[key].items()):
                    m, s = vals[col]
                    is_ref = lab == ref
                    if is_ref:
                        ax.axvspan(x - .27, x + .27, color=C_REF, alpha=.07, linewidth=0)
                    ax.errorbar(x, m, yerr=s, fmt='D' if is_ref else 'o',
                                color=C_REF if is_ref else color, **eb)

            ticks = [X_NONE] + X_DECAY + X_INT
            ticklabels = ['No EMA'] + list(d['decay'].keys()) + list(d['interval'].keys())
            ax.set_xticks(ticks, ticklabels)
            for i, lab in enumerate(ticklabels):
                if (1 <= i <= 4 and lab == REF_DECAY) or (i >= 5 and lab == REF_INTERVAL):
                    ax.get_xticklabels()[i].set_color('#A74D0D')
                    ax.get_xticklabels()[i].set_fontweight('bold')
            ax.set_xlim(-.5, 9.1)
            ax.set_ylim(*YLIM[src][tgt])
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if col == 0:
                ax.set_ylabel('mIoU (%)')
            ax.set_title(f'({chr(97 + row * 2 + col)}) {src} → {tgt}', loc='left', pad=8)
            ax.tick_params(length=3, color='#87919B')

            if row == nrows - 1:
                trans = ax.get_xaxis_transform()
                ax.text(sum(X_DECAY) / 4, -.2, r'Decay $\lambda$  ($u=20$)',
                        transform=trans, ha='center', va='top', fontsize=9.5, color=C_DECAY)
                ax.text(sum(X_INT) / 4, -.2, r'Update interval $u$  ($\lambda=0.995$)',
                        transform=trans, ha='center', va='top', fontsize=9.5, color=C_INT)

    handles = [
        Line2D([], [], marker='s', linestyle='', color=C_NONE, label='Online (no EMA)'),
        Line2D([], [], marker='o', linestyle='', color=C_DECAY, label=r'EMA, varying $\lambda$'),
        Line2D([], [], marker='o', linestyle='', color=C_INT, label=r'EMA, varying $u$'),
        Line2D([], [], marker='D', linestyle='', color=C_REF,
               label=r'Reference ($\lambda=0.995$, $u=20$)'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, 1.0),
               ncol=4, frameon=False, handletextpad=.4, columnspacing=1.8)

    stem = args.output_dir / 'ema_sensitivity_source_selected'
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(stem.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
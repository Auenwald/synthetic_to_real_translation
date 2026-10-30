"""Combined EMA sensitivity plot (decay lambda and averaging interval u) for SYNTHIA.
Plots supplied aggregates only; no raw seed data inferred.
Run: python plot_ema_sensitivity_synthia.py [--output-dir PATH]
Error bars: across-seed SD (five seeds), not confidence intervals.
Layout: one column per target domain, one row per temporal metric.
Each panel: No EMA | decay sweep (u = 20) | interval sweep (lambda = 0.995).
The reference configuration (lambda = 0.995, u = 20) is the same run and
therefore appears in both sweeps.
For GTA5: copy this file and replace SOURCE, DATA and MEAN_YLIM.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

SOURCE = 'SYNTHIA'
REF_DECAY, REF_INTERVAL = '0.995', '20'

# Each entry: (mean, sd) per metric: [temporal mean, temporal SD, temporal range].
DATA = {
    'Cityscapes': {
        'none': ([45.295, 1.436, 6.091], [.699, .330, 1.764]),
        'decay': {  # u = 20
            '0.9':   ([45.793, .960, 4.160], [.680, .179, 1.306]),
            '0.99':  ([46.039, .942, 3.517], [.729, .239, .636]),
            '0.995': ([46.082, .878, 3.123], [.676, .179, .474]),
            '0.999': ([46.526, .686, 2.424], [.750, .171, .532]),
        },
        'interval': {  # lambda = 0.995
            '1':   ([45.702, .885, 3.569], [.749, .177, .969]),
            '20':  ([46.082, .878, 3.123], [.676, .179, .474]),
            '100': ([46.323, .693, 2.584], [.557, .195, .616]),
            '500': ([46.384, .827, 3.035], [.846, .193, .807]),
        },
    },
    'BDD100K': {
        'none': ([39.524, 1.266, 5.123], [1.448, .260, 1.072]),
        'decay': {
            '0.9':   ([39.909, .975, 3.905], [1.376, .223, .857]),
            '0.99':  ([40.318, .771, 3.088], [1.139, .225, .912]),
            '0.995': ([40.131, .690, 2.774], [1.305, .211, .919]),
            '0.999': ([40.552, .664, 2.721], [1.056, .180, .850]),
        },
        'interval': {
            '1':   ([39.978, .862, 3.389], [1.180, .159, .808]),
            '20':  ([40.131, .690, 2.774], [1.305, .211, .919]),
            '100': ([40.387, .639, 2.676], [.945, .224, 1.050]),
            '500': ([40.473, .613, 2.626], [1.059, .181, .863]),
        },
    },
}

# Identical span (5 mIoU points) for both targets -> comparable effect sizes.
MEAN_YLIM = {'Cityscapes': (43.5, 48.5), 'BDD100K': (37.5, 42.5)}

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
    fig, axs = plt.subplots(3, 2, figsize=(10.2, 8.8))
    fig.subplots_adjust(left=.085, right=.99, bottom=.115, top=.87,
                        wspace=.17, hspace=.62)
    titles = [
        r'Temporal mean $\overline{\mu}_{\mathcal{W}_{\mathrm{ep}}}$',
        r'Temporal standard deviation $\overline{\sigma}_{\mathcal{W}_{\mathrm{ep}}}$',
        r'Temporal range $\overline{\Delta}_{\mathcal{W}_{\mathrm{ep}}}$',
    ]
    ylabels = ['mIoU (%)', 'mIoU points', 'mIoU points']
    eb = dict(markersize=6, capsize=3.2, elinewidth=1.3, capthick=1.1,
              markeredgecolor='white', markeredgewidth=.6, zorder=3)

    for col, (domain, d) in enumerate(DATA.items()):
        for row in range(3):
            ax = axs[row, col]
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E2E6EA', linewidth=.7)

            # Group separators
            for xs in (.65, 4.95):
                ax.axvline(xs, color='#C9CFD5', linewidth=.8, linestyle=(0, (3, 3)), zorder=1)

            # No EMA
            m, s = d['none']
            ax.errorbar(X_NONE, m[row], yerr=s[row], fmt='s', color=C_NONE, **eb)

            # Sweeps
            for xs, key, color, ref in ((X_DECAY, 'decay', C_DECAY, REF_DECAY),
                                        (X_INT, 'interval', C_INT, REF_INTERVAL)):
                for x, (lab, (m, s)) in zip(xs, d[key].items()):
                    is_ref = lab == ref
                    if is_ref:
                        ax.axvspan(x - .27, x + .27, color=C_REF, alpha=.07, linewidth=0)
                    ax.errorbar(x, m[row], yerr=s[row], fmt='D' if is_ref else 'o',
                                color=C_REF if is_ref else color, **eb)

            ticks = [X_NONE] + X_DECAY + X_INT
            ticklabels = (['No EMA'] + list(d['decay'].keys()) + list(d['interval'].keys()))
            ax.set_xticks(ticks, ticklabels)
            for i, lab in enumerate(ticklabels):
                if (1 <= i <= 4 and lab == REF_DECAY) or (i >= 5 and lab == REF_INTERVAL):
                    ax.get_xticklabels()[i].set_color('#A74D0D')
                    ax.get_xticklabels()[i].set_fontweight('bold')
            ax.set_xlim(-.5, 9.1)

            if row == 0:
                ax.set_ylim(*MEAN_YLIM[domain])
                ax.yaxis.set_major_locator(MultipleLocator(1))
            elif row == 1:
                ax.set_ylim(0, 1.9)
                ax.yaxis.set_major_locator(MultipleLocator(.5))
            else:
                ax.set_ylim(0, 8.4)
                ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            if col == 0:
                ax.set_ylabel(ylabels[row])
            ax.set_title(f'({chr(97 + row * 2 + col)}) {titles[row]}', loc='left', pad=8)
            ax.tick_params(length=3, color='#87919B')

            # Group labels below tick labels (bottom row only)
            if row == 2:
                trans = ax.get_xaxis_transform()
                ax.text(sum(X_DECAY) / 4, -.27, r'Decay $\lambda$  ($u=20$)',
                        transform=trans, ha='center', va='top', fontsize=9.5, color=C_DECAY)
                ax.text(sum(X_INT) / 4, -.27, r'Interval $u$  ($\lambda=0.995$)',
                        transform=trans, ha='center', va='top', fontsize=9.5, color=C_INT)

        pos = axs[0, col].get_position()
        fig.text((pos.x0 + pos.x1) / 2, .91, domain, ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    handles = [
        Line2D([], [], marker='s', linestyle='', color=C_NONE, label='Online (no EMA)'),
        Line2D([], [], marker='o', linestyle='', color=C_DECAY, label=r'EMA, varying $\lambda$'),
        Line2D([], [], marker='o', linestyle='', color=C_INT, label=r'EMA, varying $u$'),
        Line2D([], [], marker='D', linestyle='', color=C_REF,
               label=r'Reference ($\lambda=0.995$, $u=20$)'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .995),
               ncol=4, frameon=False, handletextpad=.4, columnspacing=1.8)
    fig.text(.54, .012, f'{SOURCE} · Epochs 6–30 · Mean ± standard deviation across five seeds',
             ha='center', fontsize=9, color='#4D555D')

    stem = args.output_dir / f'ema_sensitivity_temporal_{SOURCE.lower()}'
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(stem.with_suffix('.pdf'))
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
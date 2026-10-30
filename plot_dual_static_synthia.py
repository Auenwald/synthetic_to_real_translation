"""Static unidirectional dual-encoder vs. single encoder (SYNTHIA training).
Plots supplied aggregates only; three seeds (0, 1, 2) for all configurations.
Run: python plot_dual_static_synthia.py [--output-dir PATH] [--show-ema]
Default: online only (isolated architecture effect). --show-ema adds EMA markers.
Layout: one column per target domain, one row per temporal metric.
Each configuration shows online (square) and EMA (circle) side by side.
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
ARCH = 'Static unidirectional dual encoder'
CONFIGS = ['Single', 'Edge', 'FFT', 'DCT', 'HSV', 'Wavelet']

# (mean, sd) per metric [mu, sigma, delta]; 'online' / 'ema'
DATA = {
    'Cityscapes': {
        'Single':  {'online': ([45.103, 1.494, 6.206], [.799, .169, .651]),
                    'ema':    ([45.977, 0.906, 3.210], [.767, .088, .440])},
        'Edge':    {'online': ([46.324, 1.164, 5.272], [.251, .148, .995]),
                    'ema':    ([47.000, 0.895, 2.984], [.254, .093, .297])},
        'FFT':     {'online': ([46.726, 1.200, 5.393], [.407, .183, .515]),
                    'ema':    ([47.411, 0.916, 3.116], [.435, .194, .565])},
        'DCT':     {'online': ([46.598, 1.097, 4.988], [.234, .112, .509]),
                    'ema':    ([47.013, 0.686, 2.389], [.204, .017, .239])},
        'HSV':     {'online': ([46.601, 0.938, 3.744], [.191, .291, .879]),
                    'ema':    ([46.939, 0.626, 2.313], [.166, .230, .879])},
        'Wavelet': {'online': ([46.888, 0.892, 3.990], [.713, .056, .552]),
                    'ema':    ([47.370, 0.655, 2.310], [.825, .163, .550])},
    },
    'BDD100K': {
        'Single':  {'online': ([38.878, 1.175, 4.765], [1.313, .232, .874]),
                    'ema':    ([39.508, 0.737, 2.938], [1.055, .228, 1.031])},
        'Edge':    {'online': ([41.739, 1.206, 5.470], [.709, .111, .500]),
                    'ema':    ([42.304, 0.813, 2.800], [.743, .142, .453])},
        'FFT':     {'online': ([41.543, 1.238, 5.621], [.128, .073, .588]),
                    'ema':    ([42.083, 0.839, 3.087], [.202, .195, .536])},
        'DCT':     {'online': ([41.645, 1.107, 4.730], [.731, .121, .843]),
                    'ema':    ([41.871, 0.574, 1.983], [.747, .072, .331])},
        'HSV':     {'online': ([40.657, 1.224, 4.850], [.346, .276, 1.473]),
                    'ema':    ([40.810, 0.948, 3.380], [.520, .260, 1.197])},
        'Wavelet': {'online': ([41.265, 1.301, 5.236], [.177, .216, .833]),
                    'ema':    ([41.669, 1.030, 3.792], [.276, .047, .236])},
    },
}

MEAN_YLIM = {'Cityscapes': (43.5, 48.5), 'BDD100K': (37.5, 43.5)}
C_ONLINE, C_EMA = '#646C76', '#2878A5'
OFFSET = .17


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--show-ema', action='store_true',
                        help='Show EMA markers next to the online markers.')
    args = parser.parse_args()
    show_ema = args.show_ema
    off = OFFSET if show_ema else 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#87919B', 'axes.linewidth': .7,
        'text.color': '#222222', 'axes.labelcolor': '#222222',
    })
    fig, axs = plt.subplots(3, 2, figsize=(9.6, 8.4))
    fig.subplots_adjust(left=.09, right=.99, bottom=.09, top=.875,
                        wspace=.2, hspace=.62)
    titles = [
        r'Temporal mean $\overline{\mu}_{\mathcal{W}_{\mathrm{ep}}}$',
        r'Temporal standard deviation $\overline{\sigma}_{\mathcal{W}_{\mathrm{ep}}}$',
        r'Temporal range $\overline{\Delta}_{\mathcal{W}_{\mathrm{ep}}}$',
    ]
    ylabels = ['mIoU (%)', 'mIoU points', 'mIoU points']
    eb = dict(markersize=6, capsize=3, elinewidth=1.25, capthick=1.1,
              markeredgecolor='white', markeredgewidth=.6, zorder=3)
    x = np.arange(len(CONFIGS))

    for col, (tgt, d) in enumerate(DATA.items()):
        for row in range(3):
            ax = axs[row, col]
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E2E6EA', linewidth=.7)
            ax.axvspan(-.45, .45, color='#8A939C', alpha=.08, linewidth=0)
            ax.axvline(.5, color='#C9CFD5', linewidth=.8, linestyle=(0, (3, 3)), zorder=1)
            # Single-encoder online level as reference
            ax.axhline(d['Single']['online'][0][row], color=C_ONLINE, linewidth=.8,
                       linestyle=(0, (1, 2)), alpha=.8, zorder=1)
            for i, cfg in enumerate(CONFIGS):
                m, s = d[cfg]['online']
                ax.errorbar(x[i] - off, m[row], yerr=s[row], fmt='s', color=C_ONLINE, **eb)
                if show_ema:
                    m, s = d[cfg]['ema']
                    ax.errorbar(x[i] + off, m[row], yerr=s[row], fmt='o', color=C_EMA, **eb)
            ax.set_xticks(x, ['RGB\n(single)'] + [f'RGB\n+ {c}' for c in CONFIGS[1:]])
            ax.get_xticklabels()[0].set_fontweight('bold')
            ax.set_xlim(-.6, len(CONFIGS) - .4)
            if row == 0:
                ax.set_ylim(*MEAN_YLIM[tgt])
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
        pos = axs[0, col].get_position()
        fig.text((pos.x0 + pos.x1) / 2, .915, tgt, ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    handles = [Line2D([], [], marker='s', linestyle='', color=C_ONLINE, label='Online')]
    if show_ema:
        handles.append(Line2D([], [], marker='o', linestyle='', color=C_EMA, label='EMA'))
    handles.append(Line2D([], [], color=C_ONLINE, linewidth=.8, linestyle=(0, (1, 2)),
                          label='Single encoder, online'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .995),
               ncol=len(handles), frameon=False, handletextpad=.5, columnspacing=2.2)
    fig.text(.54, .005, f'{SOURCE} · Epochs 6–30 · Mean ± standard deviation across three seeds',
             ha='center', fontsize=9, color='#4D555D')

    suffix = '_online_ema' if show_ema else '_online'
    stem = args.output_dir / f'dual_static_temporal_{SOURCE.lower()}{suffix}'
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(stem.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
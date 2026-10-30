"""Relative EMA effects for the single-encoder model across source/target settings
and evaluation windows. Computed from supplied aggregates (means across five seeds).
Run: python plot_ema_relative_effects_single.py [--output-dir PATH]
 
Relative reduction = 1 - metric_EMA / metric_Online, computed from seed-averaged values.
Mean gain = mu_EMA - mu_Online in mIoU points.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
 
WINDOWS = ['6–30', '11–30', '16–30']
 
# (mu, sigma, delta) seed-averaged, per window: Online, EMA
DATA = {
    'SYNTHIA → Cityscapes': {
        'Online': [(45.295, 1.436, 6.091), (45.130, 1.340, 5.603), (44.836, 1.149, 4.626)],
        'EMA':    [(46.082, 0.878, 3.123), (45.948, 0.769, 2.683), (45.651, 0.521, 1.788)],
    },
    'SYNTHIA → BDD100K': {
        'Online': [(39.524, 1.266, 5.123), (39.533, 1.240, 4.840), (39.297, 1.139, 4.199)],
        'EMA':    [(40.131, 0.690, 2.774), (40.195, 0.586, 2.009), (40.027, 0.552, 1.827)],
    },
    'GTA5 → Cityscapes': {
        'Online': [(47.943, 1.079, 4.274), (48.141, 0.915, 3.571), (48.098, 0.824, 2.903)],
        'EMA':    [(48.446, 0.721, 2.889), (48.609, 0.577, 1.950), (48.632, 0.469, 1.536)],
    },
    'GTA5 → BDD100K': {
        'Online': [(49.019, 0.866, 3.609), (49.084, 0.808, 3.259), (49.066, 0.739, 2.998)],
        'EMA':    [(49.570, 0.469, 1.975), (49.618, 0.387, 1.483), (49.646, 0.334, 1.240)],
    },
}
 
WIN_COLORS = ['#9CC3DC', '#4F95C2', '#1B5A85']
WIN_MARKERS = ['o', 'o', 'o']
 
 
def compute():
    settings = list(DATA.keys())
    gain = np.zeros((len(settings), 3))
    red_sd = np.zeros_like(gain)
    red_rng = np.zeros_like(gain)
    for i, s in enumerate(settings):
        on, ema = np.array(DATA[s]['Online']), np.array(DATA[s]['EMA'])
        gain[i] = ema[:, 0] - on[:, 0]
        red_sd[i] = 100 * (1 - ema[:, 1] / on[:, 1])
        red_rng[i] = 100 * (1 - ema[:, 2] / on[:, 2])
    return settings, gain, red_sd, red_rng
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#87919B', 'axes.linewidth': .7,
        'text.color': '#222222', 'axes.labelcolor': '#222222',
    })
    settings, gain, red_sd, red_rng = compute()
    n = len(settings)
    y = np.arange(n)[::-1]
 
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=True)
    fig.subplots_adjust(left=.175, right=.99, bottom=.2, top=.74, wspace=.12)
 
    panels = [
        (gain, '(a) Gain in temporal mean\n' r'$\overline{\mu}_{\mathcal{W}_{\mathrm{ep}}}$',
         'mIoU points', (0, 1.2), .2, '%.1f'),
        (red_sd, '(b) Reduction of temporal\nstandard deviation ' r'$\overline{\sigma}_{\mathcal{W}_{\mathrm{ep}}}$',
         'Relative reduction (%)', (0, 70), 10, '%d'),
        (red_rng, '(c) Reduction of temporal\nrange ' r'$\overline{\Delta}_{\mathcal{W}_{\mathrm{ep}}}$',
         'Relative reduction (%)', (0, 70), 10, '%d'),
    ]
    for ax, (vals, title, xlabel, xlim, step, fmt) in zip(axs, panels):
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E2E6EA', linewidth=.7)
        for i in range(n):
            ax.plot([vals[i].min(), vals[i].max()], [y[i], y[i]],
                    color='#B8C0C8', linewidth=2.2, solid_capstyle='round', zorder=2)
            for w in range(3):
                ax.plot(vals[i, w], y[i], marker=WIN_MARKERS[w], linestyle='',
                        color=WIN_COLORS[w], markersize=7.5,
                        markeredgecolor='white', markeredgewidth=.7, zorder=3)
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc='center', pad=8, multialignment='center')
        ax.tick_params(length=3, color='#87919B')
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(-.6, n - .4)
 
    axs[0].set_yticks(y, settings)
    # Separator between source datasets
    for ax in axs:
        ax.axhline(1.5, color='#D5DAE0', linewidth=.8, linestyle=(0, (3, 3)), zorder=1)
 
    handles = [Line2D([], [], marker='o', linestyle='', color=c, markersize=7.5,
                      label=f'Epochs {w}') for c, w in zip(WIN_COLORS, WINDOWS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.58, 1.0),
               ncol=3, frameon=False, handletextpad=.3, columnspacing=2.0,
               title='Evaluation window', title_fontsize=9.5)
    fig.text(.58, .01, r'Single-encoder model · EMA ($\lambda=0.995$, $u=20$) vs. online · '
             'computed from seed-averaged values (five seeds)',
             ha='center', fontsize=8.5, color='#4D555D')
 
    stem = args.output_dir / 'ema_relative_effects_single'
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
 
    for i, s in enumerate(settings):
        print(f'{s:24s} gain {np.round(gain[i], 2)}  SD {np.round(red_sd[i])}  range {np.round(red_rng[i])}')
 
 
if __name__ == '__main__':
    main()
 
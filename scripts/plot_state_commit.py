"""Plot matched exact-answer validation trajectories for each seed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    arms = {'always': ('Always / stored', '#687983', '--'),
            'held': ('Held / stored', '#087B60', '-'),
            'always_fresh': ('Always / fresh', '#A47735', '--'),
            'held_fresh': ('Held / fresh', '#714CB3', '-')}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True, layout='constrained')
    for ax, seed in zip(axes, (41, 43)):
        for arm, (label, color, style) in arms.items():
            history = [json.loads(line) for line in (args.root/str(seed)/arm/'history.jsonl').read_text().splitlines()]
            ax.plot([row['step'] for row in history],
                    [row['validation']['groups']['exact']['nll'] for row in history],
                    label=label, color=color, linestyle=style, marker='o', markersize=3)
        ax.axvline(200, color='#89938F', linewidth=.7, alpha=.6)
        ax.axvline(400, color='#89938F', linewidth=.7, alpha=.6)
        ax.set_title(f'Initialization seed {seed}')
        ax.set_xlabel('Optimizer updates')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Exact-answer validation NLL (nats/byte)')
    axes[1].legend(frameon=False, fontsize=7.8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

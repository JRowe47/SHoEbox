"""Plot the locked paired validation curves without evaluating test data."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/joint_stability/study')
    p.add_argument('--out', type=Path, default=ROOT/'experiments/text_regions/results/joint_stability/learning_curves.png')
    a = p.parse_args()
    lock = json.loads((a.root/'protocol_lock.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), sharex=True)
    styles = {'sum': ('Ordinary sum', '#b86631'), 'group_clip': ('Task/group cap', '#087f72')}
    memory = []
    for col, run in enumerate(lock['runs']):
        for mode, (label, color) in styles.items():
            history = json.loads((a.root/str(run['seed'])/mode/'history.json').read_text())
            rows = [r for r in history if 'text_validation' in r]
            x = [r['step'] for r in rows]
            y = [100*min(f['accuracy'] for s in r['memory_validation'] for f in s['fields']) for r in rows]
            memory.extend(y)
            axes[0,col].plot(x, [r['text_validation']['nats'] for r in rows], 'o-', color=color, label=label, linewidth=1.8, markersize=4)
            axes[1,col].plot(x, y, 'o-', color=color, linewidth=1.8, markersize=4)
        axes[0,col].set_title(f"Pair {col+1} · seed {run['seed']}", fontsize=11)
        axes[1,col].axhline(90, color='#5b6770', linestyle='--', linewidth=1)
        axes[1,col].set_xlabel('Continuation update')
        axes[1,col].set_xticks([0,20,40,60])
        for row in range(2):
            axes[row,col].grid(alpha=.16)
            axes[row,col].spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('Text validation NLL (nats/byte)')
    axes[1,0].set_ylabel('Weakest memory field (%)')
    for ax in axes[1]: ax.set_ylim(min(65, min(memory)-3), 101)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('SHoEbox · joint text and memory continuation', fontsize=15, fontweight='bold', y=.98)
    fig.tight_layout(rect=(0,.06,1,.95), h_pad=1.6)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180, facecolor='white')
    plt.close(fig)
    print(a.out)


if __name__ == '__main__': main()

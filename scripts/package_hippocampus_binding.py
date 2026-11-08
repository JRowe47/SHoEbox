"""Package the committed binding experiment and exact recovery evidence."""
import argparse,json,subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.package_hippocampus import make_archive,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--public-commit',required=True);p.add_argument('--public-tree',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    if tree!=a.public_tree:raise ValueError('Published and local source trees differ.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True):
        raise ValueError('Commit tracked changes before packaging.')
    root=ROOT/'artifacts/hippocampus_binding';prior=ROOT/'artifacts/hippocampus'
    summary_path=ROOT/'experiments/text_regions/results/hippocampus_binding/summary.json'
    summary=json.loads(summary_path.read_text())
    if not summary['source_schedules_identical']:raise ValueError('Source schedules differ.')
    checkpoints={'parent':prior/'parent.pt','legacy':prior/'C/last.pt','event_expert':root/'event_expert/last.pt'}
    for name,path in checkpoints.items():
        if digest(path)!=summary['models'][name]['checkpoint_sha256']:raise ValueError('Checkpoint changed.')
    if not summary['ca1']['source_schedules_identical'] or not all(summary['ca1']['initial_state_identity'].values()):
        raise ValueError('CA1 continuation identities differ.')
    for name in ('legacy','bounded'):
        if digest(root/'ca1'/name/'last.pt')!=summary['ca1']['models'][name]['checkpoint_sha256']:
            raise ValueError('CA1 checkpoint changed.')
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    source=[ROOT/name for name in names if name]
    revision=f'Published commit: {a.public_commit}\nGit source tree: {tree}\n'
    result=[make_archive(a.out/'SHoEbox_Hippocampus_Binding_Source.zip',source,'SHoEbox/',{'SOURCE_REVISION.txt':revision})]
    files=[summary_path,ROOT/'data/tiny_shakespeare.txt',prior/'parent.pt']
    for folder in (prior/'C',root/'event_expert'):
        files.extend(folder/name for name in ('best.pt','last.pt','protocol.json','initial.json','history.jsonl'))
    files.append(root/'event_expert/update16.pt')
    files.extend(prior/'A'/name for name in ('protocol.json','initial.json'))
    files.extend(root.glob('*.json'));files.extend(root.glob('*.log'))
    files.extend(root.glob('evaluation/*.json'));files.extend(root.glob('evaluation/*.log'))
    files.append(summary_path.with_name('ca1_summary.json'))
    for name in ('legacy','bounded'):
        folder=root/'ca1'/name
        files.extend(folder/item for item in ('best.pt','last.pt','protocol.json','initial.json','history.jsonl'))
    files.extend(root.glob('ca1/*.json'));files.extend(root.glob('ca1/*.log'))
    files.extend(root.glob('ca1/evaluation/*.json'));files.extend(root.glob('ca1/evaluation/*.log'))
    recovery='''# Recovery

Extract both archives into one directory; their SHoEbox folders merge.
The retained parent is artifacts/hippocampus/parent.pt. The new fixed-budget
endpoint is artifacts/hippocampus_binding/event_expert/last.pt; best.pt is the
validation-selected candidate. The original hippocampus endpoint remains at
artifacts/hippocampus/C/last.pt. Optimizer, RNG, stream and carried state are saved.
The matched 16-update controller continuations are in
artifacts/hippocampus_binding/ca1/{legacy,bounded}/last.pt. Their common initial
weights/state are the event_expert update64 parent already included above.

See docs/hippocampus_binding.md for training, evaluation and recovery commands.
SHA256SUMS verifies every included file. Final test predictions and diagnostics
are under artifacts/hippocampus_binding/evaluation and the adjacent JSON files.
'''
    result.append(make_archive(a.out/'SHoEbox_Hippocampus_Binding_Checkpoints.zip',files,'SHoEbox/',
                               {'RECOVERY.md':recovery,'SOURCE_REVISION.txt':revision}))
    (a.out/'binding_package_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()

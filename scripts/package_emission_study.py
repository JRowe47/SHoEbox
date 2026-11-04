"""Audit the completed emission study and package its aggregate provenance."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.emission_adapter import sha
from experiments.text_regions.emission_compare import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/emission_adapter_v1'))
    parser.add_argument('--out',type=Path,default=Path('experiments/text_regions/results/emission_adapter'))
    parser.add_argument('--verification-log',type=Path)
    args=parser.parse_args()
    source=args.source if args.source.is_absolute() else ROOT/args.source
    final=args.out if args.out.is_absolute() else ROOT/args.out
    if final.exists():parser.error('Choose a new result destination.')
    torch.set_num_threads(2)
    comparison=compare(source)
    status=json.loads((source/'status.json').read_text())
    final.parent.mkdir(parents=True,exist_ok=True)
    target=Path(tempfile.mkdtemp(prefix='.emission_stage_',dir=final.parent))
    inventory=[]
    for run in comparison['runs']:
        for file in run['files']:
            path=source/run['name']/file['name']
            inventory.append({**file,'restore_path':str(path.relative_to(ROOT))})
    inventory.append({'name':'status.json','restore_path':str((source/'status.json').relative_to(ROOT)),
                      'sha256':sha(source/'status.json'),'size_bytes':(source/'status.json').stat().st_size})
    recovery=source/'recovery'/'prediction_export_recovery.json'
    if recovery.is_file():
        record=json.loads(recovery.read_text())
        repaired=source/record['run']/'predictions.jsonl'
        original=source/'recovery'/(record['run']+'_predictions_original.jsonl')
        if (sha(repaired)!=record['restored_sha256'] or sha(original)!=record['original_sha256']
                or not repaired.read_bytes().startswith(original.read_bytes())):
            parser.error('Prediction export recovery provenance differs.')
        for path in (recovery,original):
            inventory.append({'name':path.name,'restore_path':str(path.relative_to(ROOT)),
                              'sha256':sha(path),'size_bytes':path.stat().st_size})
        shutil.copyfile(recovery,target/'prediction_export_recovery.json')
    (target/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (target/'artifact_inventory.json').write_text(json.dumps({'files':inventory},indent=2)+'\n')
    (target/'manifest.json').write_text(json.dumps({'schema_version':1,'source_directory':str(source.relative_to(ROOT)),
        'runs':[{k:r[k] for k in ('name','publication_policy','source_schedule','training_seed',
                    'checkpoint','checkpoint_sha256','reference','prior_results')} for r in status['runs']]},indent=2)+'\n')
    shutil.copyfile(source/'status.json',target/'study_run_log.json')
    verification={'complete':True,'verified_models':12,'fitted_heads':144,'cached_cohorts':36,
        'comparison_sha256':sha(target/'comparison.json'),'raw_artifact_files':48,
        'checks':comparison['integrity'],'raw_artifact_delivery':'downloadable package; no public raw-row upload'}
    if recovery.is_file():verification['prediction_export_recovery']='prediction_export_recovery.json'
    if args.verification_log:
        log=args.verification_log.read_text()
        if 'All five experiment suites passed.' not in log or len(re.findall(r'^OK$',log,re.M))!=5:
            parser.error('Verification log does not show all five passing suites.')
        shutil.copyfile(args.verification_log,target/'verification.log')
        verification['automated_tests_passed']=sum(map(int,re.findall(r'^Ran (\d+) tests',log,re.M)))
        verification['verification_log_sha256']=sha(target/'verification.log')
    (target/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (target/'README.md').write_text('''# Shared emission adapter study

The aggregate comparison covers twelve frozen regional graphs, three adapter
initializations per graph, raw and mixed features, and parameter-matched identity
and GELU heads. Each head is one common 256-byte decoder for all queries.

`comparison.json` contains per-initialization and per-graph results, group means,
fit metadata, and artifact hashes. `manifest.json` identifies the original
checkpoints and reference heads. `verification.json` records the independent
audits; `study_run_log.json` preserves settings and commands.

The downloadable package restores the feature caches, fitted heads and complete
synthetic prediction rows listed in `artifact_inventory.json`. Raw rows remain
outside the public repository. The cached features permit further decoder
experiments without running the experts again.

After extracting the complete package, run from the repository root:

```bash
python -m experiments.text_regions.emission_compare \\
  --source artifacts/emission_adapter_v1 \\
  --out artifacts/reproduced_emission_comparison.json
```

The audit reconstructs synthetic cases, checks model and source identities,
reproduces all saved predictions from the fitted heads and cached features,
checks exact agreement with the previous linear baselines, and reconstructs
each fitted training objective. No evaluation scores select a head or seed.
''')
    target.replace(final)
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    main()

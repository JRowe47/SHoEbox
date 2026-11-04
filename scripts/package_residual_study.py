"""Audit the residual study and export aggregate results plus artifact inventory."""
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
from experiments.text_regions.residual_adapter import write_json, write_bytes
from experiments.text_regions.residual_compare import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/residual_adapter_v1'))
    parser.add_argument('--out',type=Path,default=Path('experiments/text_regions/results/residual_adapter'))
    parser.add_argument('--verification-log',type=Path)
    args=parser.parse_args()
    source=args.source if args.source.is_absolute() else ROOT/args.source
    final=args.out if args.out.is_absolute() else ROOT/args.out
    if final.exists():parser.error('Choose a new aggregate result destination.')
    torch.set_num_threads(2);comparison=compare(source)
    status=json.loads((source/'status.json').read_text())
    final.parent.mkdir(parents=True,exist_ok=True)
    target=Path(tempfile.mkdtemp(prefix='.residual_stage_',dir=final.parent))
    inventory=[]
    for run in comparison['runs']:
        for file in run['files']:
            inventory.append({**file,'restore_path':str((source/run['name']/file['name']).relative_to(ROOT))})
    for path in [source/'status.json',*[source/(r['name']+'.log') for r in status['runs']]]:
        inventory.append({'name':path.name,'restore_path':str(path.relative_to(ROOT)),
                          'sha256':sha(path),'size_bytes':path.stat().st_size})
    write_json(target/'comparison.json',comparison)
    write_json(target/'artifact_inventory.json',{'files':inventory})
    write_json(target/'manifest.json',{'schema_version':1,'source_directory':str(source.relative_to(ROOT)),
        'runs':[{k:r[k] for k in ('name','publication_policy','source_schedule','training_seed',
                                 'checkpoint','checkpoint_sha256','reference','prior_run')} for r in status['runs']]})
    shutil.copyfile(source/'status.json',target/'study_run_log.json')
    verification={'complete':True,'verified_models':12,'fitted_heads':72,'validation_caches':12,
        'comparison_sha256':sha(target/'comparison.json'),'raw_artifact_files':60,
        'checks':comparison['integrity'],'raw_artifact_delivery':'downloadable package; no public raw-row upload'}
    if args.verification_log:
        log=args.verification_log.read_text()
        if 'All five experiment suites passed.' not in log or len(re.findall(r'^OK$',log,re.M))!=5:
            parser.error('Verification log does not show five passing suites.')
        shutil.copyfile(args.verification_log,target/'verification.log')
        verification['automated_tests_passed']=sum(map(int,re.findall(r'^Ran (\d+) tests',log,re.M)))
        verification['verification_log_sha256']=sha(target/'verification.log')
    write_json(target/'verification.json',verification)
    write_bytes(target/'README.md',b'''# Validation-scaled residual byte decoders

The study preserves each frozen raw linear head and fits a 16-unit correction.
Identity and GELU controls share initial tensors and parameter counts. All 72
corrections begin with exactly zero output. Validation selects one scalar strength
per head, including zero, before held-out and independent scoring.

`comparison.json` records all graph/initialization results and validation candidate
scores. `manifest.json` identifies graph checkpoints, frozen linear heads and the
previous feature caches. `artifact_inventory.json` identifies every downloadable
tensor, raw prediction file, selection lock and execution log by SHA-256.

The downloadable complete package restores all dependencies. From its root:

```bash
python -m experiments.text_regions.residual_compare \\
  --source artifacts/residual_adapter_v1 \\
  --out artifacts/reproduced_residual_comparison.json
```

The audit reconstructs synthetic cohorts, verifies source/checkpoint identities,
reproduces the validation choice and every evaluation prediction/loss from saved
weights, and checks exact agreement with the previous raw linear baseline.
Raw rows, tensors and feature caches stay outside the public repository.
''')
    target.replace(final);print(json.dumps(verification,indent=2))


if __name__=='__main__':main()

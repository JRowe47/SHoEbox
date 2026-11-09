"""Package numerical handoff; repository source is delivered as a separate patch."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checks',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    selected=[]
    for name in ('hybrid_legacy','hybrid_latent','hybrid_calibration'):
        selected.extend(p for p in (root/'artifacts'/name).rglob('*') if p.is_file())
    selected.append(root/'artifacts/joint_fresh256/last.pt')
    for name in ('hybrid_matched262k','hybrid_calibration8k','fineweb_phase2_raw'):
        selected.extend(p for p in (root/'data'/name).rglob('*') if p.is_file())
    for name in ('hybrid_legacy','hybrid_latent','hybrid_calibration','hybrid_cli_verification_final'):
        selected.append(root/'artifacts'/(name+'.log'))
    verified=root/'artifacts/hybrid_cli_verification_final'
    selected.append(verified/'verification.json')
    selected.extend(verified.glob('*.log'))
    args.out.mkdir(parents=True,exist_ok=True)
    digest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
    text='''# SHoEbox hybrid learning checkpoint handoff

Original shared seed: artifacts/joint_fresh256/last.pt
Calibrated Phase A seed: artifacts/hybrid_calibration/seed.pt
Legacy endpoint: artifacts/hybrid_legacy/last.pt
Phase A endpoint: artifacts/hybrid_latent/last.pt

Endpoints retain model tensors, named Adam states, RNG, complete recurrent state,
normalizer statistics, objective configuration, histories and data cursors.
Each endpoint is 896 lifetime updates / 1,310,720 optimizer byte exposures.

Check out JRowe47/SHoEbox implementation commit
67ac68c65feae1a2328dd0d3d94a26e2089a2f37 (feat/hierarchy-pilot), or apply the
accompanying patch to b4507b4. Extract this archive alongside
that checkout. Source is intentionally delivered separately from numeric data.
See docs/hybrid_learning.md and the accompanying report for commands.
The existing objective JSON must be supplied for latent continuation.

The 128-update comparison budget is complete. Adding training requires an
explicit --initialize-from continuation with a new update budget; --resume
preserves the existing budget and source/data/objective protocol exactly.

All cached source rows include their original provenance in sources.json.
Training IDs/hashes exclude historical and held-out sources. Calibration is
separate from both comparison training and held-out evaluation.
'''
    target=args.out/'SHoEbox_Hybrid_Learning_Checkpoints.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in selected:
            archive.write(path,'SHoEbox/'+str(path.relative_to(root)))
        archive.writestr('SHoEbox/HYBRID_CHECKPOINTS_README.md',text)
        archive.writestr('SHoEbox/HYBRID_SHA256SUMS.json',json.dumps(digest,indent=2)+'\n')
        checks=args.checks
        archive.write(checks,'SHoEbox/artifacts/hybrid_verification/checks.log')
    with zipfile.ZipFile(target) as archive:
        bad=archive.testzip()
        if bad:raise ValueError(bad)
    print(json.dumps(dict(path=str(target.resolve()),files=len(selected)+3,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))


if __name__=='__main__':
    main()

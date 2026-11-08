"""Ordinary source/checkpoint archives with independently verifiable checksums."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def archive(destination,files):
    files=sorted(set(p for p in files if p.is_file()))
    manifest={}
    temporary=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            data=p.read_bytes();name=str(p.relative_to(ROOT))
            manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            z.writestr(name,data)
        z.writestr('LACE_ARCHIVE_CHECKSUMS.json',json.dumps(manifest,indent=2)+'\n')
    temporary.replace(destination)
    with zipfile.ZipFile(destination) as z:assert z.testzip() is None
    return dict(path=str(destination),bytes=destination.stat().st_size,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),members=len(files)+1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=ROOT/'artifacts/lace_separation'
    complete=json.loads((root/'training_complete.json').read_text());assert complete['matched_ledgers']
    summary=json.loads((root/'summary.json').read_text())
    assert summary['evaluation_audit']['complete'] and summary['evaluation_audit']['sections']==38
    assert json.loads((root/'pipeline.json').read_text())['stage']=='complete'
    a.out.mkdir(parents=True,exist_ok=True)
    tracked=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    source=[ROOT/x for x in tracked if x and not x.startswith('artifacts/')]
    evidence=[p for p in root.rglob('*') if p.suffix in ('.json','.jsonl','.log')]
    parents=[ROOT/'artifacts/hippocampus/parent.pt',ROOT/'artifacts/hippocampus_binding/ca1/bounded/last.pt']
    corpus=ROOT/'data/tiny_shakespeare.txt'
    checkpoint_files=[p for p in root.rglob('*') if p.suffix in ('.pt','.json','.jsonl','.log')]
    outputs=[archive(a.out/'SHoEbox_Lace_Separation_Source.zip',source+evidence+[corpus]),
        archive(a.out/'SHoEbox_Lace_Separation_Checkpoints.zip',checkpoint_files+parents+[corpus])]
    (a.out/'SHoEbox_Lace_Separation_Archives.json').write_text(json.dumps(outputs,indent=2)+'\n')
    print(json.dumps(outputs))

if __name__=='__main__':main()

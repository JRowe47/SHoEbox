"""Isolated CPU thread/batch sweep; measure before choosing parallel settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=1)
    args=p.parse_args()
    if args.out.exists():p.error('Choose a new output directory.')
    args.out.mkdir(parents=True)
    settings=[('dense',4,1,1),('sparse',4,1,1),('sparse',4,2,1),('sparse',4,4,1),
              ('sparse',4,1,2),('sparse',16,1,1),('sparse',32,1,1),('sparse',32,2,1)]
    rows=[]
    for mode,batch,threads,workers in settings:
        name=f'{mode}_b{batch}_t{threads}_w{workers}'
        command=[sys.executable,'-m','shoebox.cpu_run','--mode','benchmark',
                 '--preset','wide' if mode=='dense' else 'cpu',
                 '--backend','fused' if mode=='dense' else 'cpu',
                 '--data',str(args.data.resolve()),'--out',str((args.out/name).resolve()),
                 '--targets','2048','--context','32','--batch',str(batch),
                 '--threads',str(threads),'--workers',str(workers),'--repeats',str(args.repeats)]
        with (args.out/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed; see its log.')
        record=json.loads((args.out/name/'results.json').read_text())
        row={k:record[k] for k in ('parameters','batch','threads','workers','median_bytes_per_second','peak_rss_mib')}
        row.update(name=name,inference_bytes_per_second=record['inference']['bytes_per_second'])
        rows.append(row)
        (args.out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()

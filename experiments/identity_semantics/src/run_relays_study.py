"""Validation-only relay rate selection, then matched two-seed 1M-byte runs."""
import argparse,json,os
from run_semantics_study import ROOT,launch,write

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['pilots','full'])
 a=p.parse_args();cpus=sorted(os.sched_getaffinity(0));jobs=[]
 if a.stage=='pilots':
  for rate in (.02,.04,.08):
   jobs.append(dict(kernel='relays',seed=3,peak=rate,tokens=100000,out=f'relay_pilot_{rate}',core=cpus[len(jobs)%3]))
  launch(jobs,min(3,len(cpus)))
  rows=[json.loads((ROOT/'results'/j['out']/'training.json').read_text()) for j in jobs]
  best=min(rows,key=lambda r:r['validation'][-1]['nll'])
  write(ROOT/'results/relay_rate_selection.json',dict(peak_lr=best['peak_lr'],validation_nll=best['validation'][-1]['nll'],pilot_seed=3,pilot_targets=100000,candidates=[dict(rate=r['peak_lr'],validation_nll=r['validation'][-1]['nll']) for r in rows]))
 else:
  peak=json.loads((ROOT/'results/relay_rate_selection.json').read_text())['peak_lr']
  for seed in (1,2):
   for kernel,rate in [('semantics',.08),('relays',peak)]:
    jobs.append(dict(kernel=kernel,seed=seed,peak=rate,tokens=1000000,out=f'relay_full_{kernel}_seed{seed}',core=cpus[len(jobs)%len(cpus)]))
  launch(jobs,min(4,len(cpus)))

if __name__=='__main__':main()

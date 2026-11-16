"""Separate changed/untouched-key scores and explicitly measured shortcuts."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .backend import Model
from .tasks import cases,worlds
from .run import write
from experiments.mad.tasks import MAP,SEP,END,VALUE,KEY

def shortcuts(cs):
    correct={k:[] for k in ('constant_0','constant_1','first_value','latest_value')}
    for c in cs:
        d=c.ep.data;values=[d[i+3] for i in range(len(d)-4) if d[i]==MAP and d[i+2]==SEP and d[i+4]==END]
        for name,value in [('constant_0',VALUE),('constant_1',VALUE+1),('first_value',values[0]),('latest_value',values[-1])]:
            correct[name].append(all(value==t for t in c.expected))
    return {name:float(np.mean(v)) for name,v in correct.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=dict(shortcuts={condition:shortcuts(cases('dev',128,condition=condition)) for condition in ('pad0','compete16','overwrite16')},rows=[],inference_bytes=0,padding_bytes=0)
    start=time.process_time();cs=cases('dev',128,condition='overwrite16');ws=worlds('dev',128)
    for seed in (1,7,19):
        for arm in ('native','e8','a1'):
            with Model.loads((a.runs/str(seed)/(arm+'.qr')).read_bytes(),'a1' if arm=='a1' else 'e8') as m,m.clone() as f:
                changed=[];untouched=[]
                for i,c in enumerate(cs):
                    r=f.score(c.ep,2,2);key=KEY+ws[i//4].records[0][0]
                    for j,q in enumerate(c.query_keys):(changed if q==key else untouched).append(r[7+j])
                    result['inference_bytes']+=len(c.ep.data)-1;result['padding_bytes']+=c.ep.data[:-1].count(9)
                result['rows'].append(dict(seed=seed,arm=arm,changed=float(np.mean(changed)),untouched=float(np.mean(untouched))))
    result['cpu_seconds']=time.process_time()-start;result['overwrite_qualification_valid']=False
    result['reason']='Binary overwrite makes both answers equal to the latest value, so a query-free latest-value shortcut is perfect. Original failed gate and all costs are preserved; use a fresh three-value update fixture next.'
    write(a.out,result);print(result)
if __name__=='__main__':main()

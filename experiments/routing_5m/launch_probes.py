"""Use the final screen's CPU for offline probes after its training ends."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
p.add_argument('--audit-data',type=Path,required=True);p.add_argument('--audit-protocol',type=Path,required=True);a=p.parse_args()
def ready(arm,seed):
    path=a.work/f'{arm}_s{seed}/results.json'
    if not path.exists():return False
    d=json.loads(path.read_text());return d.get('status')=='complete' and d['targets']==20_000_000
while not ready('C_warmup_only',7):time.sleep(1)
todo=[(arm,seed) for seed in (7,1,19) for arm in ('D','C','D_warmup_only','C_warmup_only')]
while todo:
    selected=next(((arm,seed) for arm,seed in todo if ready(arm,seed)),None)
    if selected is None:time.sleep(1);continue
    arm,seed=selected;todo.remove(selected)
    subprocess.run([sys.executable,str(Path(__file__).with_name('probe.py')),
        '--work',str(a.work),'--data',str(a.data),'--audit-data',str(a.audit_data),
        '--audit-protocol',str(a.audit_protocol),'--arm',arm,'--seed',str(seed),'--cpu','8'],check=True)

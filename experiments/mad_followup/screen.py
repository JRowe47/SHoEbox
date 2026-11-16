"""Additional calibrated native-byte procedural tasks."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from experiments.mad_capabilities.screen import plan,compact
from experiments.mad.run import execute,atomic_json
from experiments.mad.tasks import TaskSpec

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=("variable_tracking","composition","pointer_chasing"),required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    settings=({"one_hop":{"depth":1},"base":{},"deep":{"length":8,"depth":4}}
              if args.task=="pointer_chasing" else {"one_op":{"length":1},"two_ops":{"length":2},"base":{}})
    config=plan(args.task,args.out,True)
    base=config["cells"][0];config["cells"]=[]
    for name,changes in settings.items():
        cell=copy.deepcopy(base);cell["setting"]=name
        cell["spec"]=asdict(replace(TaskSpec(**cell["spec"]),**changes))
        config["cells"].append(cell)
    config["settings"]=list(settings);config["fits"]=len(settings)*3
    config["total_training_bytes"]=config["fits"]*config["budget"]
    config["campaign_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    calibration=execute(config,args.out/"difficulty.json")
    choice=calibration["summary"]["calibration"][0]
    setting=choice["setting"] or next(iter(settings))
    selected=next(c for c in calibration["cells"] if c["setting"]==setting)
    paired=plan(args.task,args.out,False);paired["settings"]=[setting]
    paired["calibration_decision"]=choice;paired["campaign_sha256"]=config["campaign_sha256"]
    for cell in paired["cells"]:
        cell["setting"]=setting;cell["spec"]=selected["spec"]
    atomic_json(args.out/"selection.json",dict(decision=choice,paired_setting=setting,test_used_for_selection=False,diagnostic_only=choice["status"]!="calibrated"))
    print("DIFFICULTY",json.dumps(dict(task=args.task,decision=choice,paired_setting=setting)),flush=True)
    compact(execute(paired,args.out/"paired.json"))

if __name__=="__main__":main()

"""Calibrate difficulty before the paired loop rerun; preserve all raw cells."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from experiments.mad.run import make_plan, execute, atomic_json
from experiments.mad.tasks import TaskSpec
from experiments.mad.metrics import calibrate


def plan(task, out, calibration):
    args=argparse.Namespace(preset="screen",tasks=[task],settings=["base"],sizes=[500000],
        seeds=[1,7,19],steps=[0] if calibration else [0,2,4],
        lr_scales=[1.] if calibration else [.5,1.,2.],
        auxiliary=[.1] if calibration else [0.,.1],
        budget=250000,eval_every=50000,cases=128,train_examples=256,audit_probes=not calibration)
    return make_plan(args)


def compact(result):
    for cell in result["cells"]:
        rows=[]
        for r in cell["selected_runs"]:
            row={k:r[k] for k in ("seed","lr_scale","auxiliary","bytes","cpu_seconds","bytes_per_cpu_second")}
            row.update(dev=r["curve"][-1]["dev"]["task_accuracy"],
                       test_accuracy=r["test"]["task_accuracy"],test_nll=r["test"]["task_nll"],
                       exact=r["test"]["exact_answer_accuracy"],native_accuracy=r["test"]["native_byte_accuracy"],
                       parameters=r["inventory"]["parameters"])
            if "interventions" in r:
                row["interventions"]={k:{"accuracy":v["task_accuracy"],"nll":v["task_nll"],"permuted":v["active_experts_permuted"]} for k,v in r["interventions"].items()}
                row["depth_audit"]={k:v["task_accuracy"] for k,v in r["depth_audit"].items()}
                row["delay_ood"]=r["delay_ood"]["metrics"]["task_accuracy"]
            rows.append(row)
        print("SCREEN_CELL",json.dumps(dict(task=cell["task"],setting=cell["setting"],steps=cell["steps"],hyperparameters=cell["selected_hyperparameters"],tuning_cpu=cell["total_tuning_cpu_seconds"],rows=rows)),flush=True)
    print("SCREEN_SUMMARY",json.dumps(result["summary"]),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=("compression","selective_copy","recall"),required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    settings={
        "compression":{"base":{},"length":{"length":8},"vocabulary":{"vocab":16},"delay":{"delay":128}},
        "selective_copy":{"easy":{"length":2,"noise":.25},"base":{},"length":{"length":8}},
        "recall":{"easy":{"length":2},"base":{}}
    }[args.task]
    config=plan(args.task,args.out,True)
    base=config["cells"][0]
    config["cells"]=[]
    for name,changes in settings.items():
        cell=copy.deepcopy(base)
        cell["setting"]=name
        cell["spec"]=asdict(replace(TaskSpec(**cell["spec"]),**changes))
        config["cells"].append(cell)
    config["settings"]=list(settings)
    config["fits"]=len(config["cells"])*3
    config["total_training_bytes"]=config["fits"]*config["budget"]
    config["campaign_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    calibration=execute(config,args.out/"difficulty.json")
    choice=calibration["summary"]["calibration"][0]
    # Paired diagnostic on easiest registered setting when control is unresolved.
    setting=choice["setting"] or ("easy" if "easy" in settings else "length")
    chosen=next(c for c in calibration["cells"] if c["setting"]==setting)
    paired=plan(args.task,args.out,False)
    paired["settings"]=[setting]
    paired["calibration_decision"]=choice
    paired["campaign_sha256"]=config["campaign_sha256"]
    for cell in paired["cells"]:
        cell["setting"]=setting
        cell["spec"]=chosen["spec"]
    atomic_json(args.out/"selection.json",dict(decision=choice,paired_setting=setting,
                test_used_for_selection=False,diagnostic_only=choice["status"]!="calibrated"))
    print("DIFFICULTY",json.dumps(dict(task=args.task,decision=choice,paired_setting=setting)),flush=True)
    result=execute(paired,args.out/"paired.json")
    compact(result)


if __name__=="__main__":
    main()

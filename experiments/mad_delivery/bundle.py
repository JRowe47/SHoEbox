"""Combine this campaign's complete evidence into a report and primary PNG."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.mad_followup.baselines import audit

ROOT=Path("evidence")
OUT=Path("reports/mad_capabilities")
OUT.mkdir(parents=True,exist_ok=True)

def mean(rows, fn):
    return float(np.mean([fn(r) for r in rows]))

def pct(x):
    return f"{100*x:.2f}%"

calibration=[]
for path in sorted((ROOT/"calibration").glob("*.json")):
    d=json.loads(path.read_text())
    if d.get("status")=="complete":
        calibration.extend(d["cells"])
paired=[]
for path in sorted(ROOT.rglob("paired.json")):
    d=json.loads(path.read_text())
    if d.get("status")=="complete":
        paired.append((path,d))
states=[]
for path in sorted(ROOT.rglob("state_*.json")):
    d=json.loads(path.read_text())
    if d.get("status")=="complete":
        states.append(d)
if len(calibration)!=9 or len(paired)!=6 or len(states)!=2:
    raise RuntimeError(f"Incomplete evidence: calibration={len(calibration)}, paired={len(paired)}, states={len(states)}")

baselines=audit(ROOT)
(ROOT/"shortcut_baselines.json").write_text(json.dumps(baselines,indent=2)+"\n")
shortcut_name={"recall":"query_blind_majority","selective_copy":"query_blind_majority",
 "compression":"query_blind_majority","variable_tracking":"initial_binding_ignore_updates",
 "composition":"initial_binding_ignore_updates","pointer_chasing":"uniform_nonstart_node_distribution"}
summary=dict(calibration=[],loop_comparisons=[],state_comparisons=[],fits=0,training_bytes=0,
            runtime_base="6384d8ca072075ec58010b4c3d5964e9e60fcd3e",
            production_defaults_changed=False,language_guardrail_run=False)
for c in calibration:
    rs=c["selected_runs"]
    summary["calibration"].append(dict(task=c["task"],dev_accuracy=mean(rs,lambda r:r["curve"][-1]["dev"]["task_accuracy"]),
      per_seed=[r["curve"][-1]["dev"]["task_accuracy"] for r in rs]))
for path,d in paired:
    task=d["cells"][0]["task"]
    baseline=next(b for b in baselines if b["source"]==str(path) and b["steps"]==0)["baselines"]
    record=dict(task=task,setting=d["cells"][0]["setting"],calibration=d["config"]["calibration_decision"],
        shortcut_name=shortcut_name[task],shortcut_accuracy=baseline[shortcut_name[task]]["token_accuracy"],all_shortcuts=baseline,
        arms=[],comparisons=d["summary"]["comparisons"])
    for c in d["cells"]:
        rs=c["selected_runs"]
        record["arms"].append(dict(steps=c["steps"],test_accuracy=mean(rs,lambda r:r["test"]["task_accuracy"]),
            exact_accuracy=mean(rs,lambda r:r["test"]["exact_answer_accuracy"]),
            test_nll=mean(rs,lambda r:r["test"]["task_nll"]),dev_nll=mean(rs,lambda r:r["curve"][-1]["dev"]["task_nll"]),
            bytes_per_cpu_second=mean(rs,lambda r:r["bytes_per_cpu_second"]),
            parameters=rs[0]["inventory"]["parameters"],hyperparameters=c["selected_hyperparameters"],
            tuning_cpu_seconds=c["total_tuning_cpu_seconds"],
            memory_accuracy_drop=mean(rs,lambda r:r["test"]["task_accuracy"]-r["interventions"]["no_memory"]["task_accuracy"]),
            l2_permutation_accuracy_drop=mean(rs,lambda r:r["test"]["task_accuracy"]-r["interventions"]["permute_l2"]["task_accuracy"]),
            l3_permutation_accuracy_drop=mean(rs,lambda r:r["test"]["task_accuracy"]-r["interventions"]["permute_l3"]["task_accuracy"]),
            delay_ood_accuracy=mean(rs,lambda r:r["delay_ood"]["metrics"]["task_accuracy"]),
            per_seed=[{k:r[k] for k in ("seed","lr_scale","auxiliary","bytes","cpu_seconds","bytes_per_cpu_second")}|{
              "test_accuracy":r["test"]["task_accuracy"],"test_nll":r["test"]["task_nll"],"exact":r["test"]["exact_answer_accuracy"],
              "depth_audit":{k:v["task_accuracy"] for k,v in r["depth_audit"].items()},
              "interventions":{k:{"accuracy":v["task_accuracy"],"nll":v["task_nll"],"active_permuted":v["active_experts_permuted"]} for k,v in r["interventions"].items()}} for r in rs]))
    summary["loop_comparisons"].append(record)
for d in states:
    record=dict(size=d["config"]["size"],calibration=d["calibration"],comparisons=d["comparisons"],
        initial_binding_baseline=d["initial_binding_baseline"],arms=[])
    for arm in ("final_only","head_only","state_aux"):
        rs=[r for r in d["selected_runs"] if r["arm"]==arm]
        record["arms"].append(dict(arm=arm,parameters=rs[0]["parameters"],
            test_accuracy=mean(rs,lambda r:r["test"]["accuracy"]),
            long_accuracy=mean(rs,lambda r:r["long"]["accuracy"]),
            standardized_probe=mean(rs,lambda r:r["frozen_standardized_probes"]["after_query"]["accuracy"]),
            before_query_probe=mean(rs,lambda r:r["frozen_standardized_probes"]["before_query"]["accuracy"]),
            bytes_per_cpu_second=mean(rs,lambda r:r["bytes_per_cpu_second"]),
            mean_replacement_nll_delta=mean(rs,lambda r:r["mean_replacement"]["nll"]-r["test"]["nll"]),
            per_seed=[dict(seed=r["seed"],test_accuracy=r["test"]["accuracy"],test_nll=r["test"]["nll"],
                           long_accuracy=r["long"]["accuracy"],probe_accuracy=r["frozen_standardized_probes"]["after_query"]["accuracy"]) for r in rs]))
    summary["state_comparisons"].append(record)
# Count each trained fit once. Selected rows duplicate existing fit dictionaries.
for path in ROOT.rglob("*.json"):
    try:d=json.loads(path.read_text())
    except (ValueError,OSError):continue
    if not isinstance(d,dict):continue
    if "cells" in d:
        rs=[r for c in d["cells"] for r in c["runs"]]
    elif "calibration_fits" in d:
        rs=d["calibration_fits"]+[r for r in d["selected_runs"] if r["arm"]!="final_only"]
    else:continue
    summary["fits"]+=len(rs)
    summary["training_bytes"]+=sum(r["bytes"] for r in rs)

# Evidence table, also supplied in machine-readable form.
csvrows=[]
for task in summary["loop_comparisons"]:
    for arm in task["arms"]:
        csvrows.append({k:arm[k] for k in ("steps","parameters","test_accuracy","exact_accuracy","test_nll","bytes_per_cpu_second","memory_accuracy_drop","l2_permutation_accuracy_drop","l3_permutation_accuracy_drop","delay_ood_accuracy")}|{
            "task":task["task"],"setting":task["setting"],"shortcut_baseline":task["shortcut_accuracy"]})
with (OUT/"SHoEbox_MAD_Capability_Results.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
(OUT/"SHoEbox_MAD_Capability_Summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("FINAL_SUMMARY",json.dumps(summary),flush=True)

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(2,2,figsize=(16,11),layout="constrained")
colors=["#315d99","#df8b21","#789441"]
labels=[r["task"].replace("_"," ") for r in summary["calibration"]]
vals=[r["dev_accuracy"]*100 for r in summary["calibration"]]
ax[0,0].barh(labels,vals,color="#315d99")
ax[0,0].axvline(12.5,color="#999",ls=":")
ax[0,0].set(xlim=(0,108),xlabel="Development answer accuracy (%)",title="Initial capability profile / 500K / 250K bytes")
for i,v in enumerate(vals):ax[0,0].text(v+1,i,f"{v:.1f}",va="center",fontsize=9)
tasks=summary["loop_comparisons"];x=np.arange(len(tasks))
for j,steps in enumerate((0,2,4)):
    values=[next(a for a in t["arms"] if a["steps"]==steps)["test_accuracy"]*100 for t in tasks]
    ax[0,1].bar(x+(j-1)*.24,values,width=.23,color=colors[j],label=f"{steps} extra passes")
ax[0,1].scatter(x,[t["shortcut_accuracy"]*100 for t in tasks],marker="D",color="#b32433",s=25,label="Simple task shortcut",zorder=4)
ax[0,1].set(xticks=x,xticklabels=[t["task"].replace("_","\n") for t in tasks],ylim=(0,100),ylabel="Held-out answer accuracy (%)",title="Matched loop reruns / equal optimizer search")
ax[0,1].legend(frameon=False,fontsize=8,ncol=2)
for i,d in enumerate(summary["state_comparisons"]):
    for j,a in enumerate(d["arms"]):
        pos=i*4+j
        ax[1,0].bar(pos,a["test_accuracy"]*100,color=colors[j])
        ax[1,0].scatter([pos],[a["standardized_probe"]*100],marker="x",color="black",s=35)
ax[1,0].set(xticks=[0,1,2,4,5,6],xticklabels=["Final","Head","State","Final","Head","State"],ylim=(0,100),
    ylabel="Held-out answer accuracy (%)",title="State supervision / 1M (left), 5M (right)")
ax[1,0].plot([],[],marker="x",color="black",ls="",label="Frozen standardized probe")
ax[1,0].axhline(12.5,color="#999",ls=":");ax[1,0].legend(frameon=False,fontsize=8)
for j,steps in enumerate((2,4)):
    ratios=[]
    for t in tasks:
        c=next(a for a in t["arms"] if a["steps"]==0)
        a=next(a for a in t["arms"] if a["steps"]==steps)
        ratios.append(a["bytes_per_cpu_second"]/c["bytes_per_cpu_second"])
    ax[1,1].bar(x+(j-.5)*.32,ratios,width=.3,color=colors[j+1],label=f"{steps} extra passes")
ax[1,1].axhline(1,color="#555",ls=":")
ax[1,1].set(xticks=x,xticklabels=[t["task"].replace("_","\n") for t in tasks],ylabel="Training throughput / corrected C",title="Measured compute cost")
ax[1,1].legend(frameon=False)
fig.suptitle("SHoEbox · MAD capability research\nThree seeds · held-out tasks · measured compute",fontsize=18,fontweight="bold")
fig.savefig(OUT/"SHoEbox_MAD_Capability_Results.png",dpi=170)
plt.close(fig)

lines=["# SHoEbox MAD capability research","","15 November 2026","","## Scope and completed work","",
 f"Completed **{summary['fits']} fits and {summary['training_bytes']:,} observed training bytes**. The campaign calibrates nine task families, reruns selective loops on six task families with three paired seeds and equal optimizer searches, and repeats the earlier intermediate-supervision question at 1M and 5M parameters.",
 "",
 "The runtime is corrected C at commit \x606384d8ca072075ec58010b4c3d5964e9e60fcd3e\x60. The interrupted interactive run yielded no recovered results and is excluded. All measurements below come from completed GitHub runner jobs. Production defaults are unchanged.",
 "",
 "## Initial capability profile","","These are development calibration scores at 500K parameters and 250K input bytes. The alphabet contains eight values, so uniform token chance is 12.5%.",
 "","| Task | Development accuracy |","|---|---:|"]
for r in summary["calibration"]:lines.append(f"| {r['task']} | {pct(r['dev_accuracy'])} |")
lines+=["","## Rerun of selective loops","","Extra passes execute only at completed L2/L3 events. Parameter count and persistent state capacity match across depths. Each depth gets the same six-policy grid: LR multiplier 0.5/1/2 by native auxiliary weight 0/0.1. Every fit uses 250K observed input bytes; the optimizer policy is selected by mean final development NLL across seeds 1, 7 and 19.",
 "",
 "Difficulty is selected using corrected-C development scores before the paired rerun. The table reports held-out answer-token accuracy, averaged over seeds. The last column is a simple task-specific shortcut evaluated on the same test cases; it is not a fitted SHoEbox model.",
 "","| Task / selected setting | 0 extra | 2 extra | 4 extra | Simple shortcut |","|---|---:|---:|---:|---:|"]
for t in tasks:
    values={a["steps"]:a["test_accuracy"] for a in t["arms"]}
    lines.append(f"| {t['task']} / {t['setting']} | {pct(values[0])} | {pct(values[2])} | {pct(values[4])} | {pct(t['shortcut_accuracy'])} |")
lines+=["","Shortcut definitions: recall/copy/compression use the most common supplied value while ignoring query or position; variable tracking/composition retain the initial queried binding and ignore updates; pointer chasing guesses uniformly among supplied non-start nodes. This last score is an expected accuracy, not a deterministic solver.",
 "",
 "### Paired seed differences","","Positive values favor loops. Intervals bootstrap the three paired training-seed differences (10,000 resamples).",
 "","| Task | Extra passes | Accuracy change, percentage points | 95% seed interval |","|---|---:|---:|---|"]
for t in tasks:
    for c in t["comparisons"]:
        diff=c["test_accuracy_delta"];lo,hi=diff["ci95"]
        lines.append(f"| {t['task']} | {c['steps']} | {100*diff['mean']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
lines+=["","### Frozen mechanism checks","","Drops in answer accuracy after removing addressed/retained reads or cycling active expert transformations. Positive means the original mechanism helped this fitted control.",
 "","| Task, corrected C | Remove memory reads | Permute L2 | Permute L3 |","|---|---:|---:|---:|"]
for t in tasks:
    a=next(a for a in t["arms"] if a["steps"]==0)
    lines.append(f"| {t['task']} | {100*a['memory_accuracy_drop']:+.2f} pp | {100*a['l2_permutation_accuracy_drop']:+.2f} pp | {100*a['l3_permutation_accuracy_drop']:+.2f} pp |")
lines+=["","Raw evidence also contains frozen 0/1/2/4-depth overrides, longer-delay tests, exact-answer rollout scores, native 256-byte scores, active permutation counts and compression zero-bottleneck tests.",
 "",
 "## Rerun of intermediate supervision","","This extension retains the earlier verified 64-byte program-record encoding and identical four-register linear head. It applies MAD-style difficulty calibration, finite matched training sets, three seeds, structural train/dev/test separation, measured-time comparisons and frozen interventions. It is separate from the canonical nine byte tasks. Each fit uses 1M bytes with no language mixing; the 4,128 head parameters are now deducted from the total budget.",
 "",
 "Final-only, extra-head-label and direct-state-credit arms share one fixed learning policy and identical final queried-register supervision. The intermediate head-only arm isolates extra head training; state_aux adds credit into L2/L3. Difficulty is selected from 1/2/4 operations using the final-only development control. If no difficulty calibrates, the one-operation result remains diagnostic.",
 "","| Size | Arm | Held-out accuracy | 8/12-op accuracy | Frozen standardized readout |","|---|---|---:|---:|---:|"]
for d in summary["state_comparisons"]:
    for a in d["arms"]:
        lines.append(f"| {d['size']:,} | {a['arm']} | {pct(a['test_accuracy'])} | {pct(a['long_accuracy'])} | {pct(a['standardized_probe'])} |")
lines+=["","The standardized probe is trained only on separate training-fold examples. It changes both feature conditioning and the fitting method, so its advantage does not isolate centering alone. The task head receives the queried register index; native byte answers are stored separately in the raw results.",
 "",
 "## Interpretation and next decision","","**Prioritize query-dependent binding, state readout and retention. The present shared-expert refinement loop does not earn promotion as a general capability improvement.**",
 "",
 "On calibrated compression, held-out accuracy declines from 70.77% without loops to 67.06% with two extra passes and 58.33% with four. All three seeds worsen at both depths. On the four-pass-trained compression checkpoints, disabling extra inference passes raises mean accuracy from 58.33% to 63.87%; even that remains below the no-loop-trained control.",
 "",
 "Two-binding recall is the positive exception: four-pass training improves held-out token accuracy from 55.86% to 60.81%, a +4.95-point paired mean, with seed interval [+0.78, +8.20]. Evaluating those same four-pass-trained models with no extra passes still scores 60.42%. Thus almost all of the improvement persists without the repeated inference computation; this supports a training effect, not useful additional reasoning at inference.",
 "",
 "Whole-episode recall remains weak: the four-pass model answers both queries correctly on 30.99% of episodes, while a query-blind control that simply emits the supplied values in their original order gets 53.12%. The token-accuracy improvement therefore does not establish reliable binding of each value to its query.",
 "",
 "Selective copy stays essentially unchanged (87.76% control, 88.02% with two extra passes, 86.33% with four). Composition and pointer chasing show no consistent gain; two-pass variable tracking worsens. The loop policies typically cost a few percent of training throughput.",
 "",
 "Retention is a concrete failure: on independently generated 64-byte-delay cases, the no-loop selective-copy control falls from 87.76% to 14.84%, and two-binding recall falls from 55.86% to 18.88%. This measures generalization to longer delays, rather than the ceiling after dedicated delay training. Removing addressed and retained reads also collapses copying, so the model uses these memories but does not retain the task robustly under this distribution shift.",
 "",
 "Direct intermediate state credit improves all three paired seeds over final-only supervision at both sizes: +2.60 points at 1M and +3.52 at 5M. The resulting mean accuracies are only 14.71% and 15.36%, versus 71.48% for retaining the initial queried binding and ignoring the update. The positive shift is real in these runs; the procedure is still unsolved. None of the 1/2/4-operation state-control settings entered the registered calibration band.",
 "",
 "The frozen standardized readouts recover approximately 31–33% from the direct-state models. There is recoverable information that the online head is failing to use, and additional information is still missing. More generic loop depth is unlikely to resolve either problem by itself.",
 "",
 "The shortcut controls also improve the development protocol: being above uniform chance is insufficient to establish the intended skill. Native variable tracking scores 29.95% against a 64.84% ignore-updates baseline; one-operation composition scores 28.91% against 75.78%. One-hop pointer accuracy is approximately the 33.33% node-set guessing baseline. Future calibration should record these controls on development data before any capability promotion.",
 "",
 "No natural-language guardrail was run, and no architecture was promoted.",
 "",
 "The next experiment should target the weakest measured operation that is still below its simple shortcut baseline: preserve separate key/value bindings, make retrieval depend on the query, and give upper computation a useful state-update target. Retain a readout-only control when testing additional working slots.",
 "",
 "## Verification, costs and reproduction","","Existing MAD tests passed on each task runner. The structured-state adapter passed its Python checks and its 19 native checks under ASan/UBSan. Selected checkpoint reloads reproduce development predictions exactly. Each paired run checks dataset hashes, input exposure and fitted/state capacity. Generator labels are verified by independent interpreters.",
 "",
 "Training time includes context resets, learning and retirement; excludes compilation, initialization, validation and checkpoint I/O. Matched-time comparisons use the last actually measured development checkpoint within the common CPU budget, with byte counts recorded. These finite training horizons are not saturation-floor estimates.",
 "",
 "- Base protocol: https://github.com/JRowe47/SHoEbox/blob/6384d8ca072075ec58010b4c3d5964e9e60fcd3e/experiments/mad/README.md",
 "- Calibration: https://github.com/JRowe47/SHoEbox/actions/runs/43787250957",
 "- Loop and state reruns: https://github.com/JRowe47/SHoEbox/actions/runs/43787654696",
 "- Procedural reruns: https://github.com/JRowe47/SHoEbox/actions/runs/43787902832",
 "- MAD research: https://proceedings.mlr.press/v235/poli24a.html",
 "",
 "Reproduction uses the registered scripts under experiments/mad_capabilities and experiments/mad_followup on the research/mad-capabilities branch. The source patch is included with the raw JSON evidence. Datasets, trained weights and compiled binaries are excluded.",
 ""]
(OUT/"SHoEbox_MAD_Capability_Research.md").write_text("\n".join(lines))
for p in OUT.iterdir():
    if p.is_file():shutil.copy2(p,ROOT/p.name)
print("REPORT_READY",str(OUT),flush=True)

readme=Path("experiments/mad_capabilities/README.md")
readme.write_text("""# MAD capability research — completed campaign

The completed report, summary and primary PNG are in
[reports/mad_capabilities](../../reports/mad_capabilities).

The campaign completed 435 fits and 131.25M observed training bytes. It includes
nine-task calibration, calibrated 0/2/4-pass comparisons on six task families
with equal six-policy optimizer searches and three seeds, and 1M/5M reruns of
intermediate state supervision. Corrected C remains the production runtime.

Key results: shared-expert loops harm compression, provide a small recall
training benefit that mostly survives with inference loops disabled, and do
not solve procedural tasks. Intermediate state credit improves all three seeds
at both model sizes, but final task accuracy remains near 15%. Query-dependent
binding, readout conditioning and retention are the next targets.

The initial host-disconnected launch supplied no recovered measurements.
All reported fits are completed remote jobs with exact checkpoint reloads.

Commands:
    python -m experiments.mad_capabilities.screen --task compression --out results/compression
    python -m experiments.mad_capabilities.state_campaign --size 1000000 --budget 1000000 --out results/state_1000000.json
    python -m experiments.mad_followup.screen --task variable_tracking --out results/variable_tracking
    python -m unittest experiments.mad_followup.test_baselines -v

This is SHoEbox's byte adaptation and registered state extension of MAD, not an
original MAD leaderboard score. Full task contracts remain in experiments/mad.
""")

(ROOT/"source.patch").write_bytes(subprocess.check_output(["git","diff","6384d8ca072075ec58010b4c3d5964e9e60fcd3e","--",".github/workflows/mad-capabilities.yml",".github/workflows/mad-procedural.yml","experiments/mad_capabilities","experiments/mad_followup","experiments/mad_delivery",".github/workflows/mad-delivery.yml"]))

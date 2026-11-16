"""Render compact measured results and source links; no fitted weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,required=True)
    args=p.parse_args()
    root=args.directory
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    rows=[]
    for path in root.rglob("paired.json"):
        d=json.loads(path.read_text())
        if d.get("status")!="complete":continue
        cells=d["cells"]
        fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout="constrained")
        colors={0:"#315d99",2:"#df8b21",4:"#789441"}
        for c in cells:
            k=c["steps"];rs=c["selected_runs"];color=colors[k]
            points=rs[0]["curve"]
            x=[p["bytes"]/1000 for p in points]
            ys=np.array([[p["dev"]["task_accuracy"] for p in r["curve"]] for r in rs])
            axes[0].plot(x,ys.mean(0)*100,label=f"{k} extra passes",color=color)
            axes[0].fill_between(x,ys.min(0)*100,ys.max(0)*100,color=color,alpha=.12)
            values=np.array([r["test"]["task_accuracy"] for r in rs])*100
            axes[1].bar(k,values.mean(),width=.8,color=color)
            axes[1].scatter([k]*len(values),values,color="black",s=14,zorder=3)
            speed=np.array([r["bytes_per_cpu_second"] for r in rs])/1000
            axes[2].bar(k,speed.mean(),width=.8,color=color)
            axes[2].scatter([k]*len(speed),speed,color="black",s=14,zorder=3)
            rows.append({"task":c["task"],"setting":c["setting"],"extra_passes":k,
                         "test_accuracy":float(values.mean()/100),
                         "test_nll":float(np.mean([r["test"]["task_nll"] for r in rs])),
                         "train_kbytes_per_cpu_second":float(speed.mean()),
                         "parameters":rs[0]["inventory"]["parameters"]})
        task=cells[0]["task"];setting=cells[0]["setting"]
        axes[0].set(xlabel="Training input (thousand bytes)",ylabel="Development accuracy (%)",ylim=(0,100))
        axes[0].legend(frameon=False)
        axes[1].set(xlabel="Extra upper passes",ylabel="Held-out answer accuracy (%)",ylim=(0,100),xticks=[0,2,4])
        axes[2].set(xlabel="Extra upper passes",ylabel="Training thousand bytes / CPU second",xticks=[0,2,4])
        fig.suptitle(f"SHoEbox MAD: {task} / {setting} — 3 paired seeds",fontsize=14,fontweight="bold")
        fig.savefig(path.parent/f"SHoEbox_MAD_{task}.png",dpi=180)
        plt.close(fig)
    for path in root.rglob("state_*.json"):
        d=json.loads(path.read_text())
        if d.get("status")!="complete":continue
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
        arms=["final_only","head_only","state_aux"]
        for i,arm in enumerate(arms):
            rs=[r for r in d["selected_runs"] if r["arm"]==arm]
            for ax,key,color in ((axes[0],"test","#315d99"),(axes[1],"long","#789441")):
                y=np.array([r[key]["accuracy"] for r in rs])*100
                ax.bar(i,y.mean(),color=color)
                ax.scatter([i]*len(y),y,color="black",s=16,zorder=3)
        for ax in axes:
            ax.set(xticks=range(3),xticklabels=["Final answer","Extra head labels","Extra state credit"],ylim=(0,100),ylabel="Answer accuracy (%)")
            ax.axhline(12.5,color="#8a8a8a",ls=":",label="Chance")
            ax.tick_params(axis="x",labelsize=9)
        axes[0].set_title("Held-out structural folds")
        axes[1].set_title("Longer programs (8/12 operations)")
        fig.suptitle(f"Intermediate supervision / {d['config']['size']:,} parameters / 3 seeds",fontweight="bold")
        fig.savefig(path.with_suffix(".png"),dpi=180)
        plt.close(fig)
    (root/"compact_summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    print("AGGREGATE",json.dumps(rows),flush=True)

if __name__=="__main__":
    main()

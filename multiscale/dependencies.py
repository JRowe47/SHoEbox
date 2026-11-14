"""Ordered distant-cue evaluation with matched suffixes and shifted groups."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .benchmark import atomic_json
from .model import Model


def example(bit, gap, shift=0, pair=b"AB"):
    cue = pair if bit == 0 else pair[::-1]
    return b"." * shift + cue + b"x" * gap + b"?" + bytes([48+bit]) + b"\n"


def score(model, *, gaps=(16,48,96,192), intervention=-1, memory="full"):
    rows=[]
    with model.inference_copy() as frozen:
        frozen.intervene(mask=intervention)
        if memory != "full":
            frozen.intervene_memory(memory)
        for gap in gaps:
            correct=0;loss=0.;pairs=0;differences=[]
            for shift in range(8):
                probs=[]
                for bit in (0,1):
                    sequence=example(bit,gap,shift);prefix=sequence[:-2]
                    frozen.reset_context()
                    frozen.run(prefix[:-1],prefix[1:])
                    logits=frozen.observe(prefix[-1])
                    p=logits[[48,49]].astype(float);p=np.exp(p-p.max());p/=p.sum()
                    correct+=int(np.argmax(p)==bit);loss-=np.log(max(1e-30,p[bit]));pairs+=1
                    probs.append(float(p[1]))
                differences.append(probs[1]-probs[0])
            rows.append(dict(gap=gap,accuracy=correct/pairs,binary_nll=loss/pairs,
                             mean_cue_probability_difference=float(np.mean(differences))))
    return rows


def main(args):
    os.sched_setaffinity(0,{args.cpu});rows=[]
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        # Test shifts 4..7 never appear in training. Both cues have exactly the
        # same child inventory; only their order identifies the later answer.
        train=b"".join(example(int(rng.integers(2)),int(rng.choice([16,48,96])),int(rng.integers(4)))
                       for _ in range(args.examples))
        for levels in args.levels:
            with Model(levels=levels,seed=seed,memory_profile=args.memory_profile) as m:
                m.configure(learning_rate=args.lr)
                t=time.perf_counter();nll=m.train(train);m.finish();seconds=time.perf_counter()-t
                row=dict(seed=seed,levels=levels,memory_profile=args.memory_profile,
                         train_bytes=len(train)-1,seconds=seconds,training_nll=nll,
                         scores=score(m),masks={str(l):score(m,intervention=l) for l in range(1,levels)})
                if args.memory_profile == "addressed":
                    row["memory_interventions"] = {mode:score(m,memory=mode) for mode in ("no_read","no_time","no_content")}
                rows.append(row);atomic_json(Path(args.out)/"dependencies.json",rows);print(json.dumps(row),flush=True)
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True)
    p.add_argument("--examples",type=int,default=6000);p.add_argument("--levels",nargs="+",type=int,default=[1,2,4])
    p.add_argument("--seeds",nargs="+",type=int,default=[7,17,29]);p.add_argument("--cpu",type=int,default=1)
    p.add_argument("--lr",type=float,default=.08)
    p.add_argument("--memory-profile",choices=["addressed","legacy"],default="addressed")
    main(p.parse_args())

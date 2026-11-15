"""Train, inspect and sample the independent SHoE runtime."""
import argparse
import json
from pathlib import Path
import time

from .model import Model


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    train=sub.add_parser("train")
    train.add_argument("--data",required=True);train.add_argument("--out",required=True)
    train.add_argument("--resume");train.add_argument("--levels",type=int,default=4)
    train.add_argument("--parameters",type=int,default=1_000_000);train.add_argument("--seed",type=int,default=7)
    train.add_argument("--targets",type=int);train.add_argument("--learning-rate",type=float,default=None)
    train.add_argument("--slow-rate",type=float);train.add_argument("--query-rate",type=float)
    train.add_argument("--head-rate",type=float)
    train.add_argument("--router",choices=["indexed","scan"],default=None,
                       help="fresh model: exact scan for C; indexed for historical A/profiles")
    train.add_argument("--routing-arm",choices=list("ABCD"),default=None,
                       help="fresh model: corrected C by default; explicit A/B/C/D remain reproducible")
    train.add_argument("--language-credit",type=float,nargs='+',
                       help="fresh model: one language-state credit gain per level; resume preserves saved gains")
    train.add_argument("--recall-credit",type=float,nargs='+',
                       help="fresh model: one retained-value credit gain per level; resume preserves saved gains")
    train.add_argument("--balance-policy",choices=["legacy","warmup"],default=None,
                       help="fresh C/D model: coverage warmup or historical residual balancing")
    train.add_argument("--semantic-laces",action=argparse.BooleanOptionalAction,default=None,
                       help="fresh model: historical A-only lace dispatch")
    train.add_argument("--associations",action="store_true")
    train.add_argument("--memory-profile",choices=["addressed","addressed_v4","addressed_v3","addressed_v2","legacy"],default="addressed",
                       help="fresh-model policy; resume automatically preserves the saved policy")
    train.add_argument("--reconstruction-weight",type=float)
    sample=sub.add_parser("generate");sample.add_argument("--checkpoint",required=True)
    sample.add_argument("--prompt",default="ROMEO:\n");sample.add_argument("--length",type=int,default=512)
    sample.add_argument("--temperature",type=float,default=.8);sample.add_argument("--seed",type=int,default=7)
    inspect=sub.add_parser("inspect");inspect.add_argument("--checkpoint",required=True)
    args=parser.parse_args()
    if args.command=="train":
        if args.resume and any(value is not None for value in (args.router,args.routing_arm,args.semantic_laces,args.balance_policy,args.language_credit,args.recall_credit)):
            parser.error("resume preserves saved routing; routing options apply only to fresh models")
        m=Model.load(args.resume) if args.resume else Model(levels=args.levels,parameters=args.parameters,
                         seed=args.seed,router=args.router,associations=args.associations,memory_profile=args.memory_profile,
                         routing_arm=args.routing_arm,semantic_laces=args.semantic_laces,balance_policy=args.balance_policy,
                         language_credit=args.language_credit,recall_credit=args.recall_credit)
        with m:
            m.configure(learning_rate=args.learning_rate,slow_rate=args.slow_rate,query_rate=args.query_rate,
                        head_rate=args.head_rate,reconstruction_weight=args.reconstruction_weight)
            raw=Path(args.data).read_bytes()
            n=min(len(raw)-1,args.targets) if args.targets is not None else len(raw)-1
            if n<=0:parser.error("training requires at least two bytes and a positive target count")
            t=time.perf_counter();loss=m.run(raw[:n],raw[1:n+1]);elapsed=time.perf_counter()-t
            # Save partial blocks as-is for exact continuation. No implicit drain.
            m.save(args.out)
            print(json.dumps(dict(nll=loss,seconds=elapsed,bytes_per_second=n/elapsed,**m.stats()),indent=2))
    else:
        with Model.load(args.checkpoint) as m:
            if args.command=="inspect":print(json.dumps(m.stats(),indent=2))
            else:print(m.generate(args.prompt,length=args.length,temperature=args.temperature,seed=args.seed).decode("utf-8","replace"))


if __name__=="__main__":main()

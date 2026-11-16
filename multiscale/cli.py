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
    train.add_argument("--loop-steps",type=int,nargs='+',help="extra passes per scale; four-level preset: 0 1 2 2")
    train.add_argument("--loop-strength",type=float,help="anchored refinement strength in (0,1], default 0.25")
    train.add_argument("--loop-tolerance",type=float,help="relative convergence tolerance; 0 fixes depth, default 0.01")
    train.add_argument("--route-lr",action=argparse.BooleanOptionalAction,default=None,
                       help="enable bounded route-local LR cycles; explicit changes retire pending credit on resume")
    train.add_argument("--lr-lower",type=float);train.add_argument("--lr-upper",type=float)
    train.add_argument("--lr-anchor",type=float,help="shared multiplier; may also move a resumed controller")
    train.add_argument("--lr-amplitude",type=float);train.add_argument("--lr-period",type=int)
    train.add_argument("--lr-anchor-target",type=float,help="ramp every rate toward this bounded multiplier")
    train.add_argument("--lr-anchor-targets",type=int,default=0,help="future training targets for the anchor ramp")
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
    sample.add_argument("--loop-steps",type=int,nargs='+',help="override extra passes on a frozen inference copy")
    sample.add_argument("--loop-strength",type=float)
    sample.add_argument("--loop-tolerance",type=float)
    inspect=sub.add_parser("inspect");inspect.add_argument("--checkpoint",required=True)
    args=parser.parse_args()
    if args.command in ("train","generate"):
        if args.loop_steps is None and (args.loop_strength is not None or args.loop_tolerance is not None):
            parser.error("--loop-strength and --loop-tolerance require --loop-steps")
    if args.command=="train":
        if any(v is not None for v in (args.lr_lower,args.lr_upper,args.lr_amplitude,args.lr_period)) and args.route_lr is None:
            parser.error("cycle bounds, amplitude and period require --route-lr or --no-route-lr")
        if args.lr_anchor_targets and args.lr_anchor_target is None:
            parser.error("--lr-anchor-targets requires --lr-anchor-target")
        if args.resume and any(value is not None for value in (args.router,args.routing_arm,args.semantic_laces,args.balance_policy,args.language_credit,args.recall_credit)):
            parser.error("resume preserves saved routing; routing options apply only to fresh models")
        m=Model.load(args.resume) if args.resume else Model(levels=args.levels,parameters=args.parameters,
                         seed=args.seed,router=args.router,associations=args.associations,memory_profile=args.memory_profile,
                         routing_arm=args.routing_arm,semantic_laces=args.semantic_laces,balance_policy=args.balance_policy,
                         language_credit=args.language_credit,recall_credit=args.recall_credit)
        with m:
            if args.loop_steps is not None:
                if args.resume:m.finish()
                policy=m.loop_stats()
                m.configure_loops(steps=args.loop_steps,
                                  strength=policy['strength'] if args.loop_strength is None else args.loop_strength,
                                  tolerance=policy['tolerance'] if args.loop_tolerance is None else args.loop_tolerance)
            # An explicit controller change retires credit under the saved
            # policy before applying either new base rates or cycle settings.
            if args.route_lr is not None and args.resume:m.finish()
            m.configure(learning_rate=args.learning_rate,slow_rate=args.slow_rate,query_rate=args.query_rate,
                        head_rate=args.head_rate,reconstruction_weight=args.reconstruction_weight)
            if args.route_lr is not None:
                policy=m.route_lr_stats()
                options={name:value if value is not None else policy[name] for name,value in
                         (("lower",args.lr_lower),("upper",args.lr_upper),("anchor",args.lr_anchor),
                          ("amplitude",args.lr_amplitude),("period",args.lr_period))}
                m.configure_route_lr(enabled=args.route_lr,**options)
            elif args.lr_anchor is not None:
                m.set_learning_anchor(args.lr_anchor)
            if args.lr_anchor_target is not None:
                m.set_learning_anchor(args.lr_anchor_target,over_targets=args.lr_anchor_targets)
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
            elif args.loop_steps is None:
                print(m.generate(args.prompt,length=args.length,temperature=args.temperature,seed=args.seed).decode("utf-8","replace"))
            else:
                with m.inference_copy() as frozen:
                    policy=frozen.loop_stats()
                    frozen.configure_loops(steps=args.loop_steps,
                                           strength=policy['strength'] if args.loop_strength is None else args.loop_strength,
                                           tolerance=policy['tolerance'] if args.loop_tolerance is None else args.loop_tolerance)
                    print(frozen.generate(args.prompt,length=args.length,temperature=args.temperature,seed=args.seed).decode("utf-8","replace"))


if __name__=="__main__":main()

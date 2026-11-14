"""External production caller/worker sampling for Action 6 (never timed evidence)."""
import profile_current_baseline as sampler
from measure_bank_epochs import make_model

def epoch_model(args):
    args.variant=args.profile
    return make_model(args)

sampler.make_model=epoch_model
sampler.PROFILES=('event','e4l1','e4l1-w2','e16l1-w2')
sampler.__file__=__file__
if __name__=='__main__':sampler.main()

"""External production sampling for lightweight frames plus selected Action 4.

Reuses the Action-1 signal sampler, including its warmup and diagnostic-only
LR .04 / local LR .04/3. These samples are not timing/quality evidence.
"""
import profile_current_baseline as sampler

_make_model=sampler.make_model
def make_model(args):
    model=_make_model(args)
    model.configure_frames('lightweight-frames-v1')
    model.configure_prediction(dict(compact=True,rank=16,slow_stride=8))
    return model

sampler.make_model=make_model
sampler.PROFILES=('event','compact_serial','compact_threaded')
# The sampler's subprocess must enter this same wrapper to select the policy.
sampler.__file__=__file__
if __name__=='__main__':sampler.main()

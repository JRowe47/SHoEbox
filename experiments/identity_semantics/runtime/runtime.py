"""Entry point for the selected CPU runtime and its frozen initialization layout."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'hierarchy_lab/throughput_probe'))
import model as engine
from model import Model, core, evaluate
from pipeline import SnapshotPipeline
from reference_policy import REFERENCE_KERNEL, REFERENCE_PROFILE, settings_for


def fresh_model(seed=1, kernel=REFERENCE_KERNEL, *, profile=None, event=None):
    """Build the full current reference design with fresh parameters.

    Set SHOEBOX_EXPERTS=12 or 64 before importing this module. Fixed arrays
    include the original calibrated hippocampal address center; they are not
    the trained model parameters or a continuation checkpoint.

    profile='event' selects stage-5 MinKron context and local event credit.
    The event mapping can override heads, frontier size and replay/credit scales.
    Explicit historical kernels use their recorded policy. The historical
    profile on completion retains the pre-reference training options, but
    still uses the corrected primary clock. Saved checkpoints keep their policy.
    """
    size = '1m' if core.E == 12 else '5m'
    if profile == 'event':
        if kernel not in ('completion','completion_checks'):raise ValueError('Event profile requires completion')
        event={**engine.EVENT_DEFAULT,**(event or {})}
    settings = settings_for(core.E, kernel, 'reference' if profile == 'event' else profile)
    params, _ = core.initialize(seed)
    with np.load(ROOT / 'configs' / (size + '_fixed.npz'), allow_pickle=False) as arrays:
        fixed = {key: arrays[key].copy() for key in arrays.files}
    return Model(params=params, fixed=fixed, kernel=kernel, event=event,
                 local_credit=False if profile == 'historical' else None,
                 content_experts=False if profile == 'historical' else None,
                 pattern_classes=False if profile == 'historical' else None,
                 paired_memory=False if profile == 'historical' else None,
                 reverse=dict(policy='root') if profile == 'historical' else None, **settings)

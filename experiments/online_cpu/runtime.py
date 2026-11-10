"""Entry point for the selected CPU runtime and its frozen initialization layout."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'hierarchy_lab/throughput_probe'))
import model as engine
from model import Model, core, evaluate
from pipeline import SnapshotPipeline


def fresh_model(seed=1, kernel='native'):
    """Use the selected policy, original fixed transports and fresh parameters.

    Set SHOEBOX_EXPERTS=12 or 64 before importing this module. Fixed arrays
    include the original calibrated hippocampal address center; they are not
    the trained model parameters or a continuation checkpoint.
    """
    if core.E not in (12, 64):
        raise ValueError('The published presets require SHOEBOX_EXPERTS=12 or 64')
    size = '1m' if core.E == 12 else '5m'
    settings = json.loads((ROOT / 'configs' / (size + '.json')).read_text())
    params, _ = core.initialize(seed)
    with np.load(ROOT / 'configs' / (size + '_fixed.npz'), allow_pickle=False) as arrays:
        fixed = {key: arrays[key].copy() for key in arrays.files}
    return Model(params=params, fixed=fixed, kernel=kernel, **settings)

"""Default development policy; historical kernels keep their recorded settings."""
import json
from pathlib import Path

REFERENCE_KERNEL = 'completion'
REFERENCE_PROFILE = 'reference-v1'
ROOT = Path(__file__).resolve().parent


def settings_for(experts, kernel=REFERENCE_KERNEL, profile=None):
    if profile is None:
        profile = 'reference' if kernel in ('completion', 'completion_checks') else 'historical'
    if profile not in ('reference', 'historical'):
        raise ValueError('profile must be reference or historical')
    if profile == 'reference' and kernel not in ('completion', 'completion_checks'):
        raise ValueError('The reference profile requires the completion kernel')
    size = '1m' if experts == 12 else '5m'
    settings = json.loads((ROOT / 'configs' / (size + '.json')).read_text())
    if profile == 'reference':
        settings['config']['key_mix'] = 1
        settings['lace'].update(dfa_gain=1, deep_stride=1)
        settings['state_config'].update(credit_stride=1, temporal_features=64)
        settings['utility'].update(residual=1, innovation=1)
    return settings

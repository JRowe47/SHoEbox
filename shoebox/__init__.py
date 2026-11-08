"""SHoEbox's cortical model; historical experiments live under experiments/."""
from .config import SkeletonConfig, cortical_10m, cortical_wide, cortical_cpu
from .model import CorticalModel

__all__ = ['CorticalModel', 'SkeletonConfig', 'cortical_10m', 'cortical_wide', 'cortical_cpu']

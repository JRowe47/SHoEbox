"""Regenerate optimized candidates from the frozen baseline, without retraining."""
from pathlib import Path
import shutil,subprocess
ROOT=Path(__file__).resolve().parent
for name in ['build_masks.py','build_router.py','build_cached.py','build_threaded.py','build_profile.py']:
 subprocess.run(['python',str(ROOT/name)],check=True)
subprocess.run(['python',str(ROOT/'build_profile.py'),'--source','cached'],check=True)
shutil.copyfile(ROOT/'cached.cpp',ROOT/'native.cpp')

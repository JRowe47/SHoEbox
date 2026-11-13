"""Exercise every diagnostic selector and the memory transfer under sanitizers."""
from pathlib import Path
import os
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import core

(ROOT/'results').mkdir(exist_ok=True)
p,f=core.initialize(41)
arrays=[core.flat_parameters(p)]+[f[k] for k in ['embedding','feedback','slots','starts','perm','signs','fperm','fsign','cadence']]
fixture=ROOT/'results/fixture.bin'
with fixture.open('wb') as output:
    for a in arrays:output.write(struct.pack('<q',a.size));output.write(a.tobytes())
binary=ROOT/'results/sanitize_contributions'
subprocess.run(['g++','-O1','-g','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}',
                '-fsanitize=address,undefined','-fno-omit-frame-pointer',
                str(ROOT/'tests/sanitize_contributions.cpp'),'-o',str(binary)],check=True)
subprocess.run([str(binary),str(fixture)],env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'},check=True)

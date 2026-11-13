from pathlib import Path
import os,struct,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import core
p,f=core.initialize(41)
arrays=[core.flat_parameters(p)]+[f[k] for k in ['embedding','feedback','slots','starts','perm','signs','fperm','fsign','cadence']]
with (ROOT/'results/fixture.bin').open('wb') as output:
    for a in arrays:output.write(struct.pack('<q',a.size));output.write(a.tobytes())
command=['g++','-O1','-g','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(ROOT/'tests/sanitize.cpp'),'-o',str(ROOT/'results/sanitize')]
subprocess.run(command,check=True)
subprocess.run([str(ROOT/'results/sanitize'),str(ROOT/'results/fixture.bin')],env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'},check=True)

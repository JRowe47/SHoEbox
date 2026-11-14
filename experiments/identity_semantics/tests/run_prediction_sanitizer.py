from pathlib import Path
import os,struct,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import core
(ROOT/'results').mkdir(exist_ok=True);p,f=core.initialize(41)
arrays=[core.flat_parameters(p)]+[f[k] for k in ('embedding','feedback','slots','starts','perm','signs','fperm','fsign','cadence')]
fixture=ROOT/'results/prediction_fixture.bin'
with fixture.open('wb') as out:
    for a in arrays:out.write(struct.pack('<q',a.size));out.write(a.tobytes())
exe=ROOT/'results/sanitize_prediction'
subprocess.run(['g++','-O1','-g1','-std=c++17','-pthread','-march=native',f'-DSHOEBOX_EXPERTS={core.E}',
                '-fsanitize=address,undefined','-fno-omit-frame-pointer',str(ROOT/'tests/sanitize_prediction.cpp'),'-o',str(exe)],check=True)
subprocess.run([str(exe),str(fixture)],env={**os.environ,'ASAN_OPTIONS':os.environ.get('ASAN_OPTIONS','detect_leaks=1')},check=True)

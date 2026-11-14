"""Focused sanitizer coverage for the parallel and pending-credit kernels."""
from pathlib import Path
import os
import struct
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import core
with tempfile.TemporaryDirectory(prefix='shoebox-opt-sanitize-') as temporary:
    root=Path(temporary);fixture=root/'fixture.bin';binary=root/'sanitize'
    params,fixed=core.initialize(41)
    with fixture.open('wb') as stream:
        for values in [core.flat_parameters(params)]+[fixed[k] for k in ('embedding','feedback','slots','starts','perm','signs','fperm','fsign','cadence')]:
            stream.write(struct.pack('<q',values.size));stream.write(values.tobytes())
    subprocess.run(['g++','-DSHOEBOX_INLINE_DIAGNOSTICS=1','-O1','-g','-march=native','-std=c++17','-pthread',
                    f'-DSHOEBOX_EXPERTS={core.E}','-fsanitize=address,undefined','-fno-omit-frame-pointer',
                    str(ROOT/'tests/sanitize_optimizations.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(fixture)],env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'},check=True)

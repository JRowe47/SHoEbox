"""Run finite-difference and continuation checks, optionally under ASan/UBSan."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from .adapter import source,HERE,ROOT
p=argparse.ArgumentParser();p.add_argument('--sanitize',action='store_true');args=p.parse_args()
with tempfile.TemporaryDirectory(prefix='shoebox-state-check-') as tmp:
    cpp=Path(tmp)/'test.cpp';binary=Path(tmp)/'test';cpp.write_text(source()+'\n'+(HERE/'native_test.cpp').read_text())
    flags=['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined'] if args.sanitize else ['-O3','-march=native']
    subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(binary)],check=True)
    env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
    subprocess.run([str(binary)],check=True,env=env)

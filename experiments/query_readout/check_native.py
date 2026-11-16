"""Finite differences, native parity, continuation, ASan and UBSan."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from .adapter import source,HERE,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--no-leak-check',action='store_true');a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='query-tests-') as tmp:
        cpp=Path(tmp)/'test.cpp';exe=Path(tmp)/'test'
        cpp.write_text(source()+'\n'+(HERE/'native_test.cpp').read_text())
        for flags in (['-O2'],['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined']):
            cmd=[os.environ.get('CXX','g++'),'-std=c++17','-ffp-contract=off',*flags,
                 '-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(exe)]
            subprocess.run(cmd,check=True)
            subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':f'detect_leaks={int(not a.no_leak_check)}:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})


if __name__=='__main__':main()

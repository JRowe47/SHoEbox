"""Native numerical checks, Python lifecycle checks and ASan/UBSan."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]


def run(cmd, **kwargs):
    print(" ".join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True,**kwargs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--no-leak-check",action="store_true",help="explicitly disable LSan on ptrace-managed hosts")
    args=parser.parse_args()
    compiler=os.environ.get("CXX","g++")
    with tempfile.TemporaryDirectory(prefix="shoe-multiscale-test-") as tmp:
        executable=Path(tmp)/"checks"
        source=[ROOT/"multiscale/native.cpp",ROOT/"multiscale/tests/native_tests.cpp"]
        run([compiler,"-std=c++17","-O2","-ffp-contract=off",*source,"-o",executable])
        run([executable])
        run([sys.executable,"-m","unittest","discover","-s","multiscale/tests","-v"])
        run([compiler,"-std=c++17","-O1","-g","-fno-omit-frame-pointer","-fsanitize=address,undefined",
             "-ffp-contract=off",*source,"-o",executable])
        if args.no_leak_check:print("LeakSanitizer disabled explicitly; ASan and UBSan remain enabled.",flush=True)
        run([executable],env={**os.environ,"ASAN_OPTIONS":f"detect_leaks={int(not args.no_leak_check)}:halt_on_error=1","UBSAN_OPTIONS":"halt_on_error=1:print_stacktrace=1"})


if __name__=="__main__":main()

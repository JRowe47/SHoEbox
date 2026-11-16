"""Exercise the MAD native bridge, optionally with ASan and UBSan."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sanitize", action="store_true")
    args = p.parse_args()
    flags = ["-std=c++17", "-ffp-contract=off"]
    flags += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined"] if args.sanitize else ["-O2"]
    with tempfile.TemporaryDirectory(prefix="shoe-mad-check-") as d:
        exe = Path(d)/"checks"
        subprocess.run([os.environ.get("CXX", "g++"), *flags, str(Path(__file__).with_name("native_tests.cpp")), "-o", str(exe)], check=True)
        env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1", "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

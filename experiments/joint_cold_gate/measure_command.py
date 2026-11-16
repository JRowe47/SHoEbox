"""Measure a whole campaign, including child CPU, evaluation and compilation."""
import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.cost.exists():
        raise FileExistsError(args.cost)
    start = time.perf_counter()
    result = subprocess.run(command)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    args.cost.write_text(json.dumps(dict(command=command, exit_code=result.returncode,
        wall_seconds=time.perf_counter()-start, user_cpu_seconds=usage.ru_utime,
        system_cpu_seconds=usage.ru_stime, cpu_seconds=usage.ru_utime+usage.ru_stime,
        peak_rss_kib=usage.ru_maxrss,
        scope='Whole child process tree, including fit, evaluation, compilation and I/O.'), indent=2)+'\n')
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()

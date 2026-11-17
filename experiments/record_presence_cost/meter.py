"""Write-once whole-process receipts, including descendants and failed commands."""
import argparse,json,os,resource,subprocess,time
from pathlib import Path
from datetime import datetime,timezone

def main():
 p=argparse.ArgumentParser();p.add_argument('prefix',type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 receipt=Path(str(a.prefix)+'_process.json');log=Path(str(a.prefix)+'.txt')
 if receipt.exists() or log.exists():raise FileExistsError(a.prefix)
 start=datetime.now(timezone.utc).isoformat();wall=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
 with log.open('w') as f:r=subprocess.run(a.command,stdout=f,stderr=subprocess.STDOUT)
 after=resource.getrusage(resource.RUSAGE_CHILDREN);user=after.ru_utime-before.ru_utime;system=after.ru_stime-before.ru_stime
 receipt.write_text(json.dumps(dict(command=a.command,started_utc=start,completed_utc=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,wall_seconds=time.perf_counter()-wall,user_cpu_seconds=user,system_cpu_seconds=system,cpu_seconds=user+system,peak_rss_kib=after.ru_maxrss,scope='Whole child process tree including compilation, fitting, evaluation, failed checks and I/O.'),indent=2)+'\n')
 print(a.prefix.name,'exit',r.returncode,'cpu',round(user+system,2),flush=True)
 raise SystemExit(r.returncode)
if __name__=='__main__':main()

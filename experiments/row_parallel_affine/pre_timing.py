"""Passing committed controls are a mandatory boundary before the timing run."""
import argparse,hashlib,json,platform,re,subprocess
from pathlib import Path
from . import backend
from experiments.selective_record_key_credit.run import sources

REQUIRED=('python_scalar','python_candidate','native_scalar','native_candidate','runtime',
          'public_native','history','research','policy','workload','checkpoints','counts')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('work',type=Path);a=p.parse_args()
    assert not (a.work/'pre_timing.lock.json').exists()
    receipts={}
    for name in REQUIRED:
        path=a.work/(name+'_process.json');r=json.loads(path.read_text());assert r['exit_code']==0,name
        receipts[name]=r
    for name in ('python_scalar','python_candidate'):
        r=json.loads((a.work/(name+'.json')).read_text());assert r['passed'] and r['tests']==20
    assert json.loads((a.work/'research.json').read_text())['tests']==209
    assert (a.work/'policy.txt').read_text().count('23 cross-policy rejection checks passed')==2
    assert json.loads((a.work/'history.json').read_text())['exact']
    assert json.loads((a.work/'checkpoints/result.json').read_text())['count']==18
    assert json.loads((a.work/'counts.json').read_text())['passed']
    assert json.loads((a.work/'workload/lock.json').read_text())['passed']
    required=('581 E8 native checks passed','110 E8 request checks passed','83 A1 replacement/lifecycle checks passed',
              '13089 A4 selection checks passed','75295 A5 objective checks passed','7826 A6 presence checks passed','57280 A9 masked credit checks passed')
    for name in ('native_scalar','native_candidate'):
        log=(a.work/(name+'.txt')).read_text()
        for phrase in required:assert log.count(phrase)==2,(name,phrase)
    assert (a.work/'native_candidate.txt').read_text().count('A10 bitwise comparisons 71680')==2
    for name,digest in json.loads((a.work/'checkpoints/frozen_before.json').read_text()).items():
        assert sha(a.prior/'a9'/name)==digest,name
    # Exact checked transform, with the immutable scalar policy recovered.
    head=backend.header()
    restored=head.removeprefix('#include <immintrin.h>\n').replace((backend.HERE/'affine.hpp').read_text()+'\n','',1)
    for old,new in backend.CHANGES.items():assert restored.count(new)==1;restored=restored.replace(new,old)
    assert restored==backend.scalar.header('a9')
    assert backend.source()==backend.scalar.source('a9')
    lib=backend.library();binary=Path(lib._name)
    disassembly=subprocess.check_output(['objdump','-d','-C',str(binary)],text=True)
    assert not re.search(r'\bv(?:f[mn](?:add|sub)|fmaddsub|fmsubadd)\w*\b',disassembly)
    (a.work/'candidate_disassembly.txt').write_text(disassembly)
    cpu_probe=subprocess.run(['lscpu','-J'],capture_output=True,text=True)
    environment=dict(platform=platform.platform(),compiler=subprocess.check_output(['g++','--version'],text=True),
        flags=backend.FLAGS,libc=platform.libc_ver(),cpu_probe=dict(returncode=cpu_probe.returncode,stdout=cpu_probe.stdout,stderr=cpu_probe.stderr),
        cpuinfo=Path('/proc/cpuinfo').read_text(),
        linked_libraries=subprocess.check_output(['ldd',str(binary)],text=True),binary_sha256=sha(binary),
        generated_header_sha256=hashlib.sha256(head.encode()).hexdigest(),
        generated_source_sha256=hashlib.sha256(backend.source().encode()).hexdigest(),fma_instructions=0)
    (a.work/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    evidence={str(path.relative_to(a.work)):sha(path) for path in a.work.rglob('*') if path.is_file()
        and (path.suffix in ('.json','.jsonl','.txt','.bin')) and not any(x.endswith('_build') for x in path.parts)
        and path.name not in ('pre_timing.lock.json','pre_timing_retry.txt','pre_timing_retry_process.json')}
    locked=sources();locked['reports/exp0305a9f1/A10_REGISTRATION.md']=sha(backend.ROOT/'reports/exp0305a9f1/A10_REGISTRATION.md')
    result=dict(passed=True,starting_main='8270e0d60bbefd75f7c7c9604d5961702615aac2',
        source_hashes=locked,evidence_hashes=evidence,required_processes=REQUIRED,
        frozen_checkpoints=json.loads((a.work/'checkpoints/frozen_before.json').read_text()),
        exact_source_transform=True,scalar_source_unchanged=True,research_fitting_bytes=0,
        accuracy_rescoring=False,audit_episodes=0,compiler=environment,
        cpu_seconds_before_lock=sum(r['cpu_seconds'] for r in receipts.values())+json.loads((a.work/'pre_timing_process.json').read_text())['cpu_seconds'],
        preserved_failure='pre_timing_process.json: environment inventory lscpu failed; zero model inputs; /proc/cpuinfo fallback added',
        timing_authorization='Only the four registered arms, 3 rotating repeats, 65536+1024 bytes each; no replacements.')
    raw=json.dumps(result,indent=2)+'\n';(a.work/'pre_timing.lock.json').write_text(raw)
    report=backend.ROOT/'reports/exp0305a10';report.mkdir(exist_ok=True)
    (report/'pre_timing.lock.json').write_text(raw)
    (report/'timing_worlds.lock.json').write_bytes((a.work/'workload/lock.json').read_bytes())
    evidence_dir=report/'controls';evidence_dir.mkdir(exist_ok=True)
    for name in REQUIRED:
        for suffix in ('_process.json','.txt'):
            (evidence_dir/(name+suffix)).write_bytes((a.work/(name+suffix)).read_bytes())
    for name in ('python_scalar.json','python_candidate.json','history.json','research.json','counts.json','environment.json'):
        (evidence_dir/name).write_bytes((a.work/name).read_bytes())
    (evidence_dir/'checkpoints.json').write_bytes((a.work/'checkpoints/result.json').read_bytes())
    for name in ('pre_timing.txt','pre_timing_process.json'):
        (evidence_dir/name).write_bytes((a.work/name).read_bytes())
    print('All exactness/source/checkpoint/workload controls pass. Commit this lock before timing.')
if __name__=='__main__':main()

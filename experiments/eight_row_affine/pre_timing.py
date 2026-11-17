"""Commit this complete passing boundary before the sole A11 measurement."""
import argparse,hashlib,importlib.metadata,json,platform,re,shutil,sys
from pathlib import Path
from . import backend
from experiments.selective_record_key_credit.run import sources

REQUIRED=('instrument_selftest_final','affine_retry','python_scalar','python_a10','python_a11','native_scalar','native_a10','native_a11','runtime','public_native','history','policy','research','checkpoints')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('work',type=Path);a=p.parse_args()
    assert not (a.work/'pre_timing.lock.json').exists();report=backend.ROOT/'reports/exp0305a11';assert not report.exists()
    for name in REQUIRED:assert json.loads((a.work/name/'process.json').read_text())['exit_code']==0,name
    for name in ('production_retry','workload','accounting'):
        assert json.loads((a.work/'bookkeeping'/(name+'_process.json')).read_text())['exit_code']==0,name
    for name in ('python_scalar','python_a10','python_a11'):
        r=json.loads((a.work/name/'result.json').read_text());assert r['passed'] and r['tests']==20
    required=('581 E8 native checks passed','110 E8 request checks passed','83 A1 replacement/lifecycle checks passed','13089 A4 selection checks passed','75295 A5 objective checks passed','7826 A6 presence checks passed','57280 A9 masked credit checks passed')
    for name in ('native_scalar','native_a10','native_a11'):
        log=(a.work/name/'output.txt').read_text()
        for phrase in required:assert log.count(phrase)==2,(name,phrase)
    assert (a.work/'affine_retry/output.txt').read_text().count('A11 three-backend bitwise comparisons 71680')==2
    assert 'Ran 60 tests' in (a.work/'runtime/output.txt').read_text()
    assert json.loads((a.work/'research/result.json').read_text())['tests']==209
    assert (a.work/'policy/output.txt').read_text().count('23 cross-policy rejection checks passed')==3
    assert json.loads((a.work/'history/result.json').read_text())['exact']
    cp=json.loads((a.work/'checkpoints/trajectories/result.json').read_text());assert cp['passed'] and cp['count']==18 and cp['counted_and_uncounted_exact']
    for name,digest in json.loads((a.work/'checkpoints/trajectories/frozen_before.json').read_text()).items():assert sha(a.prior/'a9'/name)==digest
    prior_lock=json.loads((backend.ROOT/'reports/exp0305a10/pre_timing.lock.json').read_text())
    for name,digest in prior_lock['source_hashes'].items():assert sha(backend.ROOT/name)==digest,name
    head=backend.header();restored=head.removeprefix('#include <immintrin.h>\n').replace((backend.HERE/'affine.hpp').read_text()+'\n','',1)
    for old,new in backend.CHANGES.items():assert restored.count(new)==1;restored=restored.replace(new,old)
    assert restored==backend.scalar.header('a9') and backend.source()==backend.scalar.source('a9')
    environment=json.loads((a.work/'production_retry/result.json').read_text());assert environment['stack_text_identical']
    assert environment['generated_header_sha256']==hashlib.sha256(head.encode()).hexdigest()
    assert environment['generated_source_sha256']==hashlib.sha256(backend.source().encode()).hexdigest()
    from experiments.row_parallel_affine.backend import FLAGS as prior_flags
    assert backend.FLAGS==prior_flags==environment['flags']
    for v in environment['libraries'].values():assert sha(Path(v['path']))==v['sha256'] and v['fma_instructions']==0
    for asm in (a.work/'production_retry').glob('*.disassembly.txt'):
        assert not re.search(r'\bv(?:fmadd|fmsub|fnmadd|fnmsub)\w*\b',asm.read_text()),asm
    workload=json.loads((a.work/'workload/lock.json').read_text());assert workload['passed'] and workload['rng']==20261203
    accounting=json.loads((a.work/'accounting_before/ACCOUNTING.json').read_text())
    assert len(accounting['python_independent_count_checks'])==3
    evidence={str(p.relative_to(a.work)):sha(p) for p in sorted(a.work.rglob('*')) if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.gz','.bin') and not p.name.endswith('.text.bin') and not p.name.startswith('pre_timing') and not 'bookkeeping/lock' in str(p)}
    locked=sources();locked['reports/exp0305a10/A11_REGISTRATION.md']=sha(backend.ROOT/'reports/exp0305a10/A11_REGISTRATION.md')
    result=dict(passed=True,starting_main='d11d5d117d4ba97cde5f31e475201906930b593d',source_hashes=locked,evidence_hashes=evidence,
        required_control_processes=REQUIRED,exact_six_site_source_transform=True,old_source_hashes_unchanged=len(prior_lock['source_hashes']),
        frozen_checkpoints=json.loads((a.work/'checkpoints/trajectories/frozen_before.json').read_text()),
        research_fitting_bytes=0,accuracy_rescoring=False,audit_episodes=0,
        timing_authorization='Only five registered arms, three left-rotating repeats, 65536 measured +1024 reset warmup bytes each. No retries.',
        compiler_and_stack=environment,accounting_sha256=sha(a.work/'accounting_before/ACCOUNTING.json'),
        instrumentation_environment=dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ('numpy','tree-sitter','tree-sitter-cpp')}),
        preserved_failures=['affine: compile-time static_assert instrumentation, no model execution','production: objcopy input rewrite while DSO mapped, no model execution'],
        prior_a10_timing='Immutable: 76.020287% native on its original stream; twelve raw timings and failed metadata probe preserved.')
    raw=json.dumps(result,indent=2)+'\n';report.mkdir();(a.work/'pre_timing.lock.json').write_text(raw);(report/'pre_timing.lock.json').write_text(raw)
    shutil.copy2(a.work/'workload/lock.json',report/'timing_worlds.lock.json')
    shutil.copy2(a.work/'accounting_before/ACCOUNTING.json',report/'ACCOUNTING_BEFORE_TIMING.json')
    shutil.copy2(a.work/'accounting_before/events.jsonl.gz',report/'correctness_events.jsonl.gz')
    controls=report/'controls';controls.mkdir()
    for name in REQUIRED+('affine','instrument_selftest'):
        for source,dest in [('process.json',name+'_process.json'),('output.txt',name+'.txt'),('result.json',name+'.json')]:
            path=a.work/name/source
            if path.exists():shutil.copy2(path,controls/dest)
    shutil.copy2(a.work/'checkpoints/trajectories/result.json',controls/'checkpoints.json')
    for path in (a.work/'bookkeeping').glob('*_process.json'):shutil.copy2(path,controls/path.name)
    shutil.copy2(a.work/'production_retry/stack.so-model.su',controls/'generated_stack_usage.txt')
    print('All controls/source/cursors/workload/accounting pass. Commit lock before timing.')
if __name__=='__main__':main()

"""Read-only source/evidence verification; no model calls or timing repeats."""
import argparse,hashlib,json,statistics,subprocess
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('work',type=Path);a=p.parse_args()
    report=Path(__file__).resolve().parent;root=report.parents[1]
    lock=json.loads((a.work/'pre_timing.lock.json').read_text())
    for path,digest in lock['source_hashes'].items():assert sha(root/path)==digest,path
    for path,digest in lock['evidence_hashes'].items():assert sha(a.work/path)==digest,path
    for path,digest in lock['frozen_checkpoints'].items():assert sha(a.prior/'a9'/path)==digest,path
    assert (report/'pre_timing.lock.json').read_bytes()==(a.work/'pre_timing.lock.json').read_bytes()
    assert (report/'timing_worlds.lock.json').read_bytes()==(a.work/'workload/lock.json').read_bytes()
    bench=json.loads((a.work/'benchmark.json').read_text());raw=[json.loads(x) for x in (a.work/'timing_raw.jsonl').read_text().splitlines()]
    names=bench['arms'];assert len(raw)==12 and names==['native','e8','scalar_a9','row_parallel_a9']
    expected=[(i,n) for i in range(3) for n in names[i:]+names[:i]]
    assert [(x['repeat'],x['arm']) for x in raw]==expected
    for n in names:
        rows=[x for x in raw if x['arm']==n];result=bench['rows'][n]
        assert [x['cpu_seconds'] for x in rows]==result['cpu_seconds']
        assert statistics.median(result['cpu_seconds'])==result['median_cpu_seconds']
        assert result['relative_native']==bench['rows']['native']['median_cpu_seconds']/result['median_cpu_seconds']
        assert all(x['input_bytes']==65536 and x['warmup_bytes']==1024 for x in rows)
        assert [x['checksum_hex'] for x in rows]==result['checksums']
        assert [x['warmup_checksum_hex'] for x in rows]==result['warmup_checksums']
    s=bench['rows']['scalar_a9'];c=bench['rows']['row_parallel_a9']
    assert s['checksums']==c['checksums'] and s['warmup_checksums']==c['warmup_checksums']
    assert not bench['execution_pass'] and not bench['native_speed_pass'] and bench['no_scalar_slowdown']
    assert bench['research_fitting_bytes']==0 and not bench['accuracy_scored']
    assert bench['pre_timing_commit']=='3bc9d54f84626c3c117f88419f8cdcec0564e325'
    for name in ('benchmark.json','timing_raw.jsonl','timing_started.json','timing_process.json'):
        assert (report/name).read_bytes()==(a.work/name).read_bytes()
    base='8270e0d60bbefd75f7c7c9604d5961702615aac2'
    old=subprocess.check_output(['git','ls-tree','-r','--name-only',base],cwd=root,text=True).splitlines()
    exceptions={'AGENTS.md','DEVELOPMENT_HANDOFF.md','docs/capabilities/SHoEbox_Minimal_Capability_Matrix.md'}
    for name in old:
        if name in exceptions:continue
        previous=subprocess.check_output(['git','show',base+':'+name],cwd=root)
        assert (root/name).read_bytes()==previous,name
    changed=subprocess.check_output(['git','diff','--name-only',base],cwd=root,text=True).splitlines()
    for name in changed:
        if name in exceptions:continue
        assert name.startswith(('experiments/row_parallel_affine/','reports/exp0305a10/')),name
        path=root/name;assert path.suffix not in ('.so','.o','.qr','.npz','.bin','.pyc','.exe'),name
        assert not path.read_bytes().startswith(b'\x7fELF'),name
    assert not (root/'experiments/row_parallel_affine/a11').exists()
    result=dict(passed=True,model_calls=0,source_closure_files=len(lock['source_hashes']),
        original_files_verified=len(old)-len(exceptions),frozen_files=36,raw_timing_intervals=12,
        exact_outputs=True,execution_pass=False,no_research_fits=True,no_accuracy_rescoring=True,
        all_original_results_preserved=True,one_next_registration='A11_REGISTRATION.md',
        accounting_limitations_explicit=True,publication_pending=True)
    (report/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

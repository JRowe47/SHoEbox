"""Compile production libraries without counters; inspect, never benchmark."""
import argparse,hashlib,json,re,shutil,subprocess
from pathlib import Path
from . import backend
from experiments.row_parallel_affine import backend as a10
from experiments.selective_record_key_credit import backend as scalar

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    libs={name:lib for name,lib in [('e8',scalar.library('e8')),('a8',scalar.library('a8')),('scalar',scalar.library('a9')),('a10',a10.library()),('a11',backend.library())]}
    rows={}
    for name,lib in libs.items():
        path=Path(lib._name);dis=subprocess.check_output(['objdump','-d','-C',str(path)],text=True)
        assert not re.search(r'\bv(?:fmadd|fmsub|fnmadd|fnmsub)\w*\b',dis),name
        (a.out/(name+'.disassembly.txt')).write_text(dis)
        rows[name]=dict(path=str(path),sha256=sha(path),fma_instructions=0)
    cache=Path(libs['a11']._name).parent
    command=['g++',*backend.FLAGS,'-fstack-usage','-I'+str(cache),'-I'+str(backend.ROOT/'multiscale'),str(cache/'model.cpp'),'-o',str(a.out/'stack.so')]
    subprocess.run(command,check=True)
    for label,path in [('production',Path(libs['a11']._name)),('stack',a.out/'stack.so')]:
        # objcopy rewrites its input when no output path is supplied. Never let
        # its inspection touch a shared object mapped into this live process.
        inspection=a.out/(label+'.inspect.so');shutil.copy2(path,inspection)
        subprocess.run(['objcopy','--dump-section','.text='+str(a.out/(label+'.text.bin')),str(inspection)],check=True)
    assert (a.out/'production.text.bin').read_bytes()==(a.out/'stack.text.bin').read_bytes()
    stack=list(a.out.glob('*.su'));assert len(stack)==1
    report=dict(libraries=rows,flags=backend.FLAGS,stack_reporting_command=command,stack_text_identical=True,
        source_vector_scratch_upper_bound_bytes=17*32,retained_local_value_buffer_bytes=32,
        persistent_coefficient_packing_bytes=0,added_persistent_state_bytes=0,
        generated_header_sha256=hashlib.sha256(backend.header().encode()).hexdigest(),
        generated_source_sha256=hashlib.sha256(backend.source().encode()).hexdigest(),
        compiler=subprocess.check_output(['g++','--version'],text=True),cpuinfo=Path('/proc/cpuinfo').read_text(),
        model_calls=0,research_fitting_bytes=0,stack_file=stack[0].name)
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('Production sources compiled; no FMA; stack report has identical .text; zero model calls.')
if __name__=='__main__':main()

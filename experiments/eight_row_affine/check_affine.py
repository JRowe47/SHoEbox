"""Fixed original seed/trials against independent scalar, A9, A10 and A11."""
import os,subprocess
from pathlib import Path
from . import backend
from experiments.row_parallel_affine import backend as a10
from experiments.row_parallel_affine.check_native import AFFINE

def main():
    work=Path(os.environ['A11_INSTRUMENT_ROOT']).parent/'affine';work.mkdir()
    head=backend.header();anchor='// Reconstruction targets only.'
    head=head.replace(anchor,(a10.HERE/'affine.hpp').read_text()+'\n'+anchor)
    (work/'readout.hpp').write_text(head)
    body=AFFINE.replace('a10_affine<R>(rows,x,out.data());','a11_affine<R>(rows,x,out.data());std::array<float,R> four;a10_affine<R>(rows,x,four.data());')
    body=body.replace('std::memcmp(&legacy,&out[r],4))','std::memcmp(&legacy,&out[r],4)||std::memcmp(&four[r],&out[r],4))')
    body=body.replace('A10 bitwise','A11 three-backend bitwise')
    body=body.replace('float y=std::tanh(out[r]),z=std::tanh(expected);','float y=std::tanh(out[r]),z=std::tanh(expected),t=std::tanh(four[r]),v=std::tanh(legacy);')
    body=body.replace('std::memcmp(&y,&z,4))','std::memcmp(&y,&z,4)||std::memcmp(&y,&t,4)||std::memcmp(&y,&v,4))')
    cpp=work/'affine.cpp';cpp.write_text(backend.source()+body)
    for tag,flags in [('optimized',['-O3','-march=native']),('sanitized',['-O1','-mavx2','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined'])]:
        exe=work/tag
        subprocess.run(['g++','-std=c++17','-ffp-contract=off',*flags,'-I'+str(work),'-I'+str(backend.ROOT/'multiscale'),str(cpp),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
if __name__=='__main__':main()

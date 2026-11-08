"""Optional CPU MinKron fusion; the portable optimized path remains available.

Builds locally once with the installed C++ compiler; no binary downloads.
SHOEBOX_FUSED=0 disables native execution. First-order reverse-mode autograd
is supported; higher-order derivatives use the portable backend explicitly.
"""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import torch
from torch.autograd.function import once_differentiable


_status = {'available':False,'reason':'not requested'}


@lru_cache(None)
def load_native():
    global _status
    if sys.platform != 'linux':
        _status={'available':False,'reason':'native build currently supports Linux'}
        return None
    compiler=shutil.which(os.environ.get('CXX','c++'))
    if compiler is None:
        _status={'available':False,'reason':'no C++ compiler'}
        return None
    try:
        from torch.utils.cpp_extension import include_paths,library_paths,get_default_build_root
        import fcntl
        source=Path(__file__).with_name('minkron_cpu.cpp')
        cpu=''
        if Path('/proc/cpuinfo').exists():
            cpu=next((line for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith(('flags','Features'))),'')
        flags=['-O3','-std=c++20','-shared','-fPIC','-ffp-contract=off']
        if platform.machine() in ('x86_64','AMD64'):flags+=['-march=native']
        # ATen's header-only parallel_for is serial unless the extension itself
        # enables OpenMP, even when the installed PyTorch build uses it.
        if 'ATen parallel backend: OpenMP' in torch.__config__.parallel_info():
            flags+=['-fopenmp']
        identity=(str(torch.__version__),compiler,torch._C._GLIBCXX_USE_CXX11_ABI,cpu,flags)
        fingerprint=hashlib.sha256(source.read_bytes()+Path(__file__).read_bytes()+repr(identity).encode()).hexdigest()[:16]
        namespace='shoebox_minkron_'+fingerprint
        folder=Path(get_default_build_root())/namespace
        folder.mkdir(parents=True,exist_ok=True)
        library=folder/'minkron_cpu.so'
        with (folder/'build.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not library.exists():
                generated=folder/'kernel.cpp'
                generated.write_text(source.read_text().replace('TORCH_LIBRARY(shoebox_minkron_cpu, m)',
                                                               f'TORCH_LIBRARY({namespace}, m)'))
                temporary=folder/f'minkron_cpu.{os.getpid()}.so'
                command=[compiler,*flags,f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
                         *['-I'+p for p in include_paths()],str(generated),*['-L'+p for p in library_paths()],
                         '-ltorch_cpu','-ltorch','-lc10','-o',str(temporary)]
                try:
                    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
                    (folder/'build.log').write_text(result.stdout+result.stderr)
                    if result.returncode:raise RuntimeError('C++ compilation failed')
                    temporary.replace(library)
                finally:
                    temporary.unlink(missing_ok=True)
        torch.ops.load_library(str(library))
        _status={'available':True,'namespace':namespace,'compiler':compiler,'flags':flags,
                 'isa':getattr(torch.ops,namespace).cpu_isa(),
                 'openmp':'-fopenmp' in flags,
                 'library':str(library),'build_log':str(folder/'build.log')}
        return getattr(torch.ops,namespace)
    except (OSError,RuntimeError,ImportError,subprocess.TimeoutExpired) as exc:
        _status={'available':False,'reason':str(exc)}
        return None


def native_status():
    if os.environ.get('SHOEBOX_FUSED','auto')=='0':return {'available':False,'reason':'disabled by SHOEBOX_FUSED'}
    load_native()
    return dict(_status)


def for_tensor(tensor):
    if os.environ.get('SHOEBOX_FUSED','auto')!='0' and tensor.device.type=='cpu' and tensor.dtype in (torch.float32,torch.float64):
        return load_native()
    return None


class _FinalState(torch.autograd.Function):
    @staticmethod
    def forward(ctx,k,v,g,beta,write,previous,lengths):
        native=load_native()
        final,tape=native.final_forward(k,v,g,beta,write,previous,True,lengths)
        ctx.save_for_backward(k,v,g,beta,write,previous,tape,lengths)
        ctx.native=native
        return final

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_final):
        return (*ctx.native.final_backward(grad_final.contiguous(),*ctx.saved_tensors),None)


def fused_final(k,v,g,beta,write,state,lengths=None):
    """Return a fused final state, or None when the portable path is needed."""
    native=for_tensor(k)
    if native is None:return None
    if state is None:state=k.new_zeros(k.shape[0],k.shape[2],k.shape[3],v.shape[3])
    args=tuple(t.contiguous() for t in (k,v,g,beta,write,state))
    if torch.is_grad_enabled() and any(t.requires_grad for t in args):
        return _FinalState.apply(*args,lengths)
    return native.final_forward(*args,False,lengths)[0]

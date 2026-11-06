"""Optional local CPU extension; portable PyTorch remains the fallback.

SHOEBOX_NATIVE=0 disables compilation. No downloads or extra build packages are
needed. A compiler builds once into PyTorch's extension cache, outside the repo.
"""
from functools import lru_cache
import hashlib
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import torch


@lru_cache(None)
def load_native():
    try:
        return _load_native()
    except (OSError, RuntimeError, ImportError):
        # A missing toolchain, read-only cache or incompatible build must not
        # prevent the portable model from running.
        return None


def _load_native():
    if os.environ.get('SHOEBOX_NATIVE', 'auto') == '0' or sys.platform != 'linux':
        return None
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    if compiler is None:
        return None
    from torch.utils.cpp_extension import include_paths, library_paths, get_default_build_root
    source = Path(__file__).with_name('kron_cpu.cpp')
    cpu_flags = ''
    if Path('/proc/cpuinfo').exists():
        cpu_flags = next((line for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith(('flags', 'Features'))), '')
    # Generated binaries are local to this ISA; none are shipped in the repo.
    machine_flags = ['-march=native'] if platform.machine() in ('x86_64', 'AMD64') else []
    fingerprint = hashlib.sha256(source.read_bytes()+Path(__file__).read_bytes()+str((torch.__version__, compiler,
        torch._C._GLIBCXX_USE_CXX11_ABI, cpu_flags)).encode()).hexdigest()[:16]
    namespace = 'shoebox_kron_'+fingerprint
    folder = Path(get_default_build_root())/('shoebox_kron_'+fingerprint)
    folder.mkdir(parents=True, exist_ok=True)
    library = folder/'kron_cpu.so'
    # Serialize both the build and atomic replacement across concurrent workers.
    import fcntl
    with (folder/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = folder/f'kron_cpu.{os.getpid()}.so'
            generated = folder/'kernels.cpp'
            generated.write_text(source.read_text().replace('TORCH_LIBRARY(shoebox_kron_cpu, m)',
                                                          f'TORCH_LIBRARY({namespace}, m)'))
            command = [compiler, '-O3', '-std=c++20', '-shared', '-fPIC', '-ffp-contract=off',
                       *machine_flags,
                       f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
                       *['-I'+p for p in include_paths()], str(generated),
                       *['-L'+p for p in library_paths()], '-ltorch_cpu', '-ltorch', '-lc10',
                       '-o', str(temporary)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=180)
                (folder/'build.log').write_text(result.stdout+result.stderr)
                if result.returncode:
                    return None
                temporary.replace(library)
            except (OSError, subprocess.TimeoutExpired):
                temporary.unlink(missing_ok=True)
                return None
    try:
        torch.ops.load_library(str(library))
    except (OSError, RuntimeError):
        return None
    return getattr(torch.ops, namespace)


def for_tensor(tensor):
    if tensor.device.type == 'cpu' and tensor.dtype in (torch.float32, torch.float64):
        return load_native()
    return None

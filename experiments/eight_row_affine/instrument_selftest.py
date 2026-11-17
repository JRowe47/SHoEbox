"""Validate counter placement before any model controls, using no model bytes."""
import json,os,struct,subprocess
from pathlib import Path

def main():
    d=Path(os.environ['A11_INSTRUMENT_ROOT']).parent;p=d/'fixture.cpp'
    p.write_text('''#include <cstdio>
constexpr bool valid(int n){return n==8;} static_assert(valid(8));
int f(int x){float a[8]{}; int y=1,z=2; for(int i=0;i<3;++i)y+=x; if(y==7)return y;return 0;}
int main(){int y=f(2);std::printf("%d\\n",y);return y!=7;}
''')
    exe=d/'fixture';subprocess.run(['g++','-std=c++17','-O2',str(p),'-o',str(exe)],check=True)
    result=subprocess.check_output([str(exe)],text=True);assert result=='7\n'
    build=next((d/'builds').glob('build_*'));sites=json.loads((build/'sites.json').read_text())
    binary=next((d/'events').glob('*.bin')).read_bytes();counts=struct.unpack('<'+'Q'*len(sites),binary)
    pairs=list(zip(sites,counts));fn=[v for s,v in pairs if s['kind']=='function'];assert fn==[1,1],fn
    assert sum(v for s,v in pairs if s['kind']=='iteration')==3
    assert sum(v for s,v in pairs if s['kind']=='comparison')==6
    assert sum(v for s,v in pairs if s['kind']=='automatic_object_bytes')==44
    print('Instrumentation functions/iterations/comparisons/scratch exact; model inputs/targets zero.')
if __name__=='__main__':main()

"""Add compile-time binding hooks while preserving the composition comparator."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'runtime/hierarchy_lab/throughput_probe'
p=root/'composition.cpp';s=p.read_text()
old='''        if(contextual&&!delay){
          // Driver constituents are already present.'''
new='''        if(contextual&&!delay){
#ifdef SHOEBOX_MULTI_BINDING
          if(r>=3)continue; // Every immediate slot is already a bound constituent.
#endif
          // Driver constituents are already present.'''
if new not in s:
 assert s.count(old)==1;s=s.replace(old,new)
p.write_text(s)

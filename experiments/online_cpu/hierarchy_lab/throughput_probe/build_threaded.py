from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT/'cached.cpp').read_text()
s=s.replace('#include <immintrin.h>','#include <immintrin.h>\n#include "thread_pool.inc"')
s=s.replace('struct Model {','struct Model {\n  RowWorkers*workers=nullptr;\n  ~Model(){delete workers;}')
start=s.index('      float*w=m.W+(r*E+best)*D*D,*b=m.bias+(r*E+best)*D;')
end=s.index('      if(m.diag.enabled',start)
old=s[start:end]
body=old.replace('for(int i=0;i<D;i++)','for(int i=lo;i<hi;i++)')
helper='''
struct ForwardRows {Model*m;int r,best;bool task_route;float gate;float*residual;};
static void forward_rows(void*opaque,int lo,int hi){
  auto&c=*static_cast<ForwardRows*>(opaque);auto&m=*c.m;auto&s=m.s;
  int r=c.r,best=c.best;bool task_route=c.task_route;float gate=c.gate;float*residual=c.residual;
'''+body+'}\n'
s=s[:start]+'''      ForwardRows job{&m,r,best,task_route,gate,residual};
      if(m.workers)m.workers->run(forward_rows,&job,D);else forward_rows(&job,0,D);
'''+s[end:]
at=s.index('extern "C" {');s=s[:at]+helper+s[at:]
s=s.replace('  Model&m=*static_cast<Model*>(p);State&s=m.s;double loss=0;','  Model&m=*static_cast<Model*>(p);State&s=m.s;double loss=0;\n  if(m.workers)m.workers->begin();')
s=s.replace('  return loss/std::max(1,length);','  if(m.workers)m.workers->end();\n  return loss/std::max(1,length);')
s+='''\nextern "C" void configure_threads(void*p,int count,const int*cpus){
  auto&m=*static_cast<Model*>(p);delete m.workers;m.workers=count>1?new RowWorkers(count,cpus):nullptr;
}\n'''
(ROOT/'threaded.cpp').write_text(s)

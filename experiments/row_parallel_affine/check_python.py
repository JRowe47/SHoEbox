"""Run the original twenty A9 controls with explicit scalar/candidate execution."""
import argparse,importlib,json,unittest
from pathlib import Path
from . import backend
from experiments.selective_record_key_credit import backend as old
from experiments.joint_cold_gate.adapter import Adapter
from experiments.selective_record_key_credit.credit import Counter,augmented

def main():
    p=argparse.ArgumentParser();p.add_argument('execution',choices=['scalar','row_parallel']);p.add_argument('out',type=Path);a=p.parse_args()
    assert not a.out.exists()
    names=['experiments.selective_record_key_credit.'+n for n in ('test_research','test_objective','test_spacers')]
    modules=[importlib.import_module(n) for n in names]
    if a.execution=='row_parallel':
        for module in modules:module.Model=backend.Model
        # Only the runtime entry point differs; original source-isolation test
        # still validates the immutable A9 policy recipe against A8.
        original_library=old.library
        def selected(policy='e8'):
            if policy=='a9':return backend.library()
            return original_library(policy)
        # Compile before substituting, avoiding a reference lookup recursion.
        backend.library();old.library=selected
    counters={};observed={};rejected=[];depth=0
    def wrap(fn):
        def run(self,x,y,mask=None,labels=None,**kw):
            nonlocal depth
            top=depth==0;depth+=1;policy=getattr(self,'policy','e8');n=self.inventory()['bytes']
            try:r=fn(self,x,y,mask,labels,**kw)
            except Exception:
                if top:rejected.append(dict(policy=policy,requested_bytes=len(x)))
                raise
            finally:depth-=1
            if top:
                c=counters.setdefault(policy,Counter());c.n=n;c.run(x,y,mask,labels,**kw)
            return r
        return run
    oldrun=old.Model.run;adapterrun=Adapter.run;obs=Adapter.observe
    def observe(self,b):
        r=obs(self,b);policy=getattr(self,'policy','e8');observed[policy]=observed.get(policy,0)+1;return r
    old.Model.run=wrap(oldrun);Adapter.run=wrap(adapterrun);Adapter.observe=observe
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    a.out.write_text(json.dumps(dict(execution=a.execution,tests=result.testsRun,passed=result.wasSuccessful(),
      updates={k:augmented(v.total,k) for k,v in counters.items()},observed_bytes=observed,rejected_calls=rejected,
      research_fitting_bytes=0),indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
if __name__=='__main__':main()

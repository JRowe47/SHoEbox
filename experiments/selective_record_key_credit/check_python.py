"""Instrument successful discarded Python controls, including explicit credit."""
import json,sys,unittest
from pathlib import Path
from experiments.joint_cold_gate.adapter import Adapter
from .backend import Model
from .credit import Counter

def main():
    out=Path(sys.argv[1]);assert not out.exists();rows={};observe={};rejected=[];depth=0
    original=Adapter.run;candidate=Model.run;obs=Adapter.observe
    def wrapper(fn):
        def counted(self,x,y,mask=None,labels=None,**kwargs):
            nonlocal depth
            top=depth==0;depth+=1;policy=getattr(self,'policy','historical-e8');n=self.inventory()['bytes']
            try:result=fn(self,x,y,mask,labels,**kwargs)
            except Exception:
                if top:rejected.append(dict(policy=policy,input_bytes_requested=len(x)))
                raise
            finally:depth-=1
            if top:
                m=rows.setdefault(policy,Counter());m.n=n;m.run(x,y,mask,labels,**kwargs)
            return result
        return counted
    def observed(self,byte):
        result=obs(self,byte);policy=getattr(self,'policy','historical-e8');observe[policy]=observe.get(policy,0)+1;return result
    Adapter.run=wrapper(original);Model.run=wrapper(candidate);Adapter.observe=observed
    try:
        suite=unittest.defaultTestLoader.loadTestsFromNames(['experiments.selective_record_key_credit.test_research','experiments.selective_record_key_credit.test_objective','experiments.selective_record_key_credit.test_spacers'])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        out.write_text(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),updates={k:v.total for k,v in rows.items()},observed_bytes=observe,rejected_calls=rejected),indent=2)+'\n')
    finally:Adapter.run=original;Model.run=candidate;Adapter.observe=obs
    raise SystemExit(not result.wasSuccessful())
if __name__=='__main__':main()

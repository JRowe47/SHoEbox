"""Run pre-fit Python tests with exact successful-update event instrumentation."""
import json,sys,unittest
from pathlib import Path
from experiments.joint_cold_gate.adapter import Adapter
from experiments.record_presence_cost.account_correctness import Counter

def main():
    out=Path(sys.argv[1]);assert not out.exists();rows={};observe={};rejected=[]
    run=Adapter.run;obs=Adapter.observe
    def counted(self,x,y,mask=None,labels=None):
        policy=getattr(self,'policy','historical-e8');n=self.inventory()['bytes']
        try:result=run(self,x,y,mask,labels)
        except Exception:
            rejected.append(dict(policy=policy,input_bytes_requested=len(x)));raise
        m=rows.setdefault(policy,Counter());m.n=n;m.run(x,y,mask,labels)
        return result
    def observed(self,byte):
        result=obs(self,byte);policy=getattr(self,'policy','historical-e8')
        observe[policy]=observe.get(policy,0)+1;return result
    Adapter.run=counted;Adapter.observe=observed
    try:
        suite=unittest.defaultTestLoader.loadTestsFromNames([
            'experiments.balanced_record_context.test_research',
            'experiments.balanced_record_context.test_objective',
            'experiments.balanced_record_context.test_spacers'])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        out.write_text(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),updates={k:v.total for k,v in rows.items()},observed_bytes=observe,rejected_calls=rejected),indent=2)+'\n')
    finally:Adapter.run=run;Adapter.observe=obs
    raise SystemExit(not result.wasSuccessful())

if __name__=='__main__':main()

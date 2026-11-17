"""Preserved historical Python suites, with successful adapter-call accounting."""
import argparse,importlib,json,re,unittest
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('prior_log',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    assert not a.out.exists()
    modules=sorted(set(re.findall(r'\((experiments\.[\w.]+)\.[^.]+\.[^.]+\)',a.prior_log.read_text())))
    assert len(modules)==33
    for name in modules:importlib.import_module(name)
    import sys
    classes=set()
    for name,m in list(sys.modules.items()):
        if not name.startswith('experiments.'):continue
        for value in vars(m).values():
            if isinstance(value,type) and value.__module__.startswith('experiments.') and 'run' in value.__dict__:classes.add(value)
    rows={};depth=0
    def wrap(fn):
        def run(self,x,y,*args,**kw):
            nonlocal depth
            top=depth==0;depth+=1
            try:result=fn(self,x,y,*args,**kw)
            finally:depth-=1
            if top:
                name=type(self).__module__+'.'+type(self).__name__
                row=rows.setdefault(name,dict(calls=0,input_bytes=0,padding_bytes=0))
                row['calls']+=1;row['input_bytes']+=len(x);row['padding_bytes']+=bytes(x).count(9)
            return result
        return run
    for cls in classes:setattr(cls,'run',wrap(cls.run))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    a.out.write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,modules=modules,
        successful_adapter_calls=rows,research_fitting_bytes=0),indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
if __name__=='__main__':main()

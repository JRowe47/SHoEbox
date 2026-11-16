"""Zero-fit A4 rejection by every protected research/public loader."""
from .backend import Model
from experiments.interference_retention.backend import Model as A1
from experiments.query_first_retention.backend import Model as A2
from experiments.chronological_retention.backend import Model as A3
from experiments.joint_request_gate.adapter import Adapter as E7
from experiments.joint_cold_gate.adapter import Adapter as E8
from shoe_multiscale import Model as Public

def main():
    controls=[('a1',lambda:A1('a1'),lambda b:A1.loads(b,'a1')),
              ('a2',lambda:A2('a2'),lambda b:A2.loads(b,'a2')),
              ('a3',lambda:A3('a3'),lambda b:A3.loads(b,'a3')),
              ('e7',E7,E7.loads),('e8',E8,E8.loads),('public',Public,Public.loads)]
    checks=0
    def reject(load,raw):
        nonlocal checks
        try:
            other=load(raw);other.close()
        except (RuntimeError,ValueError):checks+=1;return
        raise AssertionError('cross-policy load accepted')
    with Model('a4') as a:
        for name,create,load in controls:
            reject(load,a.dumps())
            with create() as b:reject(lambda raw:Model.loads(raw,'a4'),b.dumps())
        reject(Public.import_e8_research,a.dumps())
    print(checks,'cross-policy rejection checks passed; zero fitting/inference bytes')
if __name__=='__main__':main()

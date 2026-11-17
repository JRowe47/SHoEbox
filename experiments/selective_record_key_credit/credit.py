"""Independent data-only executed/omitted credit ledger; no model calls."""
import numpy as np
from experiments.record_presence_cost.account_correctness import Counter as Original
class Counter(Original):
    def __init__(self):
        super().__init__()
        self.total.update(record_key_executed=0,record_key_omitted=0,record_key_omitted_rows=0)
    def run(self,x,y,mask=None,labels=None,*,allow_record_key_update=True):
        before=self.n;omitted=executed=0
        flags=np.full(len(x),allow_record_key_update,dtype=bool) if np.isscalar(allow_record_key_update) else np.asarray(allow_record_key_update,dtype=bool)
        if labels is not None:
            for i in range(len(x)):
                n=before+i+1
                if n>=8 and n%4==0:
                    positive=int(np.count_nonzero(labels[i,1:5]>=0))
                    if flags[i]:executed+=positive
                    else:omitted+=positive
        result=super().run(x,y,mask,labels)
        self.total['local_projection_steps']-=16*omitted
        self.total['record_key_executed']+=executed
        self.total['record_key_omitted']+=omitted
        self.total['record_key_omitted_rows']+=16*omitted
        return result

def augmented(t,policy):
    t=dict(t);executed=t['positive_query_keys']+t['record_key_executed'] if policy in ('a8','a9') else 0
    omitted=t['record_key_omitted'] if policy=='a9' else 0
    t.update(hybrid_attraction_components=executed,hybrid_discrimination_components=executed,
      hybrid_gradient_vectors=executed,hybrid_gradient_rows=16*executed,hybrid_code_comparisons=256*executed,
      hybrid_coordinate_distance_comparisons=4096*executed,omitted_hybrid_components=omitted,
      omitted_hybrid_code_comparisons=256*omitted,omitted_hybrid_coordinate_comparisons=4096*omitted)
    for sign in ('positive','negative'):
        n=t['record_presence_'+sign];t['record_'+sign+'_loss_components']=n;t['record_'+sign+'_gradient_rows']=n
        t['record_'+sign+'_weighted_loss_units']=n*(4 if sign=='positive' or policy in ('a8','a9') else 1)
    return t

"""Adapt exact historical cursor drivers to the public API; no new objective."""
from shoe_multiscale import Model


class PublicCursorModel:
    def __init__(self,model,ledger=None):
        self.model,self.ledger=model,ledger
    def __enter__(self):return self
    def __exit__(self,*args):self.model.close()
    def close(self):self.model.close()
    def reset(self):self.model.reset_context(clear_memory=True)
    def finish(self):self.model.finish()
    def inventory(self):return self.model.stats()
    def dumps(self):return self.model.dumps()
    @classmethod
    def loads(cls,data):return cls(Model.loads(data))
    def run(self,x,y,mask=None,labels=None):
        before=self.model.readout_stats()
        r=self.model.run_objectives(x,y,mask=mask,local_labels=labels)
        if self.ledger is not None:
            after=self.model.readout_stats()
            self.ledger['input_bytes']+=len(x)
            self.ledger['byte_9_inputs']+=x.count(9)
            self.ledger['objective_targets']+=r['targets']
            for k in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):
                self.ledger[k]+=after[k]-before[k]
        return r['loss_sum'],r['targets']
    def observe(self,b):
        # Public observe remains inference-only, as historically specified.
        # A zero objective mask consumes the final reconstruction byte exactly.
        self.run(bytes([b]),b'\0',b'\0')
    def features(self):return self.model.reconstruction_features()
    def credit(self,dx):self.model.apply_reconstruction_credit(dx)


def ledger():
    return dict(input_bytes=0,byte_9_inputs=0,objective_targets=0,
                local_projection_steps=0,local_presence_steps=0,gate_targets=0,
                local_training_windows=0)

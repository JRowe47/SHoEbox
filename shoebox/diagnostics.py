"""Bounded transport diagnostics: confidence, conditioning and hard-code usage."""
from collections import Counter
import math
import time
import torch


class PortMonitor:
    def __init__(self, model, *, tap_events=()):
        self.scale = model.cfg.port_scale
        self.context_ports = {(edge.source,edge.width) for spec in model.specs
                              for edge in spec.inputs if edge.delayed and edge.width == self.scale}
        self.primary_ports = {(name,model.cfg.max_port_bytes) for name in ('A1','B1','L2')}
        self.monitored_ports = self.context_ports|self.primary_ports
        self.records = {}
        self.seconds = 0.
        self.tap_events = set(tap_events)
        self.taps = {}
        self.expert_counts = {}

    def __call__(self,name,event,revision,result):
        if event in self.tap_events and name in ('A1','B1','L2'):
            self.taps[(name,event)] = result['latent']
        start = time.perf_counter()
        with torch.no_grad():
            for width,p in result['probabilities'].items():
                # Only requested context and early primary ports are monitored.
                if (name,width) not in self.monitored_ports:
                    continue
                key = f'{name}:{width}'
                p = p.detach().to(dtype=torch.float64,device='cpu')
                values = result['frames'][width].values.detach().cpu()
                if key not in self.records:
                    self.records[key] = {'observations':0,'p_sum':torch.zeros(width,256,dtype=torch.float64),
                                          'entropy_sum':torch.zeros(width,dtype=torch.float64),
                                          'hard_counts':torch.zeros(width,256,dtype=torch.float64),'frames':Counter()}
                r = self.records[key]
                r['observations'] += p.shape[0]
                r['p_sum'] += p.sum(0)
                r['entropy_sum'] += -(p*p.clamp_min(1e-300).log2()).sum(-1).sum(0)
                for index in range(width):
                    r['hard_counts'][index] += torch.bincount(values[:,index],minlength=256)
                r['frames'].update(bytes(row) for row in values.tolist())
            ids = result['experts'].detach().cpu()
            if name not in self.expert_counts:
                self.expert_counts[name] = [Counter() for _ in range(ids.shape[-2])]
            for layer in range(ids.shape[-2]):
                self.expert_counts[name][layer].update(ids[:,layer].reshape(-1).tolist())
        self.seconds += time.perf_counter()-start

    def snapshot(self):
        ports = {}
        def entropy(p):
            return -(p*p.clamp_min(1e-300).log2()).sum(-1)
        for key,r in self.records.items():
            n = r['observations']
            conditional = r['entropy_sum']/n
            marginal = entropy(r['p_sum']/n)
            hard = entropy(r['hard_counts']/n)
            histogram = r['frames']
            ports[key] = {'observations':n,'port_bytes':int(key.split(':')[1]),
                          'legacy_port_bytes':int(key.split(':')[1])//self.scale,
                          'role':'primary' if tuple([key.split(':')[0],int(key.split(':')[1])]) in self.primary_ports else 'context',
                          'conditional_entropy_bits':float(conditional.mean()),
                          'marginal_entropy_bits':float(marginal.mean()),
                          'conditioning_js_bits':float((marginal-conditional).mean()),
                          'hard_byte_entropy_bits':float(hard.mean()),
                          'unique_complete_frames':len(histogram),
                          'unique_frame_fraction':len(histogram)/n,
                          'most_common_frame_fraction':max(histogram.values())/n,
                          'frame_collision_probability':sum(c*c for c in histogram.values())/(n*n)}
        experts = {name:[{'unique_experts':len(counts),'most_used_fraction':max(counts.values())/sum(counts.values())}
                          for counts in layers] for name,layers in self.expert_counts.items()}
        return {'ports':ports,'expert_usage':experts,'monitor_seconds':self.seconds}


class SavedTensorAccounting:
    """Forward saved-tensor traffic and unique referenced storage, not peak RSS.

    Parameter storage and views are deduplicated. The hook does not copy or
    retain tensors beyond what ordinary autograd itself retains.
    """
    def __init__(self):
        self.save_calls = 0
        self.logical_bytes = 0
        self.storage = {}

    def pack(self,tensor):
        self.save_calls += 1
        self.logical_bytes += tensor.numel()*tensor.element_size()
        key = (str(tensor.device),tensor.untyped_storage().data_ptr())
        self.storage[key] = tensor.untyped_storage().nbytes()
        # Saved-tensor hooks must not return the input tensor object: a tensor
        # can own the very grad_fn that stores the packed value. A detached
        # alias preserves storage and backward values without that cycle.
        return tensor.detach()

    def unpack(self,tensor):
        return tensor

    def summary(self):
        return {'saved_tensor_calls':self.save_calls,'saved_tensor_logical_bytes':self.logical_bytes,
                'saved_unique_storage_bytes':sum(self.storage.values()),'saved_unique_storages':len(self.storage)}

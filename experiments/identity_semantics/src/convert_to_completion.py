"""Convert v31 relays/v32 messages to v33 completion at a fresh stream boundary."""
import argparse,ctypes as ct,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core

def convert(source,destination):
 source,destination=Path(source),Path(destination)
 if source.resolve()==destination.resolve() or destination.exists():raise ValueError('Choose a new, nonexistent checkpoint path')
 with np.load(source,allow_pickle=False) as a:
  if (int(a['version'][0]),str(a['experiment_kernel'])) not in ((31,'relays'),(32,'messages')):
   raise ValueError('Expected a version-31 relay or version-32 message checkpoint')
 old=Model.load(source)
 try:
  policies={k:getattr(old,k) for k in ('config','lace','state_config','probe','utility','teaching','regional_gains','frozen_regions','task_route_lr_scale','a1_learning','f3_task_route_gain','exploration','traffic','composition')}
  if old.runtime_kernel=='messages':policies['messages']=old.messages
  new=Model(params=core.unflatten(old.flat.copy()),extra=old.extra_flat.copy(),fixed={k:v.copy() for k,v in old.fixed.items()},kernel='completion',**policies)
  try:
   previous=old.composition_state(fitted=True);fitted=new.composition_state(fitted=True)
   if fitted.size!=previous.size+16*4 or previous[:4].view(np.uint32)[0] not in (5,6):raise ValueError('Unexpected fitted layout')
   fitted[:previous.size]=previous;fitted[:4]=np.array([7],np.uint32).view(np.uint8)
   new.import_composition(fitted,fitted=True)
   means=old.means();new.lib.probe_set_means(new.handle,means.ctypes.data_as(ct.POINTER(ct.c_float)));new.set_ymean(old.ymean())
   np.testing.assert_array_equal(old.semantic_parameters(),new.semantic_parameters())
   destination.parent.mkdir(parents=True,exist_ok=True);new.save(destination)
  finally:new.close()
 finally:old.close()
 return dict(version=33,abi=7,exact_continuation=False,preserved='neural/semantic weights, fitted dictionaries, fixed transports, policies and preprocessing means',reset='all event/working/associative memory, pending teaching and eligibility',added='16 neutral payload-gate parameters')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(json.dumps(convert(a.source,a.destination),indent=2))

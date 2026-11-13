"""Convert v31 relays/v32 messages/v33/v34 completion to v39 completion at a fresh stream boundary."""
import argparse,ctypes as ct,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core

def convert(source,destination):
 source,destination=Path(source),Path(destination)
 if source.resolve()==destination.resolve() or destination.exists():raise ValueError('Choose a new, nonexistent checkpoint path')
 with np.load(source,allow_pickle=False) as a:
  if (int(a['version'][0]),str(a['experiment_kernel'])) not in ((31,'relays'),(32,'messages'),(33,'completion'),(34,'completion')):
   raise ValueError('Expected a version-31 relay, version-32 message or version-33/34 completion checkpoint')
  if int(a['version'][0]) in (33,34):
   return convert_coverage_checkpoint(a,destination)
 old=Model.load(source)
 try:
  policies={k:getattr(old,k) for k in ('config','lace','state_config','probe','utility','teaching','regional_gains','frozen_regions','task_route_lr_scale','a1_learning','f3_task_route_gain','exploration','traffic','composition')}
  if old.runtime_kernel=='messages':policies['messages']=old.messages
  new=Model(params=core.unflatten(old.flat.copy()),extra=old.extra_flat.copy(),fixed={k:v.copy() for k,v in old.fixed.items()},kernel='completion',local_credit=False,content_experts=False,pattern_classes=False,paired_memory=False,**policies)
  try:
   previous=old.composition_state(fitted=True);fitted=new.composition_state(fitted=True)
   if fitted.size!=previous.size+16*4 or previous[:4].view(np.uint32)[0] not in (5,6):raise ValueError('Unexpected fitted layout')
   fitted[:previous.size]=previous;fitted[:4]=np.array([10],np.uint32).view(np.uint8)
   new.import_composition(fitted,fitted=True)
   means=old.means();new.lib.probe_set_means(new.handle,means.ctypes.data_as(ct.POINTER(ct.c_float)));new.set_ymean(old.ymean())
   np.testing.assert_array_equal(old.semantic_parameters(),new.semantic_parameters())
   destination.parent.mkdir(parents=True,exist_ok=True);new.save(destination)
  finally:new.close()
 finally:old.close()
 return dict(version=39,abi=10,exact_continuation=False,preserved='neural/semantic weights, fitted dictionaries, fixed transports, policies and preprocessing means',reset='all event/working/associative memory, pending teaching and eligibility',added='16 neutral payload-gate parameters')

def convert_coverage_checkpoint(a,destination):
 """Import fitted v33/v34 data; expired eligibility cannot be recovered."""
 if int(a['experts'][0])!=core.E:raise ValueError('Set matching SHOEBOX_EXPERTS before import')
 policies={k:json.loads(str(a[k+'_json'])) for k in ('config','lace','state_config','probe','utility','composition','messages')}
 # These legacy port switches were accepted no-ops in v33, not fitted behavior.
 for key in ('port_mode','port_gain','balance'):policies['probe'][key]=0
 policies.update(teaching=a['teaching_scales'],regional_gains=a['regional_gains'],
  frozen_regions=a['frozen_regions'],task_route_lr_scale=float(a['task_route_lr_scale']),
  a1_learning=a['a1_learning_scales'],f3_task_route_gain=float(a['f3_task_route_gain']),
  exploration=json.loads(str(a['exploration_json'])),traffic=json.loads(str(a['traffic_json'])))
 new=Model(params=core.unflatten(a['flat']),extra=a['extra'],fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},kernel='completion',local_credit=False,content_experts=False,pattern_classes=False,paired_memory=False,**policies)
 try:
  old=np.ascontiguousarray(a['composition_state'],np.uint8);fitted=new.composition_state(fitted=True)
  if old.size<new.composition_state().size or old[:4].view(np.uint32)[0] not in (7,8):
   raise ValueError('Unexpected historical completion layout')
  fitted[:]=old[:fitted.size];fitted[:4]=np.array([10],np.uint32).view(np.uint8)
  new.import_composition(fitted,fitted=True)
  # Preserve fitted preprocessing means, clear all runtime counters and buffers.
  for key,clean,method,count in [('probe_state',new.probe_state(),'probe_import',18*64),
                               ('utility_state',new.utility_state(),'utility_import',64)]:
   saved=np.ascontiguousarray(a[key],np.uint8)
   if saved.size!=clean.size:raise ValueError(key+' ABI mismatch')
   start=8*4;stop=start+count*4;clean[start:stop]=saved[start:stop]
   getattr(new.lib,method)(new.handle,clean.ctypes.data)
  destination.parent.mkdir(parents=True,exist_ok=True);new.save(destination)
 finally:new.close()
 return dict(version=39,abi=10,exact_continuation=False,
  preserved='neural/semantic weights, payload gates, fitted dictionaries, policies and preprocessing means',
  reset='all event/working/associative memory, pending teaching and eligibility',
  changed='primary occurrence clocks and lifetime-owned neural eligibility')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(json.dumps(convert(a.source,a.destination),indent=2))

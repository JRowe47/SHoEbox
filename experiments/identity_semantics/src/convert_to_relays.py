"""Convert fitted v30 semantics to v31 relays at a fresh stream boundary."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core


def convert(source,destination):
 source,destination=Path(source),Path(destination)
 if source.resolve()==destination.resolve() or destination.exists():
  raise ValueError('Choose a new, nonexistent checkpoint path')
 with np.load(source,allow_pickle=False) as a:
  if int(a['version'][0])!=30 or str(a['experiment_kernel'])!='semantics':
   raise ValueError('Expected a version-30 learned-semantics checkpoint')
 old=Model.load(source)
 try:
  policies={k:getattr(old,k) for k in ('config','lace','state_config','probe','utility','teaching','regional_gains','frozen_regions','task_route_lr_scale','a1_learning','f3_task_route_gain','exploration','traffic','composition')}
  new=Model(params=core.unflatten(old.flat.copy()),extra=old.extra_flat.copy(),
            fixed={k:v.copy() for k,v in old.fixed.items()},kernel='relays',**policies)
  try:
   fitted=old.composition_state(fitted=True)
   # Only the fitted prefix has the same layout. Never reinterpret old events
   # or pending teaching; their contract changed and their ABI is rejected.
   if fitted.size!=new.composition_state(fitted=True).size or fitted[:4].view(np.uint32)[0]!=4:
    raise ValueError('Unexpected semantics fitted layout')
   fitted[:4]=np.array([5],np.uint32).view(np.uint8)
   new.import_composition(fitted,fitted=True)
   means=old.means();new.lib.probe_set_means(new.handle,means.ctypes.data_as(ct.POINTER(ct.c_float)))
   new.set_ymean(old.ymean())
   np.testing.assert_array_equal(old.semantic_parameters(),new.semantic_parameters())
   destination.parent.mkdir(parents=True,exist_ok=True);new.save(destination)
  finally:new.close()
 finally:old.close()
 return dict(version=31,abi=5,exact_continuation=False,
             preserved='neural parameters, semantic matrices, dictionaries, fixed transports, policies, fitted preprocessing means',
             reset='stream context, pending forecasts, eligibility, online adaptation state, hippocampal associations')


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('destination')
 a=p.parse_args();print(json.dumps(convert(a.source,a.destination),indent=2))

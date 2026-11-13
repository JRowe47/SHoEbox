"""Explicit weight conversion; new binding dictionaries and working state."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core
def convert(source,destination):
 source=Path(source);destination=Path(destination)
 if source.resolve()==destination.resolve() or destination.exists():raise ValueError('Choose a new, nonexistent checkpoint path')
 with np.load(source,allow_pickle=False) as a:
  if int(a['version'][0])!=28 or int(a['experts'][0])!=core.E:raise ValueError('Expected a version-28 checkpoint with matching expert count')
  policy={k:json.loads(str(a[k+'_json'])) for k in ['config','lace','state_config','probe','utility']}
  m=Model(params=core.unflatten(a['flat']),extra=a['extra'],fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},kernel='binding',**policy)
  for key,method in [('teaching_scales','configure_teaching'),('regional_gains','configure_regional_gains'),('a1_learning_scales','configure_a1_learning')]:
   if key in a:getattr(m,method)(a[key])
  if 'frozen_regions' in a:m.configure_training_policy(a['frozen_regions'],float(a['task_route_lr_scale']))
  if 'composition_json' in a:m.configure_composition(json.loads(str(a['composition_json'])))
  np.testing.assert_array_equal(m.flat,a['flat']);np.testing.assert_array_equal(m.extra_flat,a['extra'])
  destination.parent.mkdir(parents=True,exist_ok=True);m.save(destination);m.close()
 return dict(source=str(source),destination=str(destination),version=29,abi=3,reset='all dictionaries, stream context, eligibility, fitted preprocessing means, and hippocampal associations',preserved='neural parameters, fixed transports, supported teaching/gain/state policies',exact_continuation=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(json.dumps(convert(a.source,a.destination),indent=2))

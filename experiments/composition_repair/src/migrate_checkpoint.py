"""Start a composition run from legacy weights with fresh event/memory state.

This is an explicit architecture conversion, not exact continuation. Existing
hippocampal keys encode contextual mixtures, so they cannot be reused as the new
content-addressed cues. The source checkpoint is never overwritten.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from runtime import Model,core

def migrate(source,destination):
    source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Choose a new checkpoint path')
    with np.load(source,allow_pickle=False) as a:
        if 'architecture_json' in a:
            arch=json.loads(str(a['architecture_json']))
            if arch['width']!=64 or len(set(arch['experts']))!=1:raise ValueError('This repair supports width 64 and uniform regional expert counts')
            count=arch['experts'][0]
            settings=json.loads(str(a['settings_json']))
            kwargs={key:settings[key] for key in ('config','lace','state_config','probe','utility')}
        else:
            count=int(a['experts'][0])
            kwargs={key:json.loads(str(a[key+'_json'])) for key in ('config','lace','state_config','probe','utility')}
        if count!=core.E:raise ValueError('Set SHOEBOX_EXPERTS to the source expert count before starting Python')
        if int(a['version'][0]) not in range(16,28):raise ValueError('Expected a legacy checkpoint, versions 16 through 27')
        model=Model(params=core.unflatten(a['flat']),fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},extra=a['extra'],kernel='composition',**kwargs)
        for field,method in (('teaching_scales','configure_teaching'),('regional_gains','configure_regional_gains'),('a1_learning_scales','configure_a1_learning')):
            if field in a:getattr(model,method)(a[field])
        destination.parent.mkdir(parents=True,exist_ok=True);model.save(destination);model.close()
    return dict(source=str(source),checkpoint=str(destination),preserved='neural weights, fixed transforms, bottom gates/prototypes and supported policies',reset='stream context, eligibility, hippocampal associations, fitted context means and constituent dictionaries',architecture_conversion=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source');parser.add_argument('destination');args=parser.parse_args()
    print(json.dumps(migrate(args.source,args.destination),indent=2))

"""Read-only category scoring through the existing native MAD bridge."""
from dataclasses import replace
import numpy as np

from .tasks import VALUE, parse


def native_categories(model, cases, spec):
    """Changed/unchanged teacher-forced accuracy, with identical observed input.

    A category mask only selects which predictions are counted. With rollout=0
    it cannot change the byte stream consumed by the frozen model.
    """
    if spec.task not in ('set', 'copy', 'swap'):
        return {}
    totals = {name:np.zeros(7) for name in ('changed','unchanged')}
    with model.clone() as frozen:
        for ep in cases:
            records, queries = parse(ep)
            initial = {k:v-VALUE for _,k,v in records[:spec.items]}
            changed = [initial[k] != y for k,y in zip(queries,ep.answer)]
            positions = [i for i,m in enumerate(ep.mask) if m]
            for name,flag in (('changed',True),('unchanged',False)):
                mask = bytearray(len(ep.mask))
                for position,is_changed in zip(positions,changed):
                    mask[position] = int(is_changed == flag)
                totals[name] += frozen.score(replace(ep,mask=bytes(mask)),rollout=0)
    result = {'category_accuracy_mode':'teacher_forced'}
    for name,row in totals.items():
        result[name+'_targets'] = int(row[0])
        result[name+'_accuracy'] = float(row[3]/row[0]) if row[0] else None
        result['native_'+name+'_accuracy'] = float(row[6]/row[0]) if row[0] else None
    return result

"""Frozen external admission attribution; no new inference machinery or fitting."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .backend import Model
from .tasks import cases
from .run import write,sha
from experiments.joint_cold_gate.adapter import prefix_labels

def logits(gates, stable):
    x=np.asarray(stable,dtype=np.float32)*np.float32(8)
    # Sequential float32 sum matches native dot; final bias is added afterward.
    return np.cumsum(gates[:,:64]*x,axis=1,dtype=np.float32)[:,-1]+gates[:,64]

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    hashes={str(p.relative_to(a.runs)):sha(p) for p in a.runs.glob('*/*.qr')}
    write(a.out.with_suffix('.lock.json'),dict(source_sha256=sha(Path(__file__)),checkpoints=hashes,fitting_bytes=0,audit_opened=False))
    total=padding=0;rows=[];start=time.process_time()
    for seed in (1,7,19):
        for arm in ('e8','a6','a7'):
            records=[];events=[];counts=dict(full_diverse_pairs=0,full_diverse_recognized=0,admission_decisions_checked=0,true_record_phase_events=0,true_record_phase_false_negatives=0,full_diverse_phase_tests=0,full_diverse_phase_false_positives=0)
            with Model.loads((a.runs/str(seed)/(arm+'.qr')).read_bytes(),arm) as model,model.clone() as f:
                # E8 ABI omits the raw head export. Its target losses are traced
                # without inventing a recognition/priority diagnosis for E8.
                gate=None if arm=='e8' else f.probe()[:325].reshape(5,65)[:4].astype(np.float32)
                for index,c in enumerate(cases('dev',32,condition='diverse256')):
                    f.reset();labels=prefix_labels(c.ep);known={};before=f.state();lost=set()
                    for t,byte in enumerate(c.ep.data[:-1]):
                        f.observe(byte);total+=1;padding+=int(byte==9)
                        if (t+1)%4:continue
                        after=f.state();bc,clock,previous,units=before;ac,newclock,incoming,newunits=after
                        # Record the required original signature once its labeled
                        # pair completes. Loss events then use exact stored bytes.
                        for ph,packed in enumerate(labels[t,1:5]):
                            if packed>=0 and int(packed)//256 in c.query_keys:
                                key=int(packed)//256;known[key]=(incoming.copy(),ph)
                        full_diverse=t-7>=c.distractor_start and t<c.distractor_end
                        if gate is not None:
                            old_scores=[float(logits(gate,u[1:]).max()) for u in units[:bc]]
                            incoming_logits=logits(gate,incoming);incoming_score=float(incoming_logits.max())
                            for ph,packed in enumerate(labels[t,1:5]):
                                if packed>=0:
                                    counts['true_record_phase_events']+=1
                                    counts['true_record_phase_false_negatives']+=int(incoming_logits[ph]<=0)
                            if full_diverse:
                                counts['full_diverse_pairs']+=1
                                counts['full_diverse_recognized']+=int(incoming_score>0)
                                counts['full_diverse_phase_tests']+=4
                                counts['full_diverse_phase_false_positives']+=int(sum(incoming_logits>0))
                            # Exact full-cache admission: unchanged pair duplicate
                            # suppression, signature refresh, recognition priority.
                            if bc==16:
                                expected=None
                                if not np.array_equal(incoming,previous):
                                    matches=[k for k in range(bc) if np.array_equal(units[k,1:],incoming)]
                                    if matches:expected=matches[0]
                                    else:
                                        neg=[k for k in range(bc) if old_scores[k]<=0]
                                        if neg:expected=min(neg,key=lambda k:units[k,0])
                                        elif incoming_score>0:expected=min(range(bc),key=lambda k:units[k,0])
                                changed=[k for k in range(bc) if newunits[k,0]!=units[k,0]]
                                assert changed==([] if expected is None else [expected]),(seed,arm,index,t,changed,expected)
                                counts['admission_decisions_checked']+=1
                        for key,(sig,ph) in known.items():
                            old=[k for k in range(bc) if np.array_equal(units[k,1:],sig)]
                            now=[k for k in range(ac) if np.array_equal(newunits[k,1:],sig)]
                            if old and not now:
                                lost.add(key);reason='historical_oldest' if gate is None else ('oldest_unrecognized' if old_scores[old[0]]<=0 else 'all_slots_recognized')
                                events.append(dict(presentation=index,key=key,byte_position=t,stage='filler' if t<c.distractor_end else 'query_tail',
                                    full_diverse=full_diverse,reason=reason,occupancy=bc,
                                    recognized_slots=None if gate is None else sum(x>0 for x in old_scores),
                                    target_score=None if gate is None else old_scores[old[0]],incoming_score=None if gate is None else incoming_score))
                        before=after
                    for key,(sig,ph) in known.items():
                        records.append(dict(presentation=index,key=key,lost_once=key in lost,present_final=any(np.array_equal(u[1:],sig) for u in after[3][:after[0]])))
            summary=dict(targets=len(records),target_lost_fraction=sum(r['lost_once'] for r in records)/len(records),
                         final_presence=sum(r['present_final'] for r in records)/len(records),loss_events=len(events),
                         loss_reasons={reason:sum(e['reason']==reason for e in events) for reason in ('historical_oldest','oldest_unrecognized','all_slots_recognized')},
                         loss_stages={stage:sum(e['stage']==stage for e in events) for stage in ('filler','query_tail')},**counts)
            rows.append(dict(seed=seed,arm=arm,summary=summary,events=events,records=records))
            write(a.out,dict(rows=rows,inference_bytes=total,padding_bytes=padding,cpu_seconds=time.process_time()-start,fitting_bytes=0,audit_opened=False))
            print(seed,arm,summary,flush=True)
    assert hashes=={str(p.relative_to(a.runs)):sha(p) for p in a.runs.glob('*/*.qr')}
if __name__=='__main__':main()

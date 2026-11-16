import unittest, math
from dataclasses import replace
import numpy as np
from .backend import Model
from .tasks import worlds,cases,make,excluded,reserved_a1,reserved_a2,reserved_a3,interpret,fold
from .stream import new_cursor,advance
from .update_metrics import shortcuts
from experiments.joint_cold_gate.adapter import prefix_labels
from experiments.chronological_retention.backend import Model as Historical


def reference_probe(raw):
    """Independent numpy implementation; export contains no candidate oracle."""
    at=0
    def take(n):
        nonlocal at
        r=raw[at:at+n];at+=n;return r
    gate=take(325).reshape(5,65);key=take(5200).reshape(5,16,65);value=take(2080).reshape(4,8,65)
    groups=[]
    for _ in range(2):
        count,clock=take(2).astype(int);units=take(1040).reshape(16,65)[:count];groups.append((clock,units))
    (clock,qs),(_,rs)=groups
    def x(u):return np.r_[u[1:]*8,1].astype(np.float32)
    def affine32(weights, source):
        w=np.asarray(weights,dtype=np.float32)
        return np.cumsum(w[...,:64]*source[:64],axis=-1,dtype=np.float32)[...,-1]+w[...,64]
    def project(weights, source):return np.tanh(affine32(weights,source))
    def distance32(a,b):
        d=np.asarray(a,dtype=np.float32)-np.asarray(b,dtype=np.float32)
        return float(np.cumsum(d*d,dtype=np.float32)[-1])
    qslot=-1;best=-1e30;latest=-1;born=0
    for k,u in enumerate(qs):
        recognition=affine32(gate[4],x(u));score=recognition-.5*(clock-u[0])
        if recognition>0 and u[0]>born:latest=k;born=u[0]
        if score>best:best=score;qslot=k
    if latest>=0:qslot=latest
    qraw=x(qs[qslot]) if qslot>=0 else np.r_[np.zeros(64),1]
    query=project(key[4],qraw)
    candidates=[]
    for k,u in enumerate(rs):
        for ph in range(4):
            score=affine32(gate[ph],x(u));z=project(key[ph],x(u));dist=distance32(z,query)
            candidates.append(dict(k=k,ph=ph,score=score,dist=dist,born=u[0]))
    recognized=sorted([a for a in candidates if a['score']>0],key=lambda a:(-a['born'],a['k'],a['ph']))
    d_min=min((a['dist'] for a in recognized),default=0.)
    remaining=0.
    for a in recognized:
        logit=4.*(d_min+1.96-a['dist'])
        a['log_weight']=remaining-float(np.logaddexp(0.,-logit))
        remaining-=float(np.logaddexp(0.,logit))
    selected=sorted(recognized,key=lambda a:(-a['log_weight'],a['k'],a['ph']))[:2]
    if not recognized:selected=sorted(candidates,key=lambda a:-a['score'])[:2]
    keys=[];values=[];slots=[]
    for a in selected:
        slots.append((a['k'],a['ph']));keys.append(project(key[a['ph']],x(rs[a['k']])));values.append(project(value[a['ph']],x(rs[a['k']])))
    native_qslot=int(take(1)[0]);native_query=take(16);native_slots=[];native_keys=[];native_values=[]
    for _ in range(2):
        native_slots.append(tuple(take(2).astype(int)));native_keys.append(take(16));native_values.append(take(8))
    native_attention=take(1)[0];native_h=take(8)
    assert native_qslot==qslot and slots==native_slots[:len(slots)],(slots,native_slots)
    np.testing.assert_allclose(query,native_query,atol=2e-6)
    np.testing.assert_allclose(keys,native_keys[:len(keys)],atol=2e-6);np.testing.assert_allclose(values,native_values[:len(values)],atol=2e-6)
    if recognized:
        attention=1. if len(selected)==1 else np.exp(-np.logaddexp(0.,selected[1]['log_weight']-selected[0]['log_weight']))
    else:
        diff=np.square(query-keys[1]).sum()-np.square(query-keys[0]).sum();attention=1/(1+np.exp(-4*diff))
    np.testing.assert_allclose(attention,native_attention,atol=2e-6)
    np.testing.assert_allclose(attention*values[0]+(1-attention)*(values[1] if len(values)>1 else 0),native_h,atol=2e-6)

class A4Tests(unittest.TestCase):
    def test_generic_objectives_freeze_local_heads(self):
        with Model('a4',seed=7) as a:
            before=a.probe()[:7605].copy();inventory=a.inventory()
            a.run(b'Generic byte objectives stay separate.',b'eneric byte objectives stay separate.!')
            np.testing.assert_array_equal(before,a.probe()[:7605])
            for name in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):
                self.assertEqual(a.inventory()[name],inventory[name])
    def test_a1_e7_public_checkpoint_separation(self):
        from experiments.interference_retention.backend import Model as A1
        from experiments.joint_request_gate.adapter import Adapter as E7
        from shoe_multiscale import Model as Public
        with Model('a4') as a:
            with self.assertRaises(RuntimeError):A1.loads(a.dumps(),'a1')
            with self.assertRaises(RuntimeError):E7.loads(a.dumps())
            with self.assertRaises((RuntimeError,ValueError)):Public.import_e8_research(a.dumps())
        with A1('a1') as a:
            with self.assertRaises(RuntimeError):Model.loads(a.dumps(),'a4')
        with E7() as a:
            with self.assertRaises(RuntimeError):Model.loads(a.dumps(),'a4')
    def test_fresh_grouping_and_reserved_exclusion(self):
        sets=[{w.group_key for w in worlds(p,n)} for p,n in [('train',512),('dev',128),('audit',256)]]
        for i,a in enumerate(sets):
            self.assertFalse(any(excluded(k) for k in a));self.assertFalse(a&(reserved_a1()|reserved_a2()|reserved_a3()))
            for b in sets[i+1:]:self.assertFalse(a&b)
        binary={w.group_key for w in worlds('dev',128,vocab=2)}
        self.assertFalse(binary&(sets[0]|sets[2]|reserved_a1()|reserved_a2()|reserved_a3()))
        self.assertTrue(all(fold(k)=='dev' for k in binary))
    def test_interpreter_and_no_shortcuts(self):
        for cond in ('pad0','repeat16','diverse16','compete16','overwrite16'):
            cs=cases('dev',128,condition=cond)
            self.assertEqual(max(shortcuts(cs).values()),0)
            for c in cs:
                d=bytes(255 if c.ep.mask[i] else b for i,b in enumerate(c.ep.data))
                self.assertEqual(interpret(replace(c.ep,data=d)),c.expected)
                self.assertNotEqual(*c.expected)
        for w in worlds('train',16):
            for changed in (0,1):
                c=make(w,'overwrite16',updated=changed)
                self.assertEqual(interpret(c.ep),c.expected)
                self.assertNotEqual(c.expected[changed],32+w.records[changed][1])
    def test_labels_causal_third_value(self):
        ep=cases('train',1,condition='overwrite16')[0].ep;labels=prefix_labels(ep)
        for t in range(1,len(ep.data)-1):
            altered=replace(ep,data=ep.data[:t]+bytes([238])*(len(ep.data)-t))
            np.testing.assert_array_equal(prefix_labels(altered)[:t],labels[:t])
        from experiments.mad.tasks import VALUE
        seen=set()
        for c in cases('train',16,condition='overwrite16'):
            seen.update(int(v)%256 for v in prefix_labels(c.ep)[:,1:5].ravel() if v>=0)
        self.assertEqual(seen,{VALUE,VALUE+1,VALUE+2})
    def test_controls_exact_and_policy_separation(self):
        ep=cases('train',1)[0].ep
        for policy in ('e8','a3'):
            with Model(policy,seed=7) as a,Historical(policy,seed=7) as b:
                self.assertEqual(a.dumps(),b.dumps())
                self.assertEqual(a.run(ep.data[:-1],ep.data[1:],labels=prefix_labels(ep)),b.run(ep.data[:-1],ep.data[1:],labels=prefix_labels(ep)))
                self.assertEqual(a.dumps(),b.dumps())
        for policy in ('e8','a3','a4'):
            with Model(policy) as a:
                for other in ('e8','a3','a4'):
                    if other!=policy:
                        with self.assertRaises(RuntimeError):Model.loads(a.dumps(),other)
    def test_python_native_parity_and_readonly(self):
        with Model('a4',seed=7) as a:
            for c in cases('train',4,condition='compete16'):
                ep=c.ep;a.reset();a.run(ep.data[:-1],ep.data[1:],labels=prefix_labels(ep))
            before=a.dumps()
            with a.clone() as f:
                f.reset()
                for i,b in enumerate(cases('train',1,condition='compete16')[0].ep.data[:-1]):
                    f.observe(b)
                    if i>=8:
                        raw=f.dumps();reference_probe(f.probe());self.assertEqual(raw,f.dumps())
                        self.assertTrue(np.isfinite(f.log_probs()).all());self.assertAlmostEqual(float(np.exp(f.log_probs()).sum()),1,places=5)
            self.assertEqual(before,a.dumps())
    def test_cursor_finish_and_lifecycle(self):
        for policy in ('e8','a3','a4'):
            with Model(policy,seed=7) as a:
                c=new_cursor();c['finish_interval']=997;c['next_finish']=997
                s,_=advance(a,c,1001)
                with Model.loads(a.dumps(),policy) as b:
                    t,_=advance(a,s,1000);u,_=advance(b,s,371)
                    with Model.loads(b.dumps(),policy) as d:
                        v,_=advance(d,u,629);self.assertEqual(t,v);self.assertEqual(a.dumps(),d.dumps())
                before=a.dumps()
                with a.clone() as f:
                    with self.assertRaises(RuntimeError):f.run(b'a',b'b')
                    f.reset();f.observe(255);self.assertEqual(len(f.log_probs()),256)
                self.assertEqual(before,a.dumps())
            with self.assertRaises(RuntimeError):a.state()
if __name__=='__main__':unittest.main()

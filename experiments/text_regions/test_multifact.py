"""Causal field learning, replica arbitration, isolation and exact continuation."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch
from .test_joint_retained import fixture as parent_fixture
from .composition_train import atomic_save
from .multifact_data import make_batch,labels,factors,identity,OWNERS,SOURCE,FIELD,batch_digest
from .multifact import build,ingest,segment,query,select_owner,trainable,optimizer_for,update,table_batches
from .multifact_eval import memory_tensors,evaluate
from .multifact_train import load

torch.set_num_threads(1)


def fixture(folder):
    parent=parent_fixture(folder);path=Path(folder)/'joint.pt'
    atomic_save(path,dict(config=parent.cfg.to_dict(),weights=parent.state_dict()))
    return build(path,731),parent


class MultiFact(unittest.TestCase):
    def test_identifiers_are_unique_across_cohort_namespaces(self):
        ids=[identity(split,index,20261222) for split in ('train','validation','test','independent') for index in range(4096)]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(all(len(value)==6 for value in ids))

    def test_field_labels_require_observed_marker_and_reset_clears_them(self):
        data=torch.tensor([[[61,58],[65,66],[58,61],[67,68],[65,65]]])
        reset=torch.zeros(1,5,dtype=torch.bool);reset[:,0]=True;reset[:,4]=True
        available=torch.ones_like(data,dtype=torch.bool);available[:,2,1]=False
        w,r=labels(data,reset,available)
        self.assertEqual(w['F'].tolist(),[[False,True,False,False,False]])
        self.assertEqual(w['L'].tolist(),[[False,False,False,True,False]])
        self.assertFalse(bool(w['G'].any()));self.assertEqual(r['R'].tolist(),[[-100,66,66,66,-100]])

    def test_factor_holdout_and_independent_missing_information_balance(self):
        train,val,test=map(lambda s:set(factors(s)),('train','validation','test'))
        self.assertEqual((len(train),len(val),len(test)),(32,16,16))
        self.assertFalse(train&val or train&test or val&test)
        b=make_batch('independent',0,64,733,0)
        for field in (1,2):
            for shared in range(65,69):
                for other in range(65,69):
                    rows=(b.answers[:,0]==shared)&(b.answers[:,3-field]==other)
                    self.assertEqual(sorted(b.answers[rows,field].tolist()),list(range(65,69)))
        self.assertEqual(batch_digest(b),batch_digest(make_batch('independent',0,64,733,0)))

    def test_initialization_preserves_composer_and_permutates_only_private_vocabulary(self):
        with tempfile.TemporaryDirectory() as folder:
            m,parent=fixture(folder)
            for name,value in parent.state_dict().items():
                torch.testing.assert_close(m.state_dict()[name],value,rtol=0,atol=0)
            for owner in ('L','R'):
                actual=m.independent_hats[owner].byte_embedding.weight
                expected=parent.independent_hats['F'].byte_embedding.weight.clone()
                expected[[58,61]]=expected[[61,58]].clone()
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertEqual(len(m.nodes),11)

    def test_raw_input_isolation_absence_and_loss_labels_cannot_enter(self):
        with tempfile.TemporaryDirectory() as folder,torch.no_grad():
            m,_=fixture(folder);b=make_batch('test',0,2,734,0)
            r=segment(m,b,0,12,capture=True)
            changed=b.inputs.clone();changed[:,:,0]=ord('x')
            x=segment(m,replace(b,inputs=changed),0,12,capture=True)
            for owner in ('G','R'):
                for a,z in zip(memory_tensors_owner(r['state'],owner),memory_tensors_owner(x['state'],owner)):
                    torch.testing.assert_close(a,z,rtol=0,atol=0)
            fake=replace(b,answers=torch.zeros_like(b.answers),writes={o:~v for o,v in b.writes.items()},
                         retained={o:torch.zeros_like(v) for o,v in b.retained.items()})
            z=segment(m,fake,0,12)
            for a,v in zip(r['state'].tensors(),z['state'].tensors()):torch.testing.assert_close(a,v,rtol=0,atol=0)
            absent=make_batch('test',0,2,734,0,control='left_only');s,_=ingest(m,absent)
            for owner in ('G','R'):
                self.assertFalse(bool(s.nodes[owner].clock.any()));self.assertFalse(bool(s.ports.latest[owner].valid.any()))

    def test_shared_arbitration_uses_validity_and_latest_occurrence_not_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            m,_=fixture(folder);s=m.initial_state(4)
            pubs={o:replace(s.ports.latest[o],valid=torch.ones(4,dtype=torch.bool),
                occurrence=torch.tensor([5,9,5,5]) if o=='F' else torch.tensor([8,2,5,7])) for o in OWNERS}
            pubs['G']=replace(pubs['G'],valid=torch.tensor([True,True,True,False]))
            s=replace(s,ports=s.ports.commit(pubs))
            self.assertEqual(select_owner(s,torch.zeros(4,dtype=torch.long)).tolist(),[2,0,0,0])
            self.assertEqual(select_owner(s,torch.tensor([1,2,1,2])).tolist(),[1,3,1,3])
            pubs={o:replace(p,payload=torch.randn_like(p.payload)*100) for o,p in pubs.items()}
            self.assertEqual(select_owner(replace(s,ports=s.ports.commit(pubs)),torch.zeros(4,dtype=torch.long)).tolist(),[2,0,0,0])

    def test_queries_preserve_all_memories_and_match_intervention_compute(self):
        with tempfile.TemporaryDirectory() as folder,torch.no_grad():
            m,_=fixture(folder);b=make_batch('test',0,2,735,0);s,_=ingest(m,b)
            before=[t.clone() for t in memory_tensors(s)]
            for field,tb in enumerate(table_batches(b,'test',0,735)):
                r=query(m,tb.first,tb.second,s,torch.full((2,),field,dtype=torch.long))
                for a,z in zip(before,memory_tensors(r['state'])):torch.testing.assert_close(a,z,rtol=0,atol=0)
                for control in ('no_edge','reset_memory','shuffle_memory'):
                    alt=query(m,tb.first,tb.second,s,torch.full((2,),field,dtype=torch.long),control=control)
                    self.assertEqual(alt['expert_executions'],r['expert_executions'])
                self.assertEqual(r['expert_executions'],264*2);s=r['state']

    def test_shuffle_each_field_starts_from_original_memory_and_cache_is_exact(self):
        with tempfile.TemporaryDirectory() as folder,torch.no_grad():
            m,_=fixture(folder);b=make_batch('test',0,4,738,0);state,_=ingest(m,b)
            cache={};actual=evaluate(m,'test',738,4,0,query_control='shuffle_memory',cache=cache)
            self.assertEqual(actual,evaluate(m,'test',738,4,0,query_control='shuffle_memory',cache=cache))
            for field,tb in enumerate(table_batches(b,'test',0,738)):
                r=query(m,tb.first,tb.second,state,torch.full((4,),field,dtype=torch.long),control='shuffle_memory')
                self.assertEqual(actual['fields'][field]['predictions'],r['logits'][:,-1].argmax(-1).tolist())
                nll=float(torch.nn.functional.cross_entropy(r['logits'][:,-1],tb.answers,reduction='sum'))/4
                self.assertEqual(actual['fields'][field]['nats'],nll)

    def test_private_query_gradient_reaches_only_its_selected_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            m,_=fixture(folder);b=make_batch('test',0,2,739,0)
            with torch.no_grad():state,_=ingest(m,b)
            pubs={o:replace(state.ports.latest[o],valid=torch.ones(2,dtype=torch.bool),
                payload=state.ports.latest[o].payload.detach().requires_grad_()) for o in OWNERS}
            state=replace(state,ports=state.ports.commit(pubs));tb=table_batches(b,'test',0,739)[1]
            r=query(m,tb.first,tb.second,state,torch.ones(2,dtype=torch.long))
            r['logits'].square().mean().backward()
            self.assertGreater(float(pubs['L'].payload.grad.abs().sum()),0.)
            for o in ('F','G','R'):
                grad=pubs[o].payload.grad
                self.assertTrue(grad is None or not bool(grad.any()))

    def test_update_ownership_and_exact_optimizer_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            m,_=fixture(folder);opt=optimizer_for(m)
            before={k:v.clone() for k,v in m.state_dict().items()}
            b=make_batch('train',0,2,736,0);tb=table_batches(b,'train',0,736)
            row=update(m,opt,b,tb)
            self.assertEqual(row['expert_executions'],48*b.inputs.shape[0]*b.inputs.shape[1]+3*264*len(b.answers))
            changed=[k for k,v in m.state_dict().items() if not torch.equal(v,before[k])]
            self.assertTrue(changed);self.assertTrue(all(trainable(k) for k in changed))
            for owner in OWNERS:self.assertTrue(any(k.startswith('nodes.'+owner+'.') for k in changed))
            path=Path(folder)/'resume.pt'
            atomic_save(path,dict(schema='shoebox-multifact-v1',config=m.cfg.to_dict(),weights=m.state_dict(),
                optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),protocol=dict(producer_lr=.0003,edge_lr=.001)))
            other,otheropt,_=load(path)
            self.assertEqual(update(m,opt,b,tb),update(other,otheropt,b,tb))
            for k,v in m.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)
            for k,values in opt.state_dict()['state'].items():
                for name,value in values.items():torch.testing.assert_close(value,otheropt.state_dict()['state'][k][name],rtol=0,atol=0)

    def test_live_suffix_contains_no_repeated_original_observation(self):
        for kind in ('shared_left','shared_right','left_private','right_private'):
            b=make_batch('test',0,8,737,16,kind)
            target={'shared_left':'F','shared_right':'G','left_private':'L','right_private':'R'}[kind]
            for owner in OWNERS:
                self.assertEqual(int(b.writes[owner][:,b.boundary:].sum()),8 if owner==target else 0)
            withheld=make_batch('test',0,8,737,16,kind,'withheld')
            self.assertFalse(any(bool(w[:,b.boundary:].any()) for w in withheld.writes.values()))
            self.assertTrue(torch.equal(b.inputs[:,:b.boundary],withheld.inputs[:,:b.boundary]))

    def test_invalid_addresses_fail_early(self):
        with tempfile.TemporaryDirectory() as folder:
            m,_=fixture(folder);s=m.initial_state(1)
            for address in (torch.tensor([3]),torch.tensor([-1]),torch.tensor([1.])):
                with self.assertRaises(ValueError):select_owner(s,address)


def memory_tensors_owner(state,owner):
    from .ports import tensor_list
    return tensor_list((state.nodes[owner],state.ingress[owner],state.ports.latest[owner]))


if __name__=='__main__':unittest.main()

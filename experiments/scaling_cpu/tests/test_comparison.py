import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import comparison as runner
from shoebox_runtime import Architecture,Model
from shoebox_runtime.model import atomic_json


class ComparisonTests(unittest.TestCase):
    def test_schedule_boundaries_and_coupled_cooldown(self):
        peak=.008;horizon=25_000_000
        start=runner.schedule(0,0,peak,horizon)
        warm=runner.schedule(1_000_000,0,peak,horizon)
        mature=runner.schedule(20_000_000,0,peak,horizon)
        end=runner.schedule(horizon,0,peak,horizon)
        self.assertAlmostEqual(start['lr'],peak*.1)
        self.assertEqual(warm['lr'],peak)
        self.assertAlmostEqual(mature['prototype_scale'],.1)
        self.assertAlmostEqual(end['lr'],peak*.1)
        self.assertAlmostEqual(end['prototype_scale']*end['a1_prototype_scale'],.001)
        self.assertAlmostEqual(end['lr']*end['task_route_lr_scale'],peak)
        self.assertAlmostEqual(end['lp'],end['lr']/3)

    def test_runner_resume_matches_uninterrupted_and_rejects_changes(self):
        # Real native training/checkpoint loop; small fixture changes only data
        # budget and banks, leaving the schedule and persistence code unchanged.
        c=runner.config();c.update(targets=4096,chunk_targets=512,validation_interval=2048,diagnostic_targets=128,seeds=[1,2])
        c['arms']={'control_5m':{'width':128,'experts':2,'parameters':Architecture(128,2).parameter_count}}
        data=np.frombuffer((b'The network remembers a sequence of bytes.\n'*150),dtype=np.uint8)[:4097]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);d=root/'data';d.mkdir();data.tofile(d/'train.bin');data[:257].tofile(d/'validation.bin')
            protocol={'peak_lr':.004,'seed':1,'arm':'control_5m','stable_guard':'same'}
            with patch.object(runner,'config',return_value=c),patch.object(runner,'protocol',return_value=protocol),patch.object(runner,'load_fixed',return_value=None),contextlib.redirect_stdout(io.StringIO()):
                whole=runner.run(d,root,root/'whole','control_5m',1)
                runner.run(d,root,root/'resumed','control_5m',1,max_targets=2048)
                resumed=runner.run(d,root,root/'resumed','control_5m',1,resume=True)
                self.assertEqual(whole['final_fingerprint'],resumed['final_fingerprint'])
                self.assertEqual(whole['actual_update_totals'],resumed['actual_update_totals'])
                self.assertEqual(whole['credit_totals'],resumed['credit_totals'])
                self.assertEqual(whole['status'],'completed');self.assertEqual(whole['targets'],4096)
                for x,y in zip(whole['history'],resumed['history']):
                    self.assertEqual(x['nll'],y['nll']);self.assertEqual(x['schedule'],y['schedule'])
                with self.assertRaisesRegex(ValueError,'use --resume'):runner.run(d,root,root/'resumed','control_5m',1)
                with self.assertRaisesRegex(ValueError,'multiple'):runner.run(d,root,root/'bad','control_5m',1,max_targets=100)
                with patch.object(runner,'protocol',return_value={**protocol,'stable_guard':'changed'}):
                    with self.assertRaisesRegex(ValueError,'Refusing to resume'):runner.run(d,root,root/'resumed','control_5m',1,resume=True)
                with Model.load(root/'resumed/last.npz') as m:self.assertEqual(m.observations,4096)

    def test_data_hash_and_overlap_guards(self):
        from prepare_data import verify,sha
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);splits={}
            for name,raw,identity in [('train',b'abc','a'),('validation',b'def','b')]:
                p=root/(name+'.bin');p.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
                splits[name]=dict(targets=2,sha256=digest,documents=[dict(id=identity,source_sha256=digest,start=0,bytes_used=3,segment_sha256=digest)])
            atomic_json(root/'manifest.json',dict(splits=splits));verify(root)
            (root/'train.bin').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'integrity'):verify(root)
            (root/'train.bin').write_bytes(b'abc');splits['validation']['documents'][0]['id']='a'
            atomic_json(root/'manifest.json',dict(splits=splits))
            with self.assertRaisesRegex(ValueError,'overlap'):verify(root)

    def test_paired_summary_requires_both_completed_seeds(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory)
            for arm,base in [('control_5m',2.),('width_20m',1.9)]:
                for seed in (1,2):
                    dest=root/f'{arm}_seed{seed}';dest.mkdir()
                    atomic_json(dest/'protocol.json',dict(arm=arm,seed=seed))
                    atomic_json(dest/'metrics.json',dict(targets=25_000_000,status='completed',training_seconds=10,best={'nll':base},
                        validation=[dict(targets=t,nll=base+seed*.01) for t in (23_000_000,24_000_000,25_000_000)]))
            result=runner.summarize(root)
            self.assertAlmostEqual(result['paired_deltas_against_control']['width_20m']['mean_delta_nll'],-.1)
            self.assertNotIn('experts_20m',result['complete_arm_means'])


if __name__=='__main__':unittest.main()

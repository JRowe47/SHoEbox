"""Exact long-run resume from a continued optimizer with newly shaped streams."""
from dataclasses import replace
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import torch
from . import CorticalModel
from .checkpoint import save
from .test_wide import small_wide
from .train_fineweb import optimizer_for


class CPUTrainingChecks(unittest.TestCase):
    def test_continued_weights_new_streams_and_exact_resume(self):
        torch.set_num_threads(1);torch.manual_seed(731)
        model=CorticalModel(replace(small_wide(),region_topk=1)).set_execution_backend('cpu')
        optimizer=optimizer_for(model,.001,.01)
        result=model(torch.tensor([[65]]),targets=torch.tensor([[66]]))
        result['loss'].backward();optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            save(root/'parent.pt',model,result['state'].detach(compact=True),optimizer,step=1,
                 metadata={'protocol':{'trained_targets':1}})
            (root/'train.bin').write_bytes(b'abcdefghijklmnopqr')
            (root/'val.bin').write_bytes(b'stuvwxyzAB')
            common=[sys.executable,'-m','shoebox.train_cpu','--data',str(root/'train.bin'),
                    '--validation',str(root/'val.bin'),'--targets','16','--validation-targets','8',
                    '--batch','2','--context','4','--threads','1','--sample-bytes','0',
                    '--checkpoint-every','1','--validation-every','2']
            for name,extra in [('whole',['--init',str(root/'parent.pt')]),
                               ('part',['--init',str(root/'parent.pt'),'--max-updates','1']),
                               ('resumed',['--resume',str(root/'part/last.pt')])]:
                run=subprocess.run(common+['--out',str(root/name)]+extra,capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            a=torch.load(root/'whole/last.pt',weights_only=True)
            b=torch.load(root/'resumed/last.pt',weights_only=True)
            def same(x,y):
                if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,atol=0,rtol=0)
                elif isinstance(x,dict):
                    self.assertEqual(x.keys(),y.keys())
                    for k in x:same(x[k],y[k])
                elif isinstance(x,(tuple,list)):
                    self.assertEqual(len(x),len(y))
                    for u,v in zip(x,y):same(u,v)
                else:self.assertEqual(x,y)
            for name in ('model','optimizer','state','rng'):same(a[name],b[name])
            self.assertEqual(b['metadata']['summary']['trained_targets'],16)
            self.assertEqual(b['metadata']['summary']['lifetime_trained_targets'],17)
            self.assertEqual(b['state']['batch'],2)
            self.assertEqual(b['state']['event'],8)
            bad=subprocess.run(common+['--out',str(root/'bad'),'--resume',str(root/'part/last.pt'),'--lr','.002'],
                               capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn('Resume protocol differs',bad.stderr)


if __name__=='__main__':unittest.main()

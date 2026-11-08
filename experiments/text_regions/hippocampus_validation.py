"""Independent validation cohorts in bounded CPU worker processes.

The same evaluate_values calls, batch shapes and fixed weights are used as the
serial validator. Training state/RNG never enters these inference workers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
ROOT=Path(__file__).resolve().parents[2]


def parallel_validation(run,workers=3):
    from .memory_selectivity import summarize_values
    from scripts.evaluate_memory_selectivity import paired_queries
    if workers<1: raise ValueError('Positive worker budget required.')
    keys=[*run.val_cases,'language']; workers=min(workers,len(keys))
    with tempfile.TemporaryDirectory(prefix='hippocampus_validation_') as temporary:
        root=Path(temporary); source=root/'input.pt'
        torch.save(dict(config=run.cfg.to_dict(),weights=run.model.state_dict(),
            cases=run.val_cases,validation=run.validation),source)
        def worker(index):
            target=root/f'{index}.json'
            command=[sys.executable,str(Path(__file__).resolve()),'--input',str(source),'--out',str(target),
                     '--keys',*keys[index::workers]]
            done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            if done.returncode: raise RuntimeError('Validation worker failed: '+done.stderr[-3000:])
            return json.loads(target.read_text())
        with ThreadPoolExecutor(max_workers=workers) as executor:
            portions=list(executor.map(worker,range(workers)))
        raw={key:value for part in portions for key,value in part.items()}
    language=raw.pop('language')
    cohorts={key:summarize_values(raw[key]) for key in run.val_cases}
    score=sum(cohorts[f'{mode}_{gap}']['nll'] for mode in ('older','newer') for gap in (0,32))/4
    pairs={str(gap):paired_queries(raw[f'older_{gap}'],raw[f'newer_{gap}'],run.data_cfg.values) for gap in (0,32)}
    return dict(selection_score=score,cohorts=cohorts,paired=pairs,language=language)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--keys',nargs='+',required=True);a=p.parse_args()
    sys.path.insert(0,str(ROOT)); torch.set_num_threads(1)
    from experiments.text_regions.config import GraphConfig
    from experiments.text_regions.model import RegionLanguageModel
    from experiments.text_regions.value_recall import evaluate_values
    from experiments.text_regions.graph_mixed_train import evaluate_language
    package=torch.load(a.input,map_location='cpu',weights_only=True)
    dtype=next(v.dtype for v in package['weights'].values() if v.is_floating_point())
    model=RegionLanguageModel(GraphConfig.from_dict(package['config'])).to(dtype=dtype)
    model.load_state_dict(package['weights']);model.eval()
    result={}
    for key in a.keys:
        result[key]=(evaluate_language(model,package['validation'],batch=4,context=32,chunks=2)
                     if key=='language' else evaluate_values(model,package['cases'][key],context=32,cases_per_batch=8))
    a.out.write_text(json.dumps(result,allow_nan=False))

if __name__=='__main__':main()

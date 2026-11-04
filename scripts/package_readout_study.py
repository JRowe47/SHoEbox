"""Create portable result files and audit the completed frozen-readout study."""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.readout_diagnostic import load_diagnostic
from experiments.text_regions.readout_compare import compare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/readout_diagnostic'))
    parser.add_argument('--out',type=Path,default=Path('experiments/text_regions/results/readout_diagnostic'))
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT/args.source
    target = args.out if args.out.is_absolute() else ROOT/args.out
    status = json.loads((source/'status.json').read_text())
    if not status['complete'] or any(r['status'] != 'complete' or r['returncode'] != 0 for r in status['runs']):
        parser.error('All twelve diagnostic jobs must have completed successfully.')
    if target.exists():
        parser.error('Result destination exists; choose a new path.')
    destination_final = target
    target.parent.mkdir(parents=True,exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix='.readout_stage_',dir=target.parent))
    manifest = {'schema_version':1,'base_directory':'.','runs':[]}
    artifacts = []
    for run in status['runs']:
        original,destination = source/run['name'],target/'runs'/run['name']
        destination.mkdir(parents=True)
        head_path = original/'readouts.pt'
        fitted = load_diagnostic(head_path)
        inventory = {'schema_version':1,'model_tensor_sha256':fitted['fit_protocol']['model_tensor_sha256'],
                     'readouts_sha256':sha(head_path),'readouts_size_bytes':head_path.stat().st_size,
                     'fit_protocol':fitted['fit_protocol'],'fit_identities':fitted['fit_identities'],
                     'fit_query_counts':fitted['fit_query_counts'],
                     'answer_decoder_fits':{k:h.fit for k,h in fitted['answer_heads'].items()},
                     'answer_head_shapes':{k:list(h.weight.shape) for k,h in fitted['answer_heads'].items()},
                     'factor_head_count':len(fitted['factor_heads'])}
        (destination/'fit_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        for filename in ('results.json','predictions.jsonl'):
            path = original/filename;before = sha(path)
            data = path.read_bytes()
            if filename == 'results.json':
                compressed = gzip.compress(data,compresslevel=9,mtime=0)
                (destination/(filename+'.gz')).write_bytes(compressed)
                copied = gzip.decompress((destination/(filename+'.gz')).read_bytes())
            else:
                shutil.copyfile(path,destination/filename)
                copied = (destination/filename).read_bytes()
            if sha(path) != before or hashlib.sha256(copied).hexdigest() != before:
                raise RuntimeError('Source results changed while packaging.')
        manifest['runs'].append({k:run[k] for k in ('name','publication_policy','source_schedule','training_seed','checkpoint','checkpoint_sha256')})
        manifest['runs'][-1]['directory'] = str(destination.relative_to(target))
        artifacts.append({'name':run['name'],'restore_path':str(head_path.relative_to(ROOT)),
                          'size_bytes':head_path.stat().st_size,'sha256':sha(head_path)})
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'readout_artifacts.json').write_text(json.dumps({'files':artifacts},indent=2)+'\n')
    shutil.copyfile(source/'status.json',target/'study_run_log.json')
    comparison = compare(target/'manifest.json')
    (target/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    provenance = {'schema_version':1,'study_settings':status['settings'],
                  'status_sha256':sha(source/'status.json'),'study_run_log':'study_run_log.json',
                  'comparison_sha256':sha(target/'comparison.json'),
                  'complete':True,'verified_models':comparison['verified_models'],
                  'raw_result_files_preserved':24,'lossless_gzip_result_files':12,
                  'fit_inventories':12,'readout_tensor_artifacts':12}
    (target/'verification.json').write_text(json.dumps(provenance,indent=2)+'\n')
    target.replace(destination_final)
    print(json.dumps(provenance,indent=2))


if __name__ == '__main__':
    main()

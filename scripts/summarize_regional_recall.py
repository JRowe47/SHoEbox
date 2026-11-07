"""Collect completed mixed-task runs and the ordinary graph integration check."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/regional_recall'))
    parser.add_argument('--out', type=Path,
                        default=Path('experiments/text_regions/results/regional_recall/summary.json'))
    args = parser.parse_args()
    study = dict(parent_sha256=hashlib.sha256((args.root/'parent.pt').read_bytes()).hexdigest(), arms={})
    for layout in ('distinct', 'pooled'):
        folder = args.root/layout
        read = lambda name: json.loads((folder/name).read_text())
        study['arms'][layout] = dict(protocol=read('protocol.json'), initial=read('initial.json'),
            evaluation=read('evaluation.json'),
            history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()])
        if study['arms'][layout]['protocol']['initialization']['sha256'] != study['parent_sha256']:
            raise ValueError('The paired arms do not share the declared parent.')
    a, b = (study['arms'][key] for key in ('distinct', 'pooled'))
    for key in ('protocol', 'raw_sha256', 'parameters'):
        if a['protocol'][key] != b['protocol'][key]:
            raise ValueError(f'Paired {key} differs.')
    for source in ('language', 'memory'):
        for key in ('updates', 'bytes', 'experts', 'answers'):
            if a['evaluation']['last']['totals'][source][key] != b['evaluation']['last']['totals'][source][key]:
                raise ValueError(f'Paired {source}/{key} differs.')
    study['graph'] = dict(protocol=json.loads((args.root/'graph/protocol.json').read_text()),
        history=[json.loads(line) for line in (args.root/'graph/history.jsonl').read_text().splitlines()])
    if (args.root/'graph/audit.json').exists():
        study['graph']['audit'] = json.loads((args.root/'graph/audit.json').read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(study, indent=2)+'\n')
    print(json.dumps({'out': str(args.out), 'complete': True}))


if __name__ == '__main__':
    main()

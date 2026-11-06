"""Evaluate saved selected/final checkpoints on fresh paired task streams."""
import argparse
import json
from pathlib import Path
import torch
from .task_memory import TaskMemoryConfig, TaskMemoryModel
from .task_stream import TaskStreamConfig
from .task_memory_train import final_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Positive thread count required.')
    torch.set_num_threads(args.threads)
    saved = torch.load(args.run/'last.pt', map_location='cpu', weights_only=False)
    model = TaskMemoryModel(TaskMemoryConfig.from_dict(saved['config']))
    cfg = TaskStreamConfig.from_dict(saved['data_config'])
    run = saved['args']
    result = final_evaluation(args.run, model, cfg, run['data_seed'], run['eval_count']*2,
                             run['context'], 'blocked' if run['arm'] == 'no_memory' else 'intact')
    print(json.dumps(dict(run=str(args.run), selected_step=result['selected_step'],
                          test_nll=result['test']['answer_nll'], test_accuracy=result['test']['accuracy'])))


if __name__ == '__main__':
    main()

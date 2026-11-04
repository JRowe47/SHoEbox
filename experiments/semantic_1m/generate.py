import argparse
import torch
from model import Config, SHoEGPT

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='results/demo/checkpoint.pt')
parser.add_argument('--prompt', default='ROMEO:\n')
parser.add_argument('--tokens', type=int, default=300)
parser.add_argument('--temperature', type=float, default=.8)
args = parser.parse_args()
torch.set_num_threads(2)
saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
model = SHoEGPT(Config(**saved['config'])).eval()
model.load_state_dict(saved['model'])
stoi = {ch:i for i,ch in enumerate(saved['chars'])}
tokens = torch.tensor([[stoi[ch] for ch in args.prompt]])
result = model.generate(tokens,args.tokens,args.temperature,
                        lace_enabled=not saved['protocol']['args'].get('no_lace',False))
print(''.join(saved['chars'][i] for i in result[0].tolist()))

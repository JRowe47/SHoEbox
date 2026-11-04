"""Two upstream neural byte producers, the same slot contract, persistent HAT."""
import json
import torch
from events import SlotDictionary
from model import HATSHoE

torch.set_num_threads(2);torch.manual_seed(7)
model=HATSHoE().eval();bank=SlotDictionary()
producers=[torch.nn.Linear(4,256).eval(),torch.nn.Linear(7,256).eval()]
state=None;rows=[]
with torch.no_grad():
    for tick in range(3):
        for link,producer in enumerate(producers,1):
            output=producer(torch.randn(5,producer.in_features)).argmax(-1)
            # Generic integer-byte emission. This is not a modality-specific encoder.
            version=bank.publish(link,3,bytes(output.tolist()))
            events,_=bank.events([(link,3)])
            result=model(events,state=state);state=result['state']
            rows.append({'tick':tick,'link':link,'slot':3,'version':version,
                         'input_bytes':output.tolist(),'predicted_bytes':result['logits'].argmax(-1)[0].tolist(),
                         'state_bytes':state.storage_bytes()})
print(json.dumps({'note':'API demonstration with random producer and consumer weights, not a trained task.',
                  'events':rows},indent=2))

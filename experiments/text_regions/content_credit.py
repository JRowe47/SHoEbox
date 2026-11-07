"""Exact content credit across execution chunks within one optimizer window.

Execution chunk size is independent of the gradient horizon. Capture historical
eligibility only at window entry; internal chunks carry the entire live state.
Otherwise capture() itself silently severs publication and recurrent paths,
even if the caller removes its explicit detach(). Applying internal boundary
receipts as well would double-count credit already supplied by exact autograd.

The full graph is released at the optimizer boundary. Existing conditional
traces still supply their advertised gate/payload credit from older windows.
No example replay or stochastic approximation is introduced.
"""
import torch
from .state_credit import ProducerRecorder, StateGraphEligibility


def content_window_forward(model, chunk, state, credit, segment=16):
    if not isinstance(segment, int) or segment < 1:
        raise ValueError('Execution segment must be a positive integer.')
    length = chunk.values.shape[1]
    if length < 1:
        raise ValueError('A connected training window cannot be empty.')
    # Both capture() implementations detach their input. Only the optimizer
    # window boundary may do this; ordinary execution chunks must not.
    state, boundary = credit.capture(state.detach())
    results = []
    for start in range(0, length, segment):
        with ProducerRecorder(model, credit.producer) as recorder:
            result = model(chunk.values[:, start:start+segment], state=state,
                reset=chunk.reset[:, start:start+segment], return_write_activity=True)
            producer = recorder.finish(result['state'])
        credit = StateGraphEligibility(credit.publication.advance(result['write_activity']), producer)
        results.append(result)
        state = result['state']  # Keep all content, routing and feedback paths.
    events = sum(r['external_events'] for r in results)
    auxiliary = sum((model.cfg.auxiliary_loss_weight*r['aux'] +
                     model.cfg.hat_balance_weight*r['hat_aux'])*r['external_events'] for r in results)/events
    return dict(logits=torch.cat([r['logits'] for r in results], 1),
        write_probs=torch.cat([r['write_activity']['write_probs'] for r in results], 1),
        aux=auxiliary/.001, state=state.detach(), external_events=events,
        expert_executions=sum(r['expert_executions'] for r in results)), credit, [boundary]

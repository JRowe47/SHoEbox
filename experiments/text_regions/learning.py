"""Opt-in losses and explicit post-step utility around the ordinary graph pass.

Use ``result = learner(inputs, targets, ...)``, backpropagate ``result['loss']``,
call ``optimizer.step()``, then ``learner.commit_feedback(result['utility_packet'])``.
No forward call commits associations. The packet contains detached statistics,
not retained examples or graph state. Save ``get_config()`` alongside the normal
``state_dict()`` and optimizer state; reconstruct with ``from_config()`` before
loading. Carried GraphState is still supplied and checkpointed separately.
"""
from dataclasses import asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import GraphConfig
from .model import RegionLanguageModel
from .objectives import DelayedUtility, PairedViewObjective, SourceView, ViewObjectiveConfig


class RegionLearningModel(nn.Module):
    """A default-off wrapper with a source-only baseline and paired-view losses.

    ``baseline_view`` names one independently encoded ingress (``text`` for the
    ordinary byte stream, or e.g. ``A`` for controlled views). Enabling utility
    also trains this local next-byte head. Its gain comparison uses only labeled
    occurrences where that source is available. Private targets are consumed by
    the view objective after the graph pass and never enter inference state.
    All-ignored target segments have zero language loss; targets=None preserves
    the base model's inference-only loss behavior. Evaluation without private
    labels skips an objective that requires them, so trained models still support
    ordinary label-free inference.
    ``baseline_detach_features`` trains the same baseline head and targets while
    preventing its auxiliary loss from updating the shared source encoder.
    """
    def __init__(self, model, *, view_objective=None, utility=None, baseline_enabled=False,
                 baseline_view='text', baseline_weight=.1, utility_scale=.1, baseline_detach_features=False):
        super().__init__()
        if not math.isfinite(baseline_weight) or baseline_weight <= 0:
            raise ValueError('Baseline weight must be finite and positive.')
        if not math.isfinite(utility_scale) or utility_scale < 0:
            raise ValueError('Utility scale must be finite and nonnegative.')
        if not isinstance(baseline_detach_features, bool):
            raise ValueError('Baseline feature detachment must be boolean.')
        if baseline_view not in ('text', *model.cfg.views):
            raise ValueError('Baseline must name a configured source ingress.')
        if view_objective is not None and (view_objective.width != model.cfg.width or
                                          not set(view_objective.views) <= set(model.cfg.views)):
            raise ValueError('View objective must match graph width and source views.')
        if utility is not None and (set(utility.edge_names) != set(model.connections.edges) or
                                    set(utility.node_names) != set(model.nodes) or
                                    utility.layers != 3 or utility.laces != model.cfg.laces):
            raise ValueError('Utility topology and lace dimensions must match the graph.')
        self.model, self.view_objective, self.utility = model, view_objective, utility
        self.baseline_enabled, self.baseline_view = baseline_enabled, baseline_view
        self.baseline_weight, self.utility_scale = baseline_weight, utility_scale
        self.baseline_detach_features = baseline_detach_features
        self.baseline = (nn.Linear(model.cfg.width, 256, bias=False)
                         if baseline_enabled or utility is not None else None)
        if self.baseline is not None:
            self.baseline.to(device=model.readout.weight.device, dtype=model.readout.weight.dtype)
            nn.init.normal_(self.baseline.weight, std=.02)

    @property
    def cfg(self):
        return self.model.cfg

    def initial_state(self, batch, device=None):
        return self.model.initial_state(batch, device)

    def get_config(self):
        """Primitive constructor data; weights and utility buffers use state_dict."""
        view, utility = self.view_objective, self.utility
        return {'model': self.cfg.to_dict(), 'baseline_enabled': self.baseline_enabled,
                'baseline_view': self.baseline_view, 'baseline_weight': self.baseline_weight,
                'baseline_detach_features': self.baseline_detach_features,
                'utility_scale': self.utility_scale,
                'view_objective': None if view is None else {'views': list(view.views), 'cfg': asdict(view.cfg)},
                'utility': None if utility is None else {
                    'edge_names': list(utility.edge_names), 'node_names': list(utility.node_names),
                    'laces': utility.laces, 'layers': utility.layers, 'enabled': utility.enabled,
                    'decay': utility.decay, 'gain_clip': utility.gain_clip}}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        model = RegionLanguageModel(GraphConfig.from_dict(config.pop('model')))
        view, utility = config.pop('view_objective'), config.pop('utility')
        view = None if view is None else PairedViewObjective(
            model.cfg.width, views=tuple(view['views']), cfg=ViewObjectiveConfig(**view['cfg']))
        utility = None if utility is None else DelayedUtility(**utility)
        return cls(model, view_objective=view, utility=utility, **config)

    @staticmethod
    def _merge_bias(explicit, learned):
        combined = dict(explicit or {})
        for key, value in learned.items():
            combined[key] = combined.get(key, 0.) + value
        return combined

    def forward(self, inputs, targets=None, *, private_targets=None, **kwargs):
        utility_on = self.utility is not None and self.utility.enabled
        baseline_on = self.baseline_enabled or utility_on
        learning_on = self.view_objective is not None or baseline_on
        if private_targets is not None and self.view_objective is None:
            raise ValueError('Private targets require a view objective.')
        if self.view_objective is not None and not isinstance(inputs, dict):
            raise ValueError('Paired objectives require independently supplied views.')
        if baseline_on and self.baseline_view not in (inputs if isinstance(inputs, dict) else ('text',)):
            raise ValueError('Configured baseline ingress is absent from these inputs.')

        supervised = None
        if targets is not None:
            first = next(iter(inputs.values())) if isinstance(inputs, dict) else inputs
            values = first if torch.is_tensor(first) else first.values
            if targets.dtype != torch.long or targets.shape != values.shape or targets.device != values.device:
                raise ValueError('Targets must be int64 [B,T] on the input device.')
            supervised = targets != -100
        has_labels = supervised is not None and bool(supervised.any())
        collect_utility = utility_on and self.training and has_labels
        requested_trace = kwargs.get('trace', False)
        if learning_on:
            kwargs['return_view_activity'] = True
        if collect_utility:
            kwargs['trace'] = True
        if utility_on:
            kwargs['edge_bias'] = self._merge_bias(kwargs.get('edge_bias'), self.utility.gate_bias(self.utility_scale))
            kwargs['lace_bias'] = self._merge_bias(kwargs.get('lace_bias'), self.utility.lace_bias(self.utility_scale))
        result = self.model(inputs, targets if has_labels else None, **kwargs)
        if targets is not None and not has_labels:
            # The core intentionally rejects empty CE; keep unsupervised chunks
            # usable without NaNs or inventing targets for their placeholders.
            result['ce'] = result['loss'] = result['logits'].sum()*0
        if not learning_on:
            return result

        language_loss = result['loss']
        total, view_result, baseline_logits, baseline_ce, baseline_loss, packet = language_loss, None, None, None, None, None
        run_views = self.view_objective is not None and not (
            not self.training and private_targets is None and self.view_objective.cfg.private_enabled)
        if run_views:
            sources = [SourceView.from_activity(name, result['view_activity']) for name in self.view_objective.views]
            view_result = self.view_objective(*sources, private_targets=private_targets)
            total = view_result['loss'] if total is None else total + view_result['loss']
        if baseline_on:
            source = SourceView.from_activity(self.baseline_view, result['view_activity'])
            safe = torch.where(source.available[..., None], source.payload, torch.zeros_like(source.payload))
            baseline_logits = self.baseline(safe.detach() if self.baseline_detach_features else safe)
            if targets is not None:
                used = supervised & source.available
                local_targets = targets.masked_fill(~used, -100)
                local_losses = F.cross_entropy(baseline_logits.flatten(0, 1), local_targets.flatten(),
                                              ignore_index=-100, reduction='none').reshape_as(targets)
                baseline_ce = local_losses.sum()/used.sum().clamp_min(1)
                baseline_loss = self.baseline_weight*baseline_ce
                total = baseline_loss if total is None else total + baseline_loss
                if collect_utility:
                    with torch.no_grad():
                        regional_losses = F.cross_entropy(result['logits'].detach().flatten(0, 1), targets.flatten(),
                                                         ignore_index=-100, reduction='none').reshape_as(targets)
                    packet = self.utility.make_packet(result['trace'], local_losses.detach(), regional_losses, used)
        if collect_utility and not requested_trace:
            result['trace'] = []
        result.update(loss=total, language_loss=language_loss, view_objective=view_result,
                      baseline_logits=baseline_logits, baseline_ce=baseline_ce, baseline_loss=baseline_loss,
                      utility_packet=packet)
        return result

    def commit_feedback(self, packet):
        """Explicitly apply a detached packet after the caller's optimizer step."""
        if not self.training:
            raise RuntimeError('Feedback commits are training-only.')
        if self.utility is None:
            if packet is not None:
                raise ValueError('This learner has no utility module.')
            return
        self.utility.commit_feedback(packet)

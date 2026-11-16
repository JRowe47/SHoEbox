# EXP01-B: online factorized address and value readout

This research adapter decodes a query, two keys, and two values from existing
64-dimensional L3 state. Five rank-16 projections feed one soft query/key
selection; the weighted value becomes a residual in SHoEbox's native 256-byte
hierarchical decoder. The lanes are recomputed views of existing state, not
additional persistent slots. No task parser or true record index enters inference.

The public minimal-v19 runtime is unchanged. This package uses checked hooks in
an isolated native build, preserving the production ABI and historical models.

## Learning paths

| Arm | Core updates | Added-head learning |
| --- | --- | --- |
| `native` | Native bounded credit | No added head |
| `answer` | Frozen or joint | Answer-only normalized SGD |
| `local` | Frozen or joint | Answer loss plus prefix-byte reconstruction |
| `rls` | Frozen only | Recursive least squares for addresses; SGD for values |

The local training adapter incrementally reads **already observed** record and
query bytes. At an answer target, it supplies the observed query byte, both key
bytes and both value bytes as auxiliary reconstruction labels. It never supplies
the matching record index. This is an explicit task-specific training objective;
it is not an unsupervised language objective. Scored answer payloads cannot alter
these labels. All inference entry points accept bytes or state only.

For SGD, online Welford statistics standardize L3 with a variance floor of 1e-6.
They update after each trained target and freeze in evaluation. Query/key heads
reconstruct fixed 16-coordinate byte codes; these codes appear only in training
and diagnostic inspection. Inference computes dot-product attention directly.
Value heads reconstruct their observed bytes through the shared byte tree.

The RLS control instead fits address preactivations in stationary raw-L3 plus
intercept coordinates. Its 65×65 inverse covariance starts at 1e6 I and updates
once per eligible observation. It has no forgetting, replay, or batch solve.
This control is restricted to a frozen core to keep the regression coordinates
stationary. Answer gradients continue to fit values; RLS address weights receive
only their local reconstruction targets. Inference still uses the same soft read.

All candidates add **9,535 fitted parameters**, deducted from the expert banks.
Normalizers cost 1,032 bytes. RLS adds 33,808 bytes of bounded training state.
There is no additional persistent sequence state, external binding store,
attention history, or refinement pass. Full inventories and measured costs are
written by the experiment runner.

```python
from experiments.factorized_readout.adapter import Adapter

with Adapter(core="minimal", kind="local", joint=False, rate=.16) as model:
    # Adopt a matching trained native core before fitting a frozen readout.
    checkpoint = model.dumps()
    with Adapter.loads(checkpoint) as restored:
        restored.run(b"observed", b"bserved!")
```

## Reproduce

```bash
python -m unittest experiments.factorized_readout.test_adapter -v
python -m experiments.factorized_readout.check_native
python -m experiments.factorized_readout.pilot --out /tmp/exp01b-pilot
python -m experiments.factorized_readout.pilot_rates --out /tmp/exp01b-pilot
python -m experiments.factorized_readout.pilot_rls --out /tmp/exp01b-pilot
python -m experiments.factorized_readout.conditioning --checkpoint /tmp/exp01b-pilot/rls.qr --out /tmp/exp01b-pilot/conditioning.json
python -m experiments.factorized_readout.run train --out /tmp/exp01b --workers 3
python -m experiments.factorized_readout.run audit --out /tmp/exp01b --workers 3
python -m experiments.factorized_readout.run offsets --out /tmp/exp01b --workers 3
python -m experiments.factorized_readout.positive_control --out /tmp/exp01b
# Run timing alone after all other work completes:
python -m experiments.factorized_readout.benchmark --runs /tmp/exp01b --out /tmp/exp01b/benchmark.json
python -m experiments.factorized_readout.report --runs /tmp/exp01b --pilot /tmp/exp01b-pilot
```

Use `--no-leak-check` for native tests only on hosts that prevent LeakSanitizer;
ASan and UBSan remain enabled. Checkpoint v2 includes covariance and normalization
state; this package also reads its initial v1 SGD checkpoints. Do not commit
checkpoints, compiled libraries, or generated datasets.

The formal plan is written before fitting: minimal v19, seeds 1/7/19, 500K
nominal parameters, 250K warmup bytes, 2M continuation bytes, six arms, fixed
final checkpoints, rate .16, local weight 1. New salted world folds separate
train/dev/test. All record/query presentation variants of a world share a fold.
Test results do not select the arm or training horizon.

Native whole-episode scoring feeds generated answers back into the model and
normalizes over all 256 bytes. Query counterfactuals fork the exact same frozen
record-prefix state. Selection ablations remove queries, average or reverse
weights, remove the head, or remove the source. Offset tests prepend padding
before records so record-to-answer distance stays fixed; separate full-block
delay tests hold absolute answer phase fixed. Both are frozen out-of-distribution
audits, not training augmentation.

## Primary sources

- Miller et al., [Key-Value Memory Networks](https://aclanthology.org/D16-1147/),
  EMNLP 2016, section 3: separate addressing and returned content. SHoEbox adapts
  the separation to its existing state instead of adding a parsed memory store.
- Sussillo and Abbott, [Generating Coherent Patterns of Activity from Chaotic
  Neural Networks](https://doi.org/10.1016/j.neuron.2009.07.018), Neuron 2009,
  equations 4–5: bounded online RLS readout updates. This is not a replication of
  their reservoir or feedback dynamics.
- Arora et al., [Zoology](https://arxiv.org/abs/2312.04927): independently ordered
  queries expose associative-recall weaknesses hidden by simpler copying.
- Poli et al., [Mechanistic Design and Scaling of Hybrid Architectures](https://proceedings.mlr.press/v235/poli24a.html),
  ICML 2024: capability fixtures with explicit parameter, state and compute costs.

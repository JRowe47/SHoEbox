# EXP01-D / EXP22: paired constituent readout

The selected research recipe and adapter default are **D3**: the `paired_latest` runtime with local
window supervision and short record-spacing variation during training. It uses
the existing minimal-v19 state and adds 9,900 fitted parameters within a nominal
500K total. Public `shoe_multiscale.Model` remains minimal v19.

Read the [results](../../reports/exp01d/SHoEbox_EXP01D_Report.md), original
[registration](../../reports/exp01d/PROTOCOL.md), [D2 query-policy amendment](../../reports/exp01d/AMENDMENT_D2.md),
and [D3 context-variation amendment](../../reports/exp01d/AMENDMENT_D3.md).

## Mechanism

An eight-byte retained pair contains two ordered four-byte constituents. A
five-byte record can start at four possible positions in its first constituent.
Four learned classifiers score these positions; four key and value projections
decode the same candidate. Only the two strongest candidate pairs are projected.
The separately decoded current query selects between them, and the same weights
combine their values before a residual enters the full native 256-byte decoder.

The query classifier reads existing two-byte retained pairs. `paired_latest`
chooses the latest positively recognized occurrence, using stored occurrence
timestamps. No positive candidate uses the historical confidence-minus-age
fallback. `paired` preserves D1's confidence-minus-age policy for every query.
Both policies retain their exact saved meaning.

Training-only labels annotate newly completed observed windows: query presence,
record presence, and observed key/value bytes. These labels do not solve the
requested lookup. They are generated only by `prefix_labels`; the prediction
and scoring APIs accept bytes/state only. Inference contains no task-symbol
tests, parser, external exact lookup or selected-record index.

D3 inserts independent 0–7-byte spacers after records during training. Leading
padding compensates for the spacers, keeping answer phases balanced. Long
full-block delays remain unseen. D3's new audit excludes every concrete world
in the previously opened D2 primary audit, independent of its success or failure.

## Budget and learning

| Weights | Parameters |
| --- | ---: |
| Five presence classifiers | 325 |
| Query and four phase-specific key projections | 5,200 |
| Four phase-specific value projections | 2,080 |
| Native residual output | 2,295 |
| Total | 9,900 |

Normalized online SGD fits local logistic presence and tanh byte-code
reconstruction objectives. The output receives native answer loss. No gradient
through discrete source routing is claimed; local supervision trains that path.
There are no normalization accumulators or optimizer moments. The readout scans
at most 16 query candidates and 64 record-phase candidates, retaining two pairs
in scratch. Scratch plus error storage is 2,108 bytes. It is recomputed from
existing state and is excluded from checkpoints. No sequence state is added.

The formal candidate freezes its reduced core after warmup. The `joint=True`
option permits native core learning, but its capability and production
regressions have not yet been qualified. Native local reconstruction from fixed
stable channels does not imply learned upper-summary specialization.

## Use and reproduce

```python
from experiments.constituent_readout.adapter import Adapter

with Adapter(core="minimal", parameters=500_000,
             kind="paired_latest", seed=7) as model:
    print(model.inventory())
    # train_episode supplies causal local labels; D3's runner adds short spacers.
    # Native inference only uses observe(byte), then log_probs().
```

Every run creates its registration before fitting and refuses to overwrite it.
Fit outputs/checkpoints belong outside git. Start from an empty output directory.

```bash
# D1 and D2 historical reproduction:
python -m experiments.constituent_readout.pilot --out /tmp/exp01d-pilot-d1
python -m experiments.constituent_readout.run train --out /tmp/exp01d-formal --workers 3
python -m experiments.constituent_readout.stage2 train --out /tmp/exp01d-d2 --d1 /tmp/exp01d-formal --workers 3
python -m experiments.constituent_readout.stage2 audit --out /tmp/exp01d-d2 --workers 3
python -m experiments.constituent_readout.positive_control --out /tmp/exp01d-controls

# Selected D3 recipe, fresh fold and held-out long delays:
python -m experiments.constituent_readout.stage3 pilot --out /tmp/exp01d-d3-pilot --workers 1
python -m experiments.constituent_readout.stage3 train --out /tmp/exp01d-d3 --workers 3
python -m experiments.constituent_readout.stage3 audit --out /tmp/exp01d-d3 --workers 3
python -m experiments.constituent_readout.positive_control_v3 --out /tmp/exp01d-d3-controls

# Run timing only after other fitting/evaluation/compilation has finished:
python -m experiments.constituent_readout.benchmark_v3 --runs /tmp/exp01d-d3 --out /tmp/exp01d-d3/benchmark.json
python -m experiments.constituent_readout.report --runs /tmp/exp01d-d3 --out reports/exp01d
```

The report builder also collects D1/D2 evidence from the shown historical
directories, including diagnostic JSONs preserved in the published evidence
archive. The archived stage registrations name the exact fitted/evaluated
source hashes. D1's original source snapshot is `41fdc6c`; D2's is `6ff405b`.
Final source preserves both numerical policies.

```bash
python -m unittest experiments.constituent_readout.test_adapter experiments.constituent_readout.test_latest experiments.constituent_readout.test_spacers -v
python -m experiments.constituent_readout.check_native --no-leak-check
python -m experiments.constituent_readout.check_latest
python -m multiscale.tests.run_all --no-leak-check
```

LeakSanitizer is explicitly disabled on ptrace-managed hosts; ASan and UBSan
remain enabled. Checkpoints include fitted weights, core state, policies and
training counters; exact continuation passes through unfinished boundaries.
The registered runner records RNG seed, episode-order hash and source hashes.
`resume.py` reconstructs and verifies the exact outer-stream cursor against that
order hash without retraining. The downloadable checkpoint bundle pairs each
selected model with both RNG states, remaining order and partial-episode position;
all three pass interrupted/resumed 1,000-byte continuation equivalence.

## Primary literature

- [Miller et al., EMNLP 2016](https://aclanthology.org/D16-1147/): separate
  key/value encodings with common pair attention, position-sensitive windows.
- [Arora et al., ICML 2024](https://proceedings.mlr.press/v235/arora24a.html):
  recall/throughput tradeoffs and multi-query recall evaluation.
- [Wang, Shi and Fox, JMLR 2026](https://jmlr.org/papers/v27/25-0903.html):
  separate memorization from retrieval and analyze query/key geometry.

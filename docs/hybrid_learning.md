# Phase A: future-input regional objectives

The six-region `HierarchyModel` now supports a versioned `latent` objective.
The default remains `legacy` pending the matched comparison. Phase A retains
joint next-byte backpropagation and the existing router penalty. It replaces
upper raw-byte auxiliary losses with prediction of subsequent real input
messages. It does not change forward topology, cadences, sparse dispatch,
MinKron, or published-message meaning.

| Predictor owner | Next complete target bundle | Fixed slot order |
|---|---|---|
| L2 | A1 at the next four bytes | time, channel |
| S2 | A1/B1 at the next four bytes | time, source, channel |
| L3 | L2/S2 at the next four middle boundaries | time, source, channel |

Each linear training head reads its owner's current published message. Targets
are detached. Huber loss is averaged over valid examples, times and channels,
then explicitly over source roles and across regional coefficients. A1/B1 and
Y retain the global byte objective. Local predictors still send gradients into
present upstream representations: owner-only learning is deferred to Phase B.
`parameter_owner()` provides the ownership map without detaching interfaces.

## Pairing and normalization

`hierarchy_objectives.py` builds pairs in batches from references collected at
the existing publication point. Records retain source identity, absolute byte
boundary, batch-stream identity, and document identity. Only full source spans
and full future bundles contained within one forward window are eligible.
Cross-document boundaries, padding and incomplete windows are masked. Calling
again after a reset starts new records; no prediction graph is held across an
optimizer update. Standalone callers should supply `document_ids` when a row
contains multiple documents; otherwise each row means one uninterrupted stream.
The trainer constructs these IDs from the corpus manifest. Existing recurrent
stream behavior is retained; the new mask restricts teaching pairs, not the
forward state trajectory.

Training-only EMA channel statistics are separate for A1, B1, L2 and S2.
They update from detached, valid publications; rate is 0.05 and normalization
variance has a 0.01 floor. Validation cannot update them. Their buffers are
checkpointed. Live A1/B1/L2/S2/L3 publications receive a bounded variance-floor
penalty (standard deviation target 0.1, weight 0.01) and bounded redundancy
penalty (weight 0.001). These penalties act across examples, before target
normalization; they do not balance regional or expert usage.

No target construction, auxiliary predictors, or statistics updates run when
`targets` is absent. Optional `collect_messages=True` is for explicit diagnostic
probes; ordinary sampling does not enable it. The original byte heads remain
stored for compatibility but receive neither gradients, Adam updates nor weight
decay during the latent objective. Parameter counts distinguish stored training
heads from the 2,652,512 inference parameters. Latent mode adds 83,200 training
parameters, bringing stored total parameters to 3,134,048.

## Reproduce the matched continuation

Start with `artifacts/joint_fresh256/last.pt` from the previous handoff. Its
SHA-256 is `4130cc9cb6f5674f231dd54ce3a0bc45cbaf6e028417676a0fbb8e4558f82301`.
The new handoff includes the prepared corpora, calibration seed, fixed objective
JSON, both endpoint checkpoints, full histories and audits.

```bash
python scripts/check_all.py
python scripts/verify_hybrid_cli.py
python scripts/calibrate_hierarchy_objectives.py \
  --checkpoint artifacts/joint_fresh256/last.pt \
  --data data/hybrid_calibration8k --out artifacts/hybrid_calibration

python -m shoebox.train_hierarchy \
  --data data/hybrid_matched262k --out artifacts/hybrid_legacy \
  --updates 128 --context 256 --threads 1 \
  --initialize-from artifacts/joint_fresh256/last.pt --new-stream \
  --objective legacy --validate-every 32 --checkpoint-every 16 --log-every 8

python -m shoebox.train_hierarchy \
  --data data/hybrid_matched262k --out artifacts/hybrid_latent \
  --updates 128 --context 256 --threads 1 \
  --initialize-from artifacts/hybrid_calibration/seed.pt --new-stream \
  --objective-config artifacts/hybrid_calibration/objective.json \
  --validate-every 32 --checkpoint-every 16 --log-every 8
```

Calibration observes two B8/T256 windows from separate fresh training documents
without updating shared weights. It sets each coefficient to 10% of the global
byte gradient divided by the local predictive gradient at that region's
publisher, averaged over the windows and bounded to [0.005, 0.2]. The measured
coefficients are L2=0.04311737197, S2=0.03430131270, L3=0.01301087895. They remain
fixed throughout the comparison. Normalizer calibration observations are
reported separately from optimizer target exposures.

Exact resume uses the same command with `--resume RUN/last.pt` in place of
`--initialize-from` and with `--new-stream` omitted. `--max-updates N` pauses
within the specified total budget. Exact resume rejects source, corpus bytes,
document manifest, objective, thread, or optimizer-policy changes. To add a
new budget, use `--initialize-from` and explicitly specify `--new-stream` when
changing corpus or horizon; retain the objective JSON for latent continuation.

New checkpoints save optimizer group parameter names. Old group names are
reconstructed only from their documented role/decay schemas and verified saved
parameter shapes and coverage. Adam histories are copied by these names;
new predictors start with no optimizer state. Mixed-window lifetime accounting
is retained. Changes to the objective are explicit initialization operations,
not silent exact-resume changes.

## Evaluation and follow-on work

```bash
python scripts/audit_hybrid_objectives.py --run artifacts/hybrid_legacy \
  --data data/hybrid_matched262k
python scripts/audit_hybrid_objectives.py --run artifacts/hybrid_latent \
  --data data/hybrid_matched262k
```

The audit reloads each checkpoint, reproduces the full fixed held-out score,
checks optimizer ages and finite statistics, and tests identical prior contexts
with changed future observations. Evaluation records predictive Huber/MSE,
persistence and shuffled-future controls, valid pair coverage, message standard
deviation, byte NLL/accuracy and decoder/feedback swap interventions. The final
report interprets these together; byte CE and latent Huber are not interchangeable.

Phase B's separate local/global gradients, Phase C's teaching ports and smooth
plasticity controller, and Phase D's eligibility traces remain follow-on work.
There is no new replay, imagined rollout, question/answer machinery or memory
store in this implementation.

# SHoEbox: current CPU reference design

`fresh_model()` and the low-level `Model()` now select `completion` and the full
compatible reference policy. From the repository root, import them from
`shoebox_reference`; from this package, use `runtime/runtime.py`.

The default enables the produced architecture repairs plus key-trace mixing,
DFA (gain 1), the A1 residual objective and local innovation targets. Temporal
credit uses every byte and all 64 coordinates; third-hop credit is unsampled.
Payload gain is 1, feedback activation/routing are 0.1, semantic LR multiplier
is 1, and neural lace depth is 3. Learned bottom state, task routing, EMA/revival,
retained-input credit and both EC recall channels remain enabled. Local prediction
errors now propagate to causal neural producers by default, independently of DFA.
Constituent-only expert encoding is also enabled by default.

`model.capabilities()` reports the effective feature switches and unavailable
historical experiments. `port_mode`, `port_gain` and `balance` are not implemented
in completion and nonzero values now raise an error. E05/E07/E08/E13/E14 and
R03/R04/R05 experiment variants are not shipped in this current kernel; their
old wrapper methods do not constitute implemented reference features.

`fresh_model(kernel="relays")` explicitly selects that historical comparator.
`fresh_model(profile="historical")` uses the new completion clock with the
previous training policy for controlled ablations. Explicit parameter settings
and saved checkpoint policies are preserved. All new development starts with
the reference policy unless a comparison explicitly requests otherwise.

The completed-constituent implementation works as follows. S2 through G5 now
admit non-overlapping primary-source occurrences, preserve every declared
immediate source role in bound frames, and learn when a sequence of frames is
complete. Side-source updates alone do not commit a new upper unit. Learned
class transitions and next-content forecasts support multi-frame completion;
unfamiliar classes remain explicit provisional units. The historical policy
retains its familiar-frame proposal heuristic. End-of-document draining preserves an odd primary tail.
L2 retains its sequence detector; EC and Y retain their fusion/output roles.

An upper expert executes when its unit completes. Frame admission and semantic
encoding run independently before that commitment. Y still executes every byte.
The next-content teacher consumes the next completed target exactly once.

The exported code recursively depends on child learned content and its own
content-selected regional expert. This is a separate pass through the existing
W/bias/keys banks, before the recurrent expert's execution:

```text
payload = tanh(normalized role transforms of (child.code - child.structure))
gain = payload_gain * (1 + 0.5 * tanh(regional_gate_parameter))
x = structure + gain * payload
expert, runner = top_two(keys @ masked(x))
response_gate = 2 * sigmoid((score[expert] - score[runner]) / temperature)
code = structure + 0.25 * tanh(local_adapter * structural_features
                               + gain * payload + bias[expert]
                               + response_gate * W[expert] @ x)
```

Sequence composition uses distinct temporal transforms; source binding uses the
original recipe transforms and weights. Exact IDs and structural descriptors
remain independent of neural values and surrounding recurrent context. The
payload is part of the exported code and is not added a second time to input.
Content routing uses no recurrent state, feedback, usage-count pressure or running
mean. With task router gain disabled the response gate is one. Byte atoms and
transparent identity relays keep their existing content semantics; each actual
encoding (including bound-frame admission) saves its own selected expert, input,
residual, mask and gate derivatives. Expert selection does not redefine the exact
content ID. A separate learner assigns reusable pattern classes after encoding.

### Reusable classes and predictive completion

`pattern_classes=True` is the fresh reference default. Exact IDs still bind every
child identity, source role, presence and coverage geometry. Separate monotonically
allocated class IDs identify learned prototypes, independently of expert indices.
`pattern_events()` exposes those IDs without replacing `content_events()`.

Each region has up to 256 online prototypes. The descriptor retains a normalized
64-coordinate learned child code per declared role, the source/presence schema,
bounded relative primary-clock geometry and order/overlap relations. Primary and
repeated-source roles receive full matching weight; singleton side roles receive
0.1 weight. Thus changing held context can preserve a class, while reversing a
primary/repeated-source arrangement cannot hide behind matches in other roles.
Side roles remain in classification, exported content and causal credit. Absolute
byte position and exact child IDs are excluded from class matching.

Matching uses a 0.22 radius, a 0.35 per-core-role distance guard, and online means
with an adaptation horizon capped at 64 observations. A class becomes recognized
at the configured `min_count`. Dictionary replacement is least-recently-used;
retired IDs are never reassigned, even while old events still name them. Metrics
report reuse, provisional-to-recognized promotions, unknown evaluation events,
class/dictionary turnover, sequence lengths and primary spans separately from
all-source coverage. `pattern_metrics()` also reports the dictionary population.

Adjacent events train separate class-transition counts and a per-class
next-content forecast. The forecast target bundles normalized learned child codes
by role, plus the current exported code; singleton side context is attenuated.
The score combines prior categorical confidence and prior continuous prediction
confidence equally. Neither current-transition counts nor the current target
enter the score before the decision. Clock gaps and overlaps still force a
boundary. The atomic L2 input path uses byte-class transition evidence.

Prototypes and forecasts use online statistics; neural representation learning
continues through the saved expert/semantic credit paths. Frozen evaluation and
frozen regions do not update the class dictionary. Stream reset retains fitted
classes and transition evidence while clearing event assignments. The neural
parameter count remains 1,031,952; learned class statistics add separate state.
The class sidecar is generation-checked against semantic ownership, follows pool
collection, and preserves relay class identity.

`configure_pattern_classes(True)` upgrades an older model only after
`reset_context()` or at a clean stream boundary. `snapshot_copy()`, frozen copies,
fingerprints and version-38+ checkpoints include the appropriate class state.
Versions 35-37 load with class learning disabled to reproduce their exact-key
recognition behavior. Run `src/validate_pattern_classes.py` for the matched cost,
frozen class-removal intervention and held-out generation checks. See
[the fourth-fix report](deliverables/SHoEbox_Pattern_Classes.md).

### Structural optimization stage 1

The completion runtime now builds each structural descriptor once and evaluates
its semantic adapter only in the final constituent encoder. Its input-credit
records omit unused legacy semantic edges and duplicated adapter derivatives;
bottom temporal eligibility is allocated only for A1/B1 archived producers.
All learning scopes, publication identities and reference defaults are retained.

`model.completion_storage()` reports the compact record sizes and live archive
payload alongside their historical equivalents. Checkpoint v39, composition
ABI 10 (including the explicit ABI 9 reader) and neural archive ABI 1 keep their
wire layouts through conversion at import/export. Unused legacy bytes are
canonicalized to zero, so serialized fingerprints can differ from old-runtime
fingerprints while actual continuation remains exact.

See [the stage 1 report](deliverables/SHoEbox_Structural_Optimization_Stage1.md).
Focused checks and cross-runtime continuation:

```bash
python -m unittest discover -s experiments/identity_semantics/tests \
  -p test_storage_cleanup.py -v
python experiments/identity_semantics/src/check_storage_compatibility.py \
  --baseline-runtime /path/to/8030a1e/experiments/identity_semantics/runtime \
  --output /tmp/storage_compatibility.json
```

Add `--bridge-mode eval` to verify evaluation followed by resumed training.

### Reference CPU optimizations

The default reference runtime reuses a semantic free-slot inventory, indexes
pending neural/input credit by region, tick and depth, and directly indexes
retained producer ticks. Generation-checked input norms and bounded immutable
DAG schedules avoid repeated reconstruction. Gradient banks and local-credit
map storage persist between objectives.

Content reverse work batches independent ready VJPs by expert bank and tiles
weight-gradient requests in their original per-element order. Clipping, reuse
normalization, teaching rates, reverse traversal, bank flush order and online
mutation boundaries remain unchanged. Checkpoints remain v39 / composition
ABI 10; derived caches rebuild on import and reset. Allocation reuse can change
physical slot indices and full-state fingerprints relative to the old runtime;
logits and fitted parameters must remain bitwise equal. Continuation within the
optimized runtime remains bitwise exact, including chunking and save/load.

Serial execution is the default. A persistent row team is available for ready
gradient tiles with at least 32 requests, without parallel parameter mutation:

```python
import os
cpus = sorted(os.sched_getaffinity(0))
model.configure_content_workers(2, cpus[:2])
# Return to serial execution; this is a transient execution setting.
model.configure_content_workers(1)
```

Compare actual baseline-checkpoint continuations on your CPU:

```bash
python experiments/identity_semantics/src/compare_optimized_runtime.py \
  --baseline-runtime /path/to/base/experiments/identity_semantics/runtime \
  --worker-counts 1 2 4 --repeats 3 --output /tmp/shoebox-comparison.json
python -m unittest discover -s experiments/identity_semantics/tests \
  -p test_optimizations.py -v
python experiments/identity_semantics/tests/run_optimization_sanitizer.py
```

The comparison checks logits, NLL, all fitted parameter groups and missing-credit
diagnostics. Timings exclude compilation, initialization, checkpoint I/O and
fingerprinting. See [implementation and measurements](deliverables/SHoEbox_Optimizations.md).

### Reference acceptance (sixth fix)

From the repository root:

```bash
python experiments/identity_semantics/src/run_reference_acceptance.py
```

This runs a fresh 4,096-target reference workload, the full package regression
suite and ASan/UBSan. The workload gates live neural/content/semantic/feedback
ownership; independent language/DFA and local credit deliveries; all 18 regions;
primary-clock hierarchy growth; class reuse/promotions/prediction; and paired
memory admission, recall and utility. Isolated native tests enforce the actual
causal weight changes, derivatives, unrelated-expert isolation, held producers,
role/order distinctions and asynchronous memory identity.
Hierarchy growth follows the actual repeated primary-source roles in the model's
recipe, including E4 -> G5. An unrelated wider side-source span is not a required
ordering relationship.

`acceptance_metrics()` reports primary and coverage sums/maxima and 65-bin
histograms: bin zero is zero length; bin b>0 covers [2**(b-1), 2**b-1] bytes.
EC/Y may publish partial startup fusions before their designated primary source
exists; these are explicit `unclocked_fusions` in primary bin zero, excluded
from primary-span means. Zero or invalid clocks elsewhere fail acceptance.
Local sequence histograms index frame counts 1-8 for L2-G5, excluding relays.
A multi-source binding alone counts as one temporal frame. Recognized formed
publications and inherited relay recognition are distinct. EC/Y source arity
does not count as temporal completion. These read-only observations and eligible
expert-update call counts restart on reset, snapshot load/import, or explicit
`reset_acceptance_metrics()`. They are excluded from learned state, fingerprints
and checkpoints; v39 / ABI 10 and historical continuation remain unchanged.
Existing delivery counters retain their existing stream/checkpoint scopes.

The gate evaluates content-expert removal at L3/F4/G5, G5 feedback removal,
class removal and paired recall suppression on copies of the same fitted model.
Cold/warm frozen memory is explicit. Channel controls preserve complete event
hashes, now including primary clocks. Nonzero downstream effects are required
on the fixed probes; the sign of NLL change is reported, never assumed positive.
Zero held-out recall is reported as unexercised, with a separate familiar-cue
probe required to exercise recall. Two unseen generation prompts use fixed
sampling seeds and must reproduce exactly in production and diagnostic kernels.
The source model, fitted weights/dictionaries and frozen episodic store cannot
change during evaluation.

Results are written with an explicit overall status, including failures.
`--mechanisms-only` skips regressions/sanitizers and is labelled partial, never
complete acceptance. `reference_smoke.py` uses the same counter and hierarchy
contracts; it remains a timing/mechanism smoke rather than the complete gate.
All checks are reproducible without external corpora or saved test models.

### Paired content/context episodic memory (fifth fix)

Fresh reference models use a single native episode store shared by EC-C and EC-X.
An episode owns two detached observed-input values, two normalized address keys,
its members' exact publication provenance and primary clocks, one stream epoch,
and an immutable monotonic ID. The memory contains observed constituent content;
recalled values and recurrent EC state are excluded from admission.

EC publications need not coincide. After the regional pass, the latest completed
content observation pairs with the latest completed context observation, once
per new occurrence tuple. Both sides must exist in the current stream. A held
member retains its original sequence, tick and boundaries. `reset_context()`
clears those observations and advances the stream epoch while preserving stored
episodes. `finish_document(reset=True)` drains before resetting. No episode can
pair an old-document observation with a new-document observation.

The first EC request in each byte or drain step locks a joint read using its
current input and the latest completed opposite-side input. It sets both return
vectors to members of one selected episode, with one similarity gate and ID.
Later EC requests in the same step reuse that pair. If the first request lacks
an opposite-side observation, both ports stay empty for that step. Writes occur
after both regions, before target scoring. Returned vectors still pass through
the existing separately learned EC matrices.

Joint similarity is the average of content and context key cosine similarities.
A signed joint projection supplies the sparse bucket probes. With default
`buckets=2048, ways=4`, memory holds at most 8,192 complete pairs. Stable/adaptive
bucket namespaces, novelty and outcome surprise, salience-weighted reservoir
priority and usefulness competition determine admission. Each episode occupies
one row. Reinforcement updates recency metadata without blending members or
changing identity; replacement gives the whole pair a new ID. There is no FIFO,
sliding context window or age-based expiry. Detached episodes do not own the
neural credit DAG.

Both EC ports can teach an episode's shared utility. Delayed records carry their
original returned values and ID: an evicted episode still teaches its learned
return port, but cannot update a replacement's utility. `clear_memory()` preserves
the serial allocator so old handles cannot alias a new admission. Memory-frozen
evaluation blocks admission, reinforcement, priority competition and utility
updates while query normalization and recurrent inference continue normally.

`paired_memory_metrics()` exposes admissions, replacements, joint hits,
incomplete and cached reads, utility deliveries, retired handles and current
member provenance. `evaluation_memory_info()` reports equal channel occupancy
for paired episodes. Cold copies start empty; warm copies retain complete
stored episodes and coordinate centers, with a fresh pairing stream.
`snapshot_copy()`, fingerprints and v39 checkpoints include the entire causal
pairing state. Malformed sidecars are rejected before replacing live state.

Versions 35-38 keep the independent transition stores for exact continuation.
To upgrade an old checkpoint explicitly, call `reset_context()`, `clear_memory()`
and `configure_paired_memory(True)`; policy changes require a fresh stream and
empty memory. The fitted neural weights remain available. New reference models
enable pairing by default; historical profiles keep it disabled. Run
`src/validate_paired_memory.py` for matched timing and frozen recall interventions.
See [the fifth-fix report](deliverables/SHoEbox_Paired_Episodic_Memory.md).

Each encoding saves its actual dependencies and derivatives. The pool starts
with 16,384 semantic records and is collected from live histories, pending frames
and neural eligibility. When every slot remains reachable, it grows instead of
overwriting live ancestry. Reverse traversal also grows with the saved DAG. Reverse-order credit accumulates duplicate paths before
updating the actual producer. Relays preserve the same semantic handle. The
pool never redirects expired credit to a newer producer. `completion_capacity()`
reports its current allocation; `completion_metrics()` retains separate counters
for expired handles, omitted work and capacity pruning, all expected to stay zero
in the reference path. Overflow records are included in exact checkpoints and
cleared on stream reset. The completion kernel retains 64 recent neural entries
per region as a cache and archives every displaced producer still needed by live
owners. Handles use the immutable region/execution-tick pair; the ring index is
only a lookup hint. Saved inputs, routing cues, feedback derivatives and bottom
state eligibility travel with the original record.

Collection follows held/relayed publications, pending frames, forecasts,
retained inputs and outstanding updates, using the configured neural credit
depth. Archived input records also own their semantic DAG. Released records are
reclaimed; both reset methods clear the archive. Expert revival defers replacement
of a bottom expert while its original generation is still retained.
`neural_metrics()` reports archive/reclamation counts and missing live references.
`reference_smoke.py` now fails on missing neural parents, feedback producers or
retained-input edges, and on overwritten outstanding credit.

The expert transform adds no neural parameters: **1,031,952 parameters** at 12 experts.
Version **39 / composition ABI 10** checkpoints the complete owned archive, pending
frames, gates, semantic dependencies, recent eligibility, local neural-credit
policy, content-expert policy/eligibility, class policy/state, and paired-memory
policy, immutable episodes, asynchronous observations and joint return cache.
The sidecars follow semantic record
lifetimes and pool growth; missing or mismatched content records are rejected.
Versions 35-37 remain exactly loadable, preserving exact-key recognition.
Versions 35 and 36 preserve their old content behavior;
version 35 additionally preserves its old local-teaching behavior. The explicit
v9 importer changes the ABI header and clears a formerly unused padding field,
without discarding or reinterpreting events. Enable either repair explicitly on
an older loaded model with `model.configure_local_credit(True)` and/or
`model.configure_content_experts(True)`; subsequent saves retain those choices.
Existing saved constituents retain their original encoder and credit semantics.
Version
33/34 completion checkpoints require explicit fresh-stream conversion; expired
records in an old snapshot cannot be reconstructed. Versions 31 and 32 remain
exactly loadable with their original kernels. Conversion preserves fitted
weights, dictionaries, saved training policy and preprocessing means, and resets
working/event/associative memory and pending updates. Fresh models use the current
reference policy:

```bash
python src/convert_to_completion.py old_messages_relays_or_completion.npz new_completion.npz
python -m unittest discover -s tests -v
python tests/run_completion_sanitizer.py
python src/run_completion_study.py --data /path/to/restored/data
```

Local prediction credit uses the saved issuer's actual input and routing
branches. Its bounded reverse pass finishes before neural parameter mutation,
uses `lp * upper_local_gain`, the issuer's local teaching scale, recipient local
scales and `credit_gain`, and preserves the lace reuse exponent and depth policy.
Context and feedback terms for one consumed role count as one use; transparent
relays consume no hop. Retained-input edges bypass the old expert's output.
Frozen recipients transmit derivatives while retaining their parameters.

Local work never enters the language/DFA queues. Semantic adapters receive each
computed branch once, with local teaching scales. `local_credit_metrics()` reports
per-region objective, delivery, update, relay and missing-producer counts; these
diagnostics reset on stream reset/load and do not change checkpoint state.
The smoke check requires local neural updates, feedback deliveries and zero
missing local producers. See [the local-credit report](deliverables/SHoEbox_Local_Prediction_Credit.md)
and [the preceding retention repair](deliverables/SHoEbox_Neural_Producer_Retention.md).

Content credit follows the saved output derivative through the selected transform
and routing margin into child payloads, including repeated roles. Adapter and
payload-gate derivatives include the new branch. Shared expert-bank gradients
are accumulated until the enclosing semantic/local reverse pass has computed
its derivatives, so a producer update cannot alter a later derivative in the
same pass. Frozen nodes still transmit child derivatives. Task/local teaching
scales and semantic learning rate remain in force. This uses saved online lace
eligibility and the current pre-update expert Jacobian, not unrolled BPTT.

`content_expert_metrics()` reports encodes, task/local updates, router updates and
missing producers. The `content_expert` diagnostic mutes only the constituent
transform; `expert_transform` still targets recurrent context execution.
See [the constituent-expert report](deliverables/SHoEbox_Constituent_Experts.md).

The earlier completion kernel implements the completion/content mechanisms. The matched 50,000-target
smoke comparison, including throughput and task contributions, is documented in
[the completion report](deliverables/SHoEbox_Completion_Fix.md). It does not
supersede the earlier million-target quality comparison. The default now selects
the current reference design independently of those historical performance
results. Test models and corpus bytes are excluded from delivery.

Primary clocks start at bottom byte atoms, survive singleton relays, and compose
from ordered primary occurrences. Content coverage still includes every held
side role. `completion_clocks()` exposes the primary spans separately from
`content_events()[region]['span']`. Confidence-1.0 transitions join adjacent
primary events despite overlapping side coverage; real primary gaps or overlaps
still force a boundary.

Run `python src/reference_smoke.py` for a small, same-core synthetic training and
throughput check. See [the current review](deliverables/SHoEbox_Reference_Defaults.md)
for verification and the remaining design work identified by the audit.

## Previous learned-message runtime

The `messages` kernel implements the upper-region message repair. It retains the
`relays` comparator, stable content IDs, immutable structural descriptors,
occurrence/role binding, and the separate next-content forecast record.

```python
import sys
sys.path.insert(0, "runtime")
from runtime import fresh_model, Model
model = fresh_model(seed=1, kernel="messages")
# Validation-selected controls; configure only before stream activity.
model.configure_messages(dict(payload_gain=1.0, feedback_activation=0.1,
                              feedback_routing=0.1, semantic_lr=1.0))
```

A child's exported code remains its structural descriptor plus **that child's
local semantic adapter**. The parent assembles `child.code - child.structure`
with the existing source/occurrence transforms, normalizes by the declared
recipe weights, bounds it with `tanh`, and adds it to the parent's neural
content input. The received bundle is not republished as if it came from the
parent's adapter. The parent's own learned code is the representation it
exports; its contextual expert response remains a separate channel.

Declared delayed feedback receives its own normalized, bounded branch, with
independent activation and routing gains. The existing context branch remains,
so setting all three new channel gains to zero with `semantic_lr=1` exactly reproduces `relays`,
including training. For an upper region, the active equations are:

```text
payload = tanh(normalized role transforms of child local semantic residuals)
q = content_normalizer * (local_code + payload_gain * payload + observed_byte_if_Y)
f = tanh(normalized declared delayed neural feedback)
u = 0.5 * previous_input + 0.5 * (q + legacy_context + feedback_activation * f)
routing_cue = q + feedback_routing * f
h = tanh(u + response_gate * W_selected * u + bias + optional_EC_recall)
```

Payload credit uses the saved source occurrence, role, actual adapter region,
features and derivatives. It updates that adapter directly, even when the
publication was held or transparently relayed; it never updates an expert for
an adapter-produced vector. Feedback credit uses the actual delayed neural
producer and saved gate Jacobian. The selected expert, runner, sparse mask and
routing cue are retained. The smooth response gate has a derivative; the hard
choice and mask boundaries do not. Two feedback Jacobian terms belonging to
one consumed occurrence count as one use for reuse normalization.

`message_metrics()` separates semantic deliveries/gradient norms from feedback
deliveries/missing producers. `relay_teaching_status()` and the existing lace
metrics retain their original meanings. Channel gains are bounded to [0, 1];
the semantic learning multiplier is in [0, 8]. Neural parameter count remains
1,031,936 for the 12-expert fixture. Saved eligibility increases runtime state,
not neural parameter count.

`evaluation_copy(diagnostic=True)` selects `message_checks` for a message model.
In addition to the controls below, `semantic_payload` removes only a source's
transferred semantic residual at its consumers, and `feedback_channel` removes
only its new bounded feedback branch. Both preserve content events. Routing
changes caused by these interventions are intentional, not parity failures.
A disabled diagnostic must still exactly reproduce production inference.

New checkpoints use **version 32 / composition ABI 6** and save all channel
controls and eligibility. Old version-31 relay checkpoints still load in the
unchanged relay kernel. Explicit fresh-stream conversion preserves learned
weights, dictionaries, fixed transforms and preprocessing means, while resetting
stream/teaching/eligibility and associative memory:

```bash
python src/convert_to_messages.py old_relays.npz new_messages.npz
python -m unittest discover -s tests -v
python tests/run_messages_sanitizer.py
python src/run_messages_study.py pilots
python src/run_messages_study.py full
python src/check_messages_results.py interventions
python src/check_messages_results.py timing
```

The study uses six validation-only candidates on seed 3, then two fresh seeds
with one million byte targets per runtime. Test targets are disjoint from the
validation segment. Timing alternates both runtimes on one CPU core after
warmup. [Implementation and results](deliverables/SHoEbox_Learned_Message_Fix.md).
Test checkpoints and intermediate runs are excluded from Git.

---

## Previous contribution and relay runtime

This package implements multi-source binding, consistent identity/vector semantics, preserved teaching/recognition through relays, and separated regional contribution checks. `relays` remains the unchanged comparison kernel. The evaluation-only `contributions` kernel compiles optional interventions and event tracing into the same source; production execution compiles every intervention hook out.

## Regional contribution checks

Use a new diagnostic copy for each condition. Controls belong to individual models, not shared global variables. Diagnostic models reject training and checkpoint serialization. A disabled diagnostic must exactly match ordinary evaluation before its measurements are accepted.

```python
model = Model.load("my_relay_checkpoint.npz")
control = model.evaluation_copy(memory="cold_adaptive", diagnostic=True)
probe = model.evaluation_copy(memory="cold_adaptive", diagnostic=True)
probe.configure_contribution("neural_output", "L3")
baseline_nll = control.run(inputs, targets)
delta_nll = probe.run(inputs, targets) - baseline_nll
print(probe.contribution_metrics())
control.close()
probe.close()
model.close()
```

| Intervention | What changes |
| --- | --- |
| `neural_output` | Zero a region's neural publication after its private memory update. Singleton relays keep their incoming context; content descriptors/codes and event metadata are preserved. This publication includes the input skip and EC recall as well as the expert response. |
| `expert_transform` | Remove the selected region's learned expert matrix/bias term before the activation. Keep its input skip, EC recall, routing decisions and relay handling. Subsequent recurrent consequences remain live. |
| `content_descriptor` | Zero the selected region's outgoing descriptor and code on every publication, including relays. Preserve IDs, event timing, recognition, context and provenance. Parent ID anchors remain, so this is a descriptor-path intervention, not deletion of all symbol information. |
| `relay_context` | Zero only singleton relay context at the selected receiving region. Preserve the relayed descriptor/code and intrinsic structure; locally executed neural publications are untouched. |
| `retained_input` | Suppress the previous-input carry term at each selected regional execution while retaining the current input's original coefficient. Bottom learned-memory reads remain separate. |
| `semantic_residual` | Suppress the learned symbol residual when a local code is encoded. Preserve its structural descriptor. Available at L2 and above; transparent relays retain inherited codes. |
| `whole_source` | Remove a selected source from constituent/context consumers and, for A1/Y, its direct decoder feature block. This control deliberately changes the event stream and combines transport with computation. |

Positive delta NLL means removing that path damaged prediction. These conditional interventions are not additive percentages of learned knowledge. Only `expert_transform` directly isolates the expert matrix/bias contribution; `neural_output` measures dependence on the complete published neural response.

Every publication is hashed over its ID, span, occurrence, intrinsic structure, recognition, source roles and actual producer. All interventions other than `whole_source` must preserve the complete content event sequence and execution counts. In the message kernel, neural interventions may intentionally change routing. Tests also cover channel isolation, memory writes, independent controls, chunking, and exact disabled parity.

Memory policy is explicit: `cold_adaptive` starts empty and admits causal associations from the test prefix; `cold_frozen` starts empty and blocks writes; `warm_frozen` retains the checkpoint store and blocks writes; `warm_adaptive` retains it and admits new associations. All use fresh recurrent/event state and frozen neural parameters/dictionaries in eval. Warm policies also preserve the store's coordinate centers and clock for meaningful timestamps. Query-normalization state still evolves during inference. The historical `frozen_copy()` remains equivalent to `cold_adaptive`.

```bash
python src/run_contribution_checks.py --workers 6
python src/contribution_credit_audit.py
python tests/run_contribution_sanitizer.py
python src/contribution_performance.py --baseline-runtime /path/to/f9eeb8a/runtime
```

The checks use the two existing relay-repair 1M-parameter checkpoints, each trained for one million byte targets, and all 78,771 reserved test targets. Restore or reproduce those test checkpoints with the preceding relay study when needed. Compact results, the [report](deliverables/SHoEbox_Contribution_Checks_Report.md), and its PDF are published; test checkpoints and intermediate evaluations are excluded.

## Run

Use Python 3.10+, NumPy, and g++ with C++17. The native module compiles automatically.

```python
import sys
sys.path.insert(0, "runtime")
from runtime import fresh_model, Model
m = fresh_model(kernel="relays")
# Exact continuation of a repaired checkpoint you save:
# m = Model.load("my_relay_checkpoint.npz")
```

Available kernels: `completion` (current default reference), `messages` (learned message repair), `relays` (relay comparator), `semantics` (previous learned representation), `semantics_fixed` (new structural encoding with its learned residual held at zero), `binding` (prior repair), `composition` (earlier comparator), and `native` (legacy numerical kernel). The default is `completion`; select earlier kernels explicitly for comparisons.

## Contract

- Exact 128-bit IDs bind namespace, kind, source/recipe role, child identity, order, presence and relative geometry. Absolute offsets and neural context are excluded.
- Each event carries an immutable structural descriptor and a separate learned semantic code. Parent descriptors use child structural descriptors, preventing held embeddings from accidentally relabeling a symbol.
- Sixteen regional 64-by-8 learned matrices provide bounded semantic residuals: 8,192 extra coefficients, 1,031,936 total. The local exact ID remains independent of expert choice and dictionary eviction.
- Semantic updates use saved publication features and derivatives, never the arriving target's features. Local prediction targets are detached.
- Relays preserve intrinsic structure, recognition, confidence and actual-producer provenance. `local_arity` and `relayed` distinguish the receiving action from the inherited content.
- A separate forecast record teaches the saved producer on the next observed regional content input, including a completed singleton. Idle ticks do not consume forecasts. Evaluation/frozen targets retire forecasts without weight updates.
- `relay_teaching_status()` reports scheduling, consumption, expiry and pending provenance; `relay_event_status()` reports local relay metadata, confidence and origin.

## Checkpoints

Repaired checkpoints use version 31 / composition ABI 5. `Model.load()` resumes pending forecasts and all stream state exactly. Older kernels retain their original checkpoint formats.

`python src/convert_to_relays.py OLD.npz NEW.npz` explicitly converts version-30 learned semantics at a fresh stream boundary. It preserves neural/semantic weights, dictionaries, fixed transports, policies and fitted preprocessing means. It resets recurrent context, pending forecasts, eligibility, online adaptation state and hippocampal associations. The old event blob is rejected rather than reinterpreted. The destination must be new.

For older version-29 binding checkpoints, `src/convert_to_semantics.py` remains available as the prior conversion step.

## Verification and reproduction

```bash
python -m unittest discover -s tests -v
python tests/run_relays_sanitizer.py
python src/run_relays_study.py pilots
python src/run_relays_study.py full
python src/relays_performance.py
python src/report_relays.py
```

The relay suite is discovered by the existing `scripts/check_all.py` integration. The study uses the exact Tiny Shakespeare data and validation-only rate selection on seed 3. Seeds 1 and 2 each train for one million next-byte targets, then evaluate on a disjoint 78,771-target test segment. The [relay report](deliverables/SHoEbox_Relay_Preservation_Report.md) and [results](deliverables/SHoEbox_Relay_Preservation_Results.json) contain the measured loss, throughput and forecast accounting.

Test models and intermediate relay runs are excluded from Git. Small results and final reports are included. `SOURCE_MANIFEST.json` identifies the current source; historical comparison records remain in `provenance/` and the earlier reports. Earlier study scripts remain available for reproduction.

## Restore data and trained checkpoints

The Git checkout contains the tested runtime and fixed initialization fixtures. Raw corpus bytes and trained checkpoints stay in run artifacts, following repository policy. Restore all nine exact data/checkpoint files from `SHoEbox_Identity_Semantics_Source_and_Checkpoints.zip`:

```bash
python src/restore_artifacts.py --archive /path/to/SHoEbox_Identity_Semantics_Source_and_Checkpoints.zip
```

Every artifact is verified against `ARTIFACT_MANIFEST.json` before installation. Existing files with different bytes are rejected. To reproduce training without the checkpoint archive, restore the exact corpus split with `python src/restore_artifacts.py --download-data`; `DATA_MANIFEST.json` pins the source hash and split.

The [multi-source binding report](provenance/binding_study/SHoEbox_Multisource_Binding_Report.md) records the first repair. The [identity/vector report](deliverables/SHoEbox_Identity_Semantics_Report.md) records the second. The [relay report](deliverables/SHoEbox_Relay_Preservation_Report.md) records the third. These are frozen study records; their statements about publication describe when the studies finished.

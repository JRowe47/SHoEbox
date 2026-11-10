# Sources and notices

## Semantic attention inspiration

The semantic-lace experiments were developed from the idea of local semantic addressing and cross-layer relationships. A conceptual reference is [Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention](https://arxiv.org/abs/2610.42147), by Ales Kubicek and colleagues. SHoE uses stable prototype groups and lace factors; this repository does not reproduce that paper's clustering implementation or benchmarks.

## minKron / KronLSTM references

The recurrent implementation preserves the compact minKron recurrence developed in the preceding Super-1 experiment. The associated project lineage uses [JRowe47/Super-1](https://github.com/JRowe47/Super-1). The retained original source is `experiments/hat_minkron/references/compact_kron_original.py`; its full historical dependencies are not part of the runnable SHoE model.

The independent KronLSTM reference was obtained from the authors' flash-linear-attention implementation. The retained reference source identifies [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) and its contributors. `author_recurrent.py` extracts its tokenwise path for independent correctness checks; `author_naive.py` preserves the broader original reference.

The original MIT notices are retained verbatim:

- [KronLSTM license](experiments/hat_minkron/references/KronLSTM_LICENSE.txt): Songlin Yang, Yu Zhang, Zhiyuan Li, and NXAI GmbH.
- [Super-1 license](experiments/hat_minkron/references/Super-1_LICENSE.txt): Rohan Arun.

The current runtime imports the small extracted tokenwise reference only for tests. It does not require the original Super-1 hierarchy, FLA package or fused kernels.

## Dataset

Tiny Shakespeare is obtained from [Andrej Karpathy's char-rnn dataset](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). `scripts/prepare_data.py` retrieves it and verifies the exact byte hash used by the recorded experiments. Dataset copies are not included in git.

The scaling comparison uses [HuggingFaceFW/FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), pinned to revision `9bb295ddab0e05d785b879661af7260fed5140fc`. The repository stores document IDs, exact content hashes and source coordinates; `experiments/scaling_cpu/restore_data.py` retrieves the selected documents from their source. Raw corpus copies and trained weights are not git assets.

## Experiment provenance

Each experiment's `PROVENANCE.json` retains its source lineage and recorded hashes. `SOURCE_MANIFEST.json` records the public package's preserved model, training and verification source. Historical archive names identify the preceding experiments; those archives and their large training checkpoints are not repository assets.


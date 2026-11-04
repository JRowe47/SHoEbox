# Recurrent reference source

| File | Purpose |
|---|---|
| `author_recurrent.py` | Extracted authors' tokenwise KronLSTM recurrence, imported by `verify.py` for independent output, state and gradient comparisons |
| `author_naive.py` | Preserved broader author reference, including its chunkwise form; not imported by the runnable demo |
| `compact_kron_original.py` | Preserved compact minKron implementation from the preceding Super-1 experiment; not imported by this demo |
| `KronLSTM_LICENSE.txt` | Original MIT notice for the author reference |
| `Super-1_LICENSE.txt` | Original MIT notice retained from the preceding Super-1 source |

The runtime recurrence is in `../kron.py`. The historical source files retain their original imports and comments for provenance. Their external dependencies are not needed to run this repository's model or test suites. See [third-party notices](../../../THIRD_PARTY_NOTICES.md) for source links and attribution.

# Design

For where the code lives, see [architecture.md](architecture.md). For
results, see [experiments.md](experiments.md).

## Motivation

AML (anti-money-laundering) detection needs to run on transaction data that
is sensitive and often spans multiple institutions. This project builds an
AML classifier that can be evaluated on encrypted transactions, so the party
running inference never needs to see plaintext data to produce a
prediction. Fully Homomorphic Encryption (FHE) is what makes that possible:
it lets a party evaluate a function on ciphertext and get back an encrypted
result, without ever holding the decryption key.

## The pipeline

Raw IBM AML transaction records go through four stages in a single run.

**1. Staging.** Kaggle's raw CSV and TXT files are loaded into Spark and
staged into partitioned parquet.

**2. Feature generation.** Laundering shows up as relational structure
across accounts (fan-out, fan-in, cyclic flows), not as attributes of a
single transaction viewed in isolation. This stage builds a transaction
graph, runs Leiden community detection over it, and derives account- and
flow-level features from the graph rather than treating each row
independently. Output: train/valid/test feature parquets.

**3. Train and evaluate two models.** Plain XGBoost is trained on the
feature parquets as an unconstrained reference, the best accuracy achievable
with no encryption constraints. Concrete ML trains the same model family
restricted to quantized, low-bit integer arithmetic, because that is the
only form of computation an FHE circuit can execute. For each model, the
decision threshold that maximizes F1 is selected on the validation split,
then applied unchanged to the test split, so the reported F1 is an honest
estimate rather than a threshold fit to the same data it is evaluated on.
Comparing the two models' F1 measures the accuracy cost of making the model
FHE-compatible.

**4. FHE correctness validation.** Once the Concrete ML model is trained,
its FHE circuit is compiled, then checked two ways. **Simulate**
(`fhe="simulate"`) is a software-only check that quantized clear inference
agrees with homomorphic evaluation; it is cheap enough to run on a large
sample. **Execute** (`fhe="execute"`) performs real key generation,
encryption, homomorphic evaluation, and decryption, the actual cryptographic
proof, but several orders of magnitude slower, so it only runs on a small
representative sample. Simulate gives breadth, execute gives proof, and
neither alone is both cheap and conclusive.

## Choosing quantization bit width

FHE circuit cost (bootstrap key size, circuit complexity, execution
latency) grows sharply with numeric precision, and quantization bit width
sets that precision directly. Finding the lowest bit width that still
preserves useful accuracy means comparing many trained models against each
other, not something decided within a single run. The hyperparameter sweep
(`sweep.py`) repeats stage 3 above across a grid of bit widths, tree depths,
tree counts, and seeds; `docs/experiments.md` reports the results and the
recommended setting that came out of it.

## Privacy mechanism

FHE schemes, including the TFHE backend Concrete ML uses, separate the
private decryption key from the public evaluation key needed to run the
circuit. A party holding only the evaluation key can compute on ciphertexts
and produce an encrypted result, but cannot decrypt anything. This project
relies on Concrete ML's built-in handling of that key separation; it does
not implement a separate key-management layer of its own. The "execute
matches clear" results in `docs/experiments.md` demonstrate that the
encrypted computation path reproduces the model's decisions, which is the
property that makes privacy-preserving inference viable in the first place.

## Constraints

- **Synthetic dataset.** IBM AML HI-Small is a public benchmark dataset, not
  real institutional transaction data. Results describe controlled technical
  validation, not production deployment performance.
- **Single dataset variant.** Only HI-Small is supported (see
  [architecture.md](architecture.md)); other IBM AML variants are not.
- **FHE execute throughput.** Actual encrypted execution runs at roughly
  4 seconds per row in the recommended setting. Serial execution over a
  full million-row test set would take weeks; production use needs
  parallelism or a narrower operational scope.

## References

**Dataset**: IBM Transactions for Anti-Money Laundering (AML), Kaggle:
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Introduced as a public benchmark for reproducible AML model evaluation by:

1. E. Altman, J. Blanuša, and L. von Niederhäusern.
   ["Realistic Synthetic Financial Transactions for Anti-Money Laundering
   Models"](https://arxiv.org/abs/2306.16424). *Advances in Neural
   Information Processing Systems (NeurIPS)*, 2023.
2. B. Egressy, L. von Niederhäusern, J. Blanuša, E. Altman, R. Wattenhofer,
   and K. Atasu.
   ["Provably Powerful Graph Neural Networks for Directed
   Multigraphs"](https://arxiv.org/abs/2306.11586). *Proceedings of the
   AAAI Conference on Artificial Intelligence*, 2024.

This repository uses only the AML detection use case from that research; it
does not implement the directed-multigraph GNN architecture from [2].

**Key tooling**:

- [ExStraQT](https://github.com/mhaseebtariq/exstraqt): the graph-based
  feature extraction methodology the `_legacy_*` feature generation modules
  are built on (see [architecture.md](architecture.md)).
- [Concrete ML](https://github.com/zama-ai/concrete-ml) (Zama): FHE-compatible
  model training and inference.
- [leidenalg](https://github.com/vtraag/leidenalg) (with `python-igraph`):
  Leiden community detection used in feature generation.
- [XGBoost](https://github.com/dmlc/xgboost): the model family trained both
  as the plain reference and as the FHE-compatible classifier.

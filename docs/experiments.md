# Experiments

Summary of how the model was evaluated, what the numbers are, and what they imply
for configuration choices. This is a development reference, not a full report.
Raw run artifacts and per-run sweep tables are not committed; reproduce them
with the commands in [usage.md](usage.md) if you need the full detail.

## Dataset

IBM AML HI-Small (synthetic, public Kaggle dataset):
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml.
See [design.md](design.md#references) for the academic citation.

| Split | Rows      | Illicit | Illicit rate |
|-------|-----------|---------|--------------|
| Train | 3,043,615 | 2,298   | 0.0755%      |
| Valid | 1,014,538 | 1,081   | 0.1066%      |
| Test  | 1,014,540 | 1,798   | 0.1772%      |

793 model input features after one-hot encoding. Training uses all illicit rows
plus 3,000,000 sampled normal rows (severe class imbalance is handled by
`scale_pos_weight` and threshold tuning, not resampling to balance).

## Method

For each hyperparameter combination: train plain XGBoost (reference) and a
quantized Concrete ML XGBoost model, tune the decision threshold on the
validation set, then evaluate F1/precision/recall with clear-mode inference
on the held-out test set (the reported F1 numbers below use this path, not
FHE simulate or execute). Two additional checks confirm FHE correctness on
top of that:

- **Simulate** (`fhe="simulate"`): fast software check that quantized clear
  inference matches homomorphic evaluation, run on a random subsample of the
  test set (`simulate_rows` in the config, 100,000 rows for the recommended
  setting).
- **Execute** (`fhe="execute"`): actual encrypted inference. Real key
  generation, encryption, homomorphic evaluation, and decryption, run on a
  smaller random subsample (`execute_rows` in the config) because it is
  orders of magnitude slower than simulate.

Swept parameters: FHE quantization bit width (2/3/4-bit), tree max depth
(3/4/5), tree count (20/50/100), and random seed (robustness check).

## Key Results

Recommended operating point: **3-bit quantization, depth 5, 50 trees**.

| Metric | Value |
|---|---|
| Test F1 | 0.7356 (precision 0.840, recall 0.655) |
| AUC-PR | 0.727 |
| 95% bootstrap CI on F1 | [0.719, 0.752] |
| FHE execute vs. clear match | 3,000/3,000 rows (0 mismatches) |
| FHE execute latency | ~4s/row |
| Bootstrap key size | 164 MB |

Config: `configs/sweep/3b5d50t_s1.toml`. Its default `execute_rows` is 20; the
3,000-row execute match above used an expanded sample to build higher
confidence in FHE correctness before committing to this setting.

## Implications

- **2-bit quantization is unusable.** F1 collapses to near-zero (0.01 to 0.30)
  across all tested depths. The quantization error destroys the classifier.
- **3-bit is the practical sweet spot.** Comparable or better F1 than 4-bit,
  at roughly 1/5 the bootstrap key size (164 MB vs. ~913 MB) and 2 to 3x
  faster execute. 4-bit is not worth its cost.
- **Depth drives accuracy, bit width drives FHE cost.** Within 3-bit, F1 rises
  from 0.69 (depth 3) to 0.72 to 0.74 (depth 5). Bootstrap key size depends
  only on bit width, not depth.
- **Tree count has a ceiling.** 100 trees at this training scale ran out of
  memory during Concrete ML training; 50 trees is the practical maximum.
- **Seed 0 is anomalous.** Same config, F1 drops to 0.64 with near-zero
  default-threshold F1 (miscalibrated). Seeds 1, 2, 3, and 42 are consistent
  (0.72 to 0.74). Don't read anything into seed 0 beyond "occasionally
  quantized training miscalibrates"; use seed 1 (or any of 1/2/3/42) in
  practice.
- **FHE execute is orders of magnitude slower than simulate** (~4s/row vs.
  ~3ms/row), which is why correctness is proven on a representative sample
  rather than the full test set. At ~4s/row, encrypting the full million-row
  test set serially would take weeks. Real deployment needs parallelism or a
  narrower use case, not a bigger single-threaded run.

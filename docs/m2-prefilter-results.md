# M2 — Feasibility of Pre-Filtering Low-Risk Transactions: Results

A cheap **bank-side plaintext** filter that discards clearly-normal transactions
*before* they enter the expensive FHE-XGBoost path, so only the suspicious minority
pays the ~4.3 s / ~839 KB per-transaction FHE cost (see P2 results).

- **Model:** Logistic Regression (`StandardScaler` + `class_weight="balanced"`), trained on the **train split only**.
- **Scripts:** `scripts/filter_LR.py` (base filter + features), `scripts/filter_LR_chainsplit.py` (chain-aware split comparison), `scripts/filter_LR_sender.py` (paying-bank-only variant).
- **Data:** IBM HI-Small (`data/HI-Small_Trans.csv`), 5,078,345 transactions, 5,177 illicit (0.10%).
- **Note:** the feature set is a first pass (for the leakage/feasibility question); it is not final and can be refined. (The mistaken `amt_diff` / `amt_ratio` features were dropped after inspection — near-constant on same-currency transactions; confirmed no effect on results.) Amounts are converted to **USD** in `load()` — the raw data is un-converted and ~1% of transactions are cross-currency; matching the main pipeline's `currency_rates` lifts the improved model's PR-AUC ~0.41 → 0.47.

> Status: feasibility pass on HI-Small. Numbers below use the transaction-level
> **OR** combine (keep if either the sender's or receiver's bank view is suspicious);
> see "Caveats" for the realistic paying-bank-only variant still to run.

## Method

1. **Local observations.** Each transaction is expanded into 2 rows, one per bank's
   local view (sender-side / receiver-side; internal transfers → debit / credit),
   so every feature is scoped to what one bank sees for one of its own accounts.
2. **Features (all bank-local, no global graph):** hour, day-of-week, amount
   diff/ratio, log amounts, same-currency, is-sender, internal-transfer,
   self-account, own-account causal history (prev count, time-since-last), bilateral
   causal history with the specific counterparty, and Payment-Format one-hots.
   Causal aggregates are time-sorted and look only backward (no leakage).
3. **Chronological split** at the 70th percentile of timestamp, grouped by
   `transaction_id` so both observations of a transaction land on the same side.
4. **Decision:** keep transactions with predicted proba ≥ threshold (suspicious),
   drop the rest (confident-clean). Threshold is chosen to retain a target % of
   illicit ("illicit retention" = recall of the positive class).

## Reproduce

```bash
# base filter (row-level + transaction-level OR), original timestamp split
python scripts/filter_LR.py data/HI-Small_Trans.csv

# chain-aware split vs timestamp split (leakage check)
python scripts/filter_LR_chainsplit.py data/HI-Small_Trans.csv data/HI-Small_Patterns.txt

# paying-bank-only variant, both splits (realistic deployment)
python scripts/filter_LR_sender.py data/HI-Small_Trans.csv data/HI-Small_Patterns.txt
```

## Results

Test split: 1,523,696 transactions, 2,321 illicit. **Transaction-level ROC-AUC ≈ 0.935.**

**PR-AUC (average precision) ≈ 0.010–0.014** (per config below) — low in absolute
terms but **~8–10× the random baseline** (positive rate ≈ 0.001). ROC-AUC is
optimistic under 0.1% prevalence (FPR stays tiny because negatives dominate), so
PR-AUC is the more honest summary. A pre-filter, however, is judged by the
retention/drop curve at a **high-recall** operating point, not by precision: at 95%
retention the kept ~18% is still only ~0.6% illicit, which is fine — those
clean-but-suspicious transactions simply proceed to FHE. The **drop % @ retention is
measured directly on the test set**, so it does not depend on either AUC.

| PR-AUC (transaction-level) | OR (both banks) | Sender-only (paying bank) |
|-----|-----|-----|
| Timestamp | 0.0142 | 0.0128 |
| Chain-aware | 0.0118 | 0.0103 |

**Precision at each retention level (sender-only, chain-aware):**

| Recall (retain illicit) | Precision | Kept | Illicit / kept |
|-------------------------|-----------|------|----------------|
| 0.99 | 0.19% | 931,038 | 1,730 |
| 0.95 | 0.61% | 270,192 | 1,660 |
| 0.90 | 1.18% | 133,600 | 1,573 |
| 0.80 | **1.55%** | 90,386 | 1,398 |
| 0.50 | 1.33% | 65,669 | 874 |
| 0.20 | 0.65% | 53,958 | 350 |

Precision stays in 0.2–1.6% (the imbalance ceiling) and is **non-monotonic** — it
*falls* at the very top (0.65% at recall 0.20 vs 1.55% at 0.80). The model's
most-confident predictions are contaminated by a broad categorical signal (Payment
Format + same currency) that also flags many clean transactions. Harmless for a
high-recall filter, but it identifies the **feature set** as the lever for improvement
(see Caveats).

**Share of transactions dropped vs. illicit retention (transaction-level):**

| Illicit retained | Transactions dropped | Illicit missed |
|------------------|----------------------|----------------|
| 99.9% | 12.9% | ~2 |
| 99.0% | 40.6% | ~23 |
| 98.0% | 58.6% | ~46 |
| **95.0% (recommended)** | **86.1%** | 116 / 2,321 |
| 90.0% | 91.7% | ~232 |

**Recommended operating point (retain 95% illicit):** threshold 0.468 → drop
**1,312,250 / 1,523,696 (86.1%)**, keep 211,446, miss 116 illicit (retain 2,205 / 2,321).

### Leakage check: chain-aware split (the honest number)

Laundering is organised as **370 chains** (`HI-Small_Patterns.txt`) that span
multiple days, so a plain timestamp cut splits **27% of chains** across train/test
— the model trains on a chain's early transactions and is tested on its
continuation (group/campaign leakage). Re-running with a **chain-aware split**
(each chain assigned whole to train or test by its first-transaction time; 100% of
3,209 pattern transactions matched back to the CSV) removes this:

| Split | Test AUC | Drop % @95% retention |
|-------|----------|-----------------------|
| Timestamp (chains straddle) | 0.9375 | 86.1% |
| **Chain-aware (leakage-free)** | 0.9361 | **81.8%** |
| Gap (optimism) | −0.0014 | **−4.3 pp** |

**The leakage is real but small (4.3 pp).** Because the filter uses only generic
local features (no account IDs), it cannot win by memorising specific chains, so
the honest feasibility number is **~82% dropped at 95% illicit retention** — still
strong. Recommendation: adopt the chain-aware split as the shared discipline across
the filter, the main model, and P1's ablation (all currently use the same
timestamp-split family, which has the same exposure).

### Paying-bank-only (realistic deployment)

The table above uses the transaction-level **OR** (max over the sender's and
receiver's bank views). In reality the **paying bank filters its own outgoing
transactions alone** — it does not have the receiving bank's score. Restricting to
the paying observation of each transaction (`role ∈ {sender, internal_debit}`, one
row per transaction) gives the realistic number:

| Split | OR (both banks) | **Sender-only (paying bank)** |
|-------|-----------------|-------------------------------|
| Timestamp | 86.1% | 87.2% |
| **Chain-aware (honest)** | 81.8% | **82.3%** |

**Sender-only is as good as OR — marginally better** (82.3% vs 81.8% at the honest
operating point). The laundering signal sits mostly on the paying side (fan-out /
structuring), and the receiver view mainly adds noise that OR's max propagates.
**Deployment implication:** no cross-bank coordination of filter scores is needed —
each paying bank filters independently, locally, in plaintext.

### Headline number

**A paying-bank-side plaintext LR retains 95% of illicit while dropping ~82% of
transactions** (chain-aware split, sender-only). Only ~18% reach the FHE pipeline.

Top feature contributions (|coef|): Payment-Format (Reinvestment −2.47, Wire −1.95,
ACH +1.49), same-currency (+1.14), counterparty-prev-count (−0.68), own-account-prev-count (+0.46).

## Savings conversion (M2 deliverable, using P2's measured sizes)

Per FHE transaction (from `docs/p2a-lifecycle-results.md`): input ciphertext
19,296 B + output ciphertext 819,872 B = **839,168 B (~839 KB)** + **~4.31 s** server
compute. (Evaluation keys ~44 MB are one-time, not per-transaction — unaffected by dropping.)

Dropping a transaction removes its **entire** per-transaction cost, so savings scale
directly with drop rate. At the 95%-retention point (drop 86.1%):

| Lever | What it shrinks | Effect |
|-------|-----------------|--------|
| **M2 pre-filter (drop 86%)** | whole per-txn cost (input + output ct + compute) | **~86%** of FHE ciphertext storage, transfer, and compute |
| P1 dimension reduction | input ciphertext only (19.3 KB = **2.3%** of the 839 KB per-txn total) | shrinks ≤2.3% of per-txn ciphertext |

Over the HI-Small test window: dropping 1.31 M transactions saves **~1.1 TB**
ciphertext and **~65 days** of FHE compute.

**Key finding:** the input ciphertext is only **2.3%** of the per-transaction total,
so feature-dimension reduction (P1) can shrink at most ~2.3% of it, while pre-filtering
(M2) removes the whole thing for every dropped transaction. Pre-filtering is by far the
dominant storage/compute lever — quantitatively confirming the spec's caution against
extrapolating "fewer features = big storage savings."

## Feature-improvement POC

The base filter's ranking is not enriched at the top (PR-AUC ~0.01; precision peaks
~1.5%) because it leans on broad categoricals (Payment Format) and lifetime counts. A
proof-of-concept (`scripts/filter_LR_features_poc.py`) adds **bank-local, causal,
no-global-graph** features — 24h rolling transaction / new-counterparty burst
(fan-in/out), repeated-identical-amount count (layering), amount vs the account's past
mean (spike), near-round-amount flag (structuring) — and swaps the linear model for a
**shallow tree** (XGBoost depth 2, spec-allowed).

Sender-only, chain-aware (the honest setting):

| Config | ROC-AUC | PR-AUC | Drop @ 95% retention |
|--------|---------|--------|----------------------|
| A) LR + old features (current) | 0.935 | 0.010 | 82.3% |
| B) LR + old + new features | 0.942 | 0.011 | 87.2% |
| **C) XGB(depth 2) + old + new** | **0.987** | **0.387** | **91.0%** |

- **Drop @ 95% retention 82% → 91%** — only ~9% reaches FHE instead of ~18%, roughly
  halving FHE volume vs the current filter.
- **PR-AUC 0.01 → 0.39** — the top of the ranking is now genuinely laundering; the
  "confident-but-clean" contamination is largely gone. New features alone (B) help a
  little; the **shallow tree (C)** captures the interactions and drives the jump.

> ⚠️ **Caveat — do not quote 91% / 0.39 externally as-is.** HI-Small is *synthetic*,
> and its laundering *is* these structured patterns (fixed-degree fan-out, regular
> structuring) that the new features directly encode — so the magnitude is inflated by
> the synthetic generator. The **direction is validated** (local velocity/structure
> features + shallow trees clearly help), but on real, messier data the lift will be
> smaller; validate on other datasets before reporting numbers. The features are causal
> and the split is chain-aware, so 0.39 is a real held-out figure *for this synthetic
> dataset*, not leakage.

This is a concrete roadmap for future feature work, not a finalized filter.

## Aligned to the main 60/20/20 split (rigorous)

`scripts/filter_LR_622.py` reproduces the main pipeline's 60/20/20 chronological split
(chain-aware) and selects the 95%-retention **threshold on VALID**, applied to TEST —
**no test-set threshold peeking**. (Exact row-join to the feature parquets isn't
possible — they drop the raw identifiers — so this aligns the split rule and the test
time-window, not bit-identical row membership. Reproduced sizes match: train 3.05M /
valid 1.02M / test 1.02M.)

Sender-only. Test: 1,015,349 txns, 1,265 illicit.

| Config | Test drop | Test retention | ROC-AUC | PR-AUC |
|--------|-----------|----------------|---------|--------|
| A) LR + old (current baseline) | 77.6% | 96.0% | 0.926 | 0.010 |
| B) LR + old + new | 70.0% | 97.1% | 0.935 | 0.010 |
| **C) XGB(depth 2) + old + new** | **93.0%** | 92.8% | **0.988** | **0.408** |

- Choosing the threshold on **valid** (not test) makes the **honest baseline ~78% drop**
  at ~96% retention — below the earlier test-peeked 82%.
- The **feature + shallow-tree improvement is robust** to this stricter evaluation: XGB
  still ~93% drop and PR-AUC 0.41.
- Because the threshold comes from valid, achieved test retention drifts from 95%
  (A 96%, B 97%, C 93%). **Compare models by AUC / PR-AUC** (C decisively best); the
  A-vs-B drop% gap is mostly retention-drift noise (B actually ranks better by AUC), not
  a regression — a reminder not to over-read a single drop% at one threshold.
- The synthetic-data caveat from the POC still applies to C's magnitude.

## Discipline / acceptance (A4)

- Filter trained on the **train split only**; test split used for evaluation only. ✅
- Features limited to the **bank's local plaintext view** — no global-graph statistics. ✅
- No FHE required: the filter runs bank-side on plaintext (training is a one-time
  offline plaintext step, like the main model's stage 0).

## Caveats & next steps

1. ~~OR-combine assumes both banks' scores are available.~~ **Resolved:** measured the
   realistic **paying-bank-only** variant (`role ∈ {sender, internal_debit}`) — it is
   as good as OR (**82.3%** chain-aware, vs 81.8% for OR). No cross-bank coordination
   needed. See "Paying-bank-only" above. Note: "95% retention" is the filter's illicit
   **pass-through** rate (it forwards 95% of illicit into FHE), **not** a detection rate
   — detection is still the downstream FHE-XGBoost model's job (recall ~65.5%).
2. **Distribution shift is visible:** test illicit rate (0.15%) is ~2× train (0.08%),
   supporting a per-bank fine-tune strategy for deployment — but the feasibility number
   here uses the simpler pooled model.
3. **Payment-Format dominates the model, and the top of the ranking is not enriched.**
   HI-Small is synthetic, so the Payment-Format weight may be a dataset artifact. More
   importantly, the current features (broad categoricals + lifetime cumulative counts)
   rank OK (ROC) but don't push laundering to the top (precision peaks at ~1.5%).
   **Improvement direction — all bank-local, no global graph:** windowed velocity
   (#tx and #distinct counterparties per own account in the last 1h/24h/7d → fan-in /
   fan-out), amount-structure (near-threshold / round-number / repeated-identical
   amounts → structuring; amount vs the account's recent mean → spikes), and
   pass-through/relay signals (in→out timing and in/out amount ratio within a short
   window → layering / stack / gather-scatter). Pair with a **shallow tree (depth ≤ 2,
   spec-allowed)** to capture interactions the linear LR cannot. Purely topological
   patterns (CYCLE, multi-hop) need the global graph and are out of a local filter's
   reach — those stay the main FHE model's job. Better features raise drop % at the
   same retention (more FHE savings), not just precision.
4. **Split alignment:** this uses the filter's own 70/30 raw-transaction split, not the
   main pipeline's feature-build test split. To convert to savings on the *actual* FHE
   workload, align to the main test set.
5. **If a server-side filter is ever chosen** (instead of bank-side), predict would need
   FHE on the encrypted amounts — Concrete ML supports LR, and FHE-LR is far cheaper than
   FHE-XGBoost, but this is a different deployment than the spec's bank-side design.

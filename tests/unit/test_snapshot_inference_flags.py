"""The flag converter, exercised against a snapshot this bank actually minted.

The sample handoff cannot be used for this: its `tx_ref_hash` preimages come from
bank-sim, so no bank-authored hash of ours can key it. The test therefore builds the
edges slice the way a snapshot composed from *our* upload would look -- taking the
`tx_ref_hash` values our batch builder emits -- and checks both outcomes that matter:
a clean join publishes the contract's four columns, and a foreign snapshot is refused
rather than sealed empty.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/snapshot_inference_flags.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from snapshot_inference_flags import payment_references, read_raw, tx_ref_hash  # noqa: E402

BANK = "bank010"


def _run(snapshot: Path, flags: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--snapshot", str(snapshot),
         "--flags", str(flags), "--bank", BANK, *extra],
        capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def refs():
    """Durable references for the first rows of the source CSV, as the batch builder mints
    them -- the converter derives the same values from the same file."""
    csv = REPO_ROOT / "data/HI-Small_Trans.csv"
    if not csv.exists():
        pytest.skip("source CSV not available")
    return payment_references(read_raw(str(csv)).head(64)).map(tx_ref_hash).to_numpy()


@pytest.fixture
def workspace(tmp_path, refs):
    """A 6-row filter table and the edges a composer would build from that bank's upload."""
    ids = list(range(6))
    kept = [True, False, True, True, False, True]
    flags = tmp_path / "flags.parquet"
    pd.DataFrame({"transaction_id": ids, "is_kept": kept,
                  "risk_score": [0.9, 0.1, 0.8, 0.7, 0.2, 0.95]}).to_parquet(flags)
    flags.with_suffix(".meta.json").write_text(json.dumps({"target_retention": 0.95}))

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    pd.DataFrame({
        # edge_id ordering is deliberately not the flag table's, so a positional
        # implementation would fail the value assertions below
        "edge_id": [f"sha256:{i:064x}" for i in (5, 3, 1, 0, 4, 2)],
        "tx_ref_hash": [refs[i] for i in (5, 3, 1, 0, 4, 2)],
        "reporting_bank": [BANK] * 5 + ["bank999"],
    }).to_parquet(snapshot / "edges.parquet")
    return snapshot, flags


def test_publishes_the_contract_columns(workspace):
    snapshot, flags = workspace
    out = _run(snapshot, flags)
    assert out.returncode == 0, out.stderr

    published = pd.read_parquet(snapshot / f"inference_flags_{BANK}.parquet")
    assert list(published.columns) == [
        "edge_id", "inference_flag", "flag_source_bank", "selection_policy_id"]
    # only this bank's edges; the foreign one is another reporter's to publish
    assert len(published) == 5
    assert published.edge_id.is_unique and published.edge_id.is_monotonic_increasing
    assert published.flag_source_bank.eq(BANK).all()
    assert published.selection_policy_id.eq("m2_lr_p95_v1").all()

    # the decision travels with the edge, not with row order
    by_edge = dict(zip(published.edge_id, published.inference_flag))
    assert by_edge[f"sha256:{0:064x}"] is True    # kept[0]
    assert by_edge[f"sha256:{1:064x}"] is False   # kept[1] -- filter dropped it
    assert by_edge[f"sha256:{4:064x}"] is False   # kept[4]
    assert by_edge[f"sha256:{5:064x}"] is True    # kept[5]


def test_refuses_a_snapshot_this_bank_did_not_mint(workspace, tmp_path):
    """A foreign preimage convention must fail loudly, not seal an all-false artifact."""
    snapshot, flags = workspace
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    edges = pd.read_parquet(snapshot / "edges.parquet")
    edges["tx_ref_hash"] = [f"sha256:{i:064x}" for i in range(100, 106)]
    edges.to_parquet(foreign / "edges.parquet")

    out = _run(foreign, flags)
    assert out.returncode != 0
    assert "preimage" in (out.stdout + out.stderr)
    assert not (foreign / f"inference_flags_{BANK}.parquet").exists()

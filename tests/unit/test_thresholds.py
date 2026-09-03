import numpy as np

from aml_fhe.evaluation.metrics import binary_metrics
from aml_fhe.evaluation.metrics import f1_at_threshold
from aml_fhe.evaluation.thresholds import threshold_sweep


def test_threshold_sweep_finds_best_threshold() -> None:
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.7, 0.9])

    result = threshold_sweep(probs, y)

    assert result.f1 == 1.0
    assert 0.25 <= result.threshold <= 0.70
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_binary_metrics_and_f1_at_threshold() -> None:
    y = np.array([0, 1, 1, 0])
    pred = np.array([0, 1, 0, 0])
    probs = np.array([0.1, 0.8, 0.3, 0.2])

    metrics = binary_metrics(y, pred)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert f1_at_threshold(y, probs, threshold=0.5) == metrics["f1"]

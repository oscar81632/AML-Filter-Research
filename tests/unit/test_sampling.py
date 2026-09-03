import numpy as np

from aml_fhe.models.sampling import eval_sample
from aml_fhe.models.sampling import random_sample
from aml_fhe.models.sampling import train_subsample


def test_train_subsample_keeps_all_illicit_and_caps_normal_rows() -> None:
    x = np.arange(20).reshape(10, 2)
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])

    x_sample, y_sample = train_subsample(x, y, normal_rows=3, seed=42)

    assert len(x_sample) == 6
    assert int(y_sample.sum()) == 3
    assert int((y_sample == 0).sum()) == 3


def test_eval_sample_returns_full_data_when_disabled() -> None:
    x = np.arange(12).reshape(6, 2)
    y = np.array([0, 1, 0, 0, 1, 0])

    x_sample, y_sample = eval_sample(x, y, n_rows=0)

    assert x_sample is x
    assert y_sample is y


def test_random_sample_caps_to_available_rows() -> None:
    x = np.arange(12).reshape(6, 2)
    y = np.array([0, 1, 0, 0, 1, 0])

    x_sample, y_sample = random_sample(x, y, n_rows=500)

    assert x_sample is x
    assert y_sample is y

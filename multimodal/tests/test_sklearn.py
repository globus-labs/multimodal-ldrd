from multimodal.sklearn import NMFWithDTW
from pytest import fixture
import numpy as np


@fixture
def signal() -> np.ndarray:
    x = np.linspace(0, 10, 64)
    def gaussian(x, b):
        return np.exp(-1 * np.power((x - b) * 2, 2))
    return np.vstack([
        gaussian(x, 2 + i) + i * gaussian(x, 8) for i in np.linspace(0, 1, 32)
    ])


def test_runs(signal):
    nmf = NMFWithDTW(2, n_jobs=2, alpha=1, l1_ratio=0.5, absolute_error_weight=0.5,
                     optimizer_args={'disp': False, 'maxfun': 10})
    nmf.fit(signal)


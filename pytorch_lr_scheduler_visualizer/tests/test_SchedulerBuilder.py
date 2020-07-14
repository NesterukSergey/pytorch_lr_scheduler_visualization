import pytest
from pytorch_lr_scheduler_visualizer.lr_scheduler.utils import get_all_default


@pytest.fixture(scope='session')
def default_results():
    return get_all_default()


def test_CosineAnnealingLR(default_results):
    assert default_results['CosineAnnealingLR'][:2] == [0.001, 0.0009999999753259891]


def test_ReduceLROnPlateau(default_results):
    assert default_results['ReduceLROnPlateau'][11:13] == [0.001, 0.0001]


def test_ExponentialLR(default_results):
    assert default_results['ExponentialLR'][:2] == [0.001, 0.0009000000000000001]


def test_MultiStepLR(default_results):
    assert default_results['MultiStepLR'][-1] == 1e-05


def test_StepLR(default_results):
    assert default_results['StepLR'][4:6] == [0.001, 0.0001]

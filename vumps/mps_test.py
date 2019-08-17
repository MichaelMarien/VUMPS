import numpy as np
import pytest
from vumps.mps import MPS


def test_mps_init():
    tensor = np.random.randn(3, 2, 3)
    mps = MPS(tensor)
    assert mps.axis_order == ["left_virtual", "physical", "right_virtual"]
    assert mps.virtual_dimension == 3
    assert mps.physical_dimension == 2
    np.testing.assert_allclose(mps.A.tensor, tensor)


def test_mps_init_different_axis_order():
    tensor = np.random.randn(3, 3, 2)
    mps = MPS(tensor, left_virtual=1, physical=2, right_virtual=0)
    assert mps.axis_order == ["left_virtual", "physical", "right_virtual"]
    assert mps.virtual_dimension == 3
    assert mps.physical_dimension == 2
    np.testing.assert_allclose(mps.A.tensor, np.transpose(tensor, (1, 2, 0)))


def test_mps_init_wrong_dimensions():
    tensor = np.random.randn(3, 2, 2)
    with pytest.raises(ValueError):
        MPS(tensor)

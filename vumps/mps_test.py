import numpy as np
from vumps.mps import MPS


def test_mps_init():
    tensor = np.random.randn(3, 2, 3)
    print(tensor)
    print(tensor.shape)
    mps = MPS(tensor)
    assert mps.axis_ordering == ["left_virtual", "physical", "right_virtual"]
    np.testing.assert_allclose(mps.A.tensor, tensor)


def test_mps_init_different_axis_order():
    tensor = np.random.randn(3, 3, 2)
    print(tensor)
    print(tensor.shape)
    mps = MPS(tensor, left_virtual=1, physical=2, right_virtual=0)
    assert mps.axis_ordering == ["right_virtual", "left_virtual", "physical"]
    np.testing.assert_allclose(mps.A.tensor, tensor)
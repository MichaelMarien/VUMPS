import numpy as np
import pytest
from vumps.mps import MPS
from numpy.linalg import norm, eigvals


def test_mps_init():
    tensor = np.random.randn(3, 2, 3)
    mps = MPS(tensor)
    assert mps.axis_order == ["left_virtual", "physical", "right_virtual"]
    assert mps.virtual_dimension == 3
    assert mps.physical_dimension == 2
    np.testing.assert_allclose(mps.tensor, tensor)


def test_mps_init_different_axis_order():
    tensor = np.random.randn(3, 3, 2)
    mps = MPS(tensor, left_virtual=1, physical=2, right_virtual=0)
    assert mps.axis_order == ["left_virtual", "physical", "right_virtual"]
    assert mps.virtual_dimension == 3
    assert mps.physical_dimension == 2
    np.testing.assert_allclose(mps.tensor, np.transpose(tensor, (1, 2, 0)))


def test_mps_init_wrong_dimensions():
    tensor = np.random.randn(3, 2, 2)
    with pytest.raises(ValueError):
        MPS(tensor)


def test_mps_to_matrix():
    tensor = np.random.randn(3, 2, 3)
    mps = MPS(tensor, left_virtual=0, physical=1, right_virtual=2)

    expected = np.reshape(tensor, (6, 3))
    actual = mps.to_matrix(left_indices=["left_virtual", "physical"],
                           right_indices=["right_virtual"])
    np.testing.assert_allclose(actual, expected)

    expected = np.reshape(np.transpose(tensor, (1, 0, 2)), (6, 3))
    actual = mps.to_matrix(left_indices=["physical", "left_virtual"],
                           right_indices=["right_virtual"])
    np.testing.assert_allclose(actual, expected)

    expected = np.reshape(np.transpose(tensor, (0, 2, 1)), (3, 6))
    actual = mps.to_matrix(left_indices=["left_virtual"],
                           right_indices=["right_virtual", "physical"])
    np.testing.assert_allclose(actual, expected)

    expected = np.reshape(tensor, (3, 6))
    actual = mps.to_matrix(left_indices=["left_virtual"],
                           right_indices=["physical", "right_virtual"])
    np.testing.assert_allclose(actual, expected)


def test_mps_to_matrix_error():
    tensor = np.random.randn(2, 3, 2)
    mps = MPS(tensor)

    with pytest.raises(ValueError):
        mps.to_matrix(left_indices=["left_virtual", "physical"],
                      right_indices=[])

    with pytest.raises(ValueError):
        mps.to_matrix(left_indices=["left_virtual", "physical"],
                      right_indices=["left_virtual"])

    with pytest.raises(ValueError):
        mps.to_matrix(left_indices=["left_virtual", "physical"],
                      right_indices=["left_virtual", "right_virtual"])

    with pytest.raises(ValueError):
        mps.to_matrix(left_indices=["left_virtual", "physical"],
                      right_indices=["a"])


def test_mps_qrpos():
    tensor = np.random.randn(2, 3, 2) + 1j*np.random.randn(2, 3, 2)
    mps = MPS(tensor)
    el = np.random.randn(2, 2) + 1j*np.random.randn(2, 2)
    q, r = mps._QRPos(el)
    expected = np.reshape(np.einsum('ijk,ai->ajk', tensor, el), (6, 2))
    np.testing.assert_allclose(expected, np.dot(q, r))
    np.testing.assert_allclose(np.diag(r), np.real(np.diag(r)))


def test_mps_apply_transfer_matrix():
    tensor = np.random.randn(2, 3, 2) + 1j*np.random.randn(2, 3, 2)
    mps = MPS(tensor)
    v = np.random.randn(2, 2)

    actual = mps.apply_transfer_matrix(v)
    transfer_matrix = np.einsum('ijk,ajb->iakb', tensor, np.conj(tensor))
    expected = np.einsum('ijkl,ji->lk', transfer_matrix, v)
    np.testing.assert_allclose(actual, expected)


def test_mps_apply_mixed_transfer_matrix():
    tensor = np.random.randn(2, 3, 2)
    mps = MPS(tensor)
    B = np.random.randn(4, 3, 4)
    v = np.random.randn(4, 2)

    actual = mps.apply_mixed_transfer_matrix(B, v)
    transfer_matrix = np.einsum('ijk,ajb->iakb', tensor, B)
    expected = np.einsum('ijkl,ji->lk', transfer_matrix, v)
    np.testing.assert_allclose(actual, expected)


def test_mps_transfer_matrix():
    tensor = np.random.randn(2, 3, 2) + 1j*np.random.randn(2, 3, 2)
    mps = MPS(tensor)
    actual = mps.transfer_matrix
    expected = np.einsum('ijk,ajb->iakb', tensor, np.conj(tensor))
    np.testing.assert_allclose(actual.tensor, expected)


def test_mps_left_orthonormalize_gives_fixed_point_single_layer():
    tensor = np.random.randn(2, 3, 2)
    mps = MPS(tensor)
    L0 = np.random.randn(2, 2)
    AL, L, lamb = mps.left_orthonormalize(L0, 10**(-10))

    expected = np.einsum('ijk,ai->ajk', tensor, L)
    actual = np.einsum('ijk,ka->ija', np.reshape(AL, (2, 3, 2)), L)
    np.testing.assert_almost_equal(norm(expected)/norm(actual), lamb, decimal=8)
    np.testing.assert_array_almost_equal(actual/np.linalg.norm(actual),
                                         expected/np.linalg.norm(expected))


def test_mps_left_orthonormalize_gives_fixed_point_qrpos():
    tensor = np.random.randn(2, 3, 2)
    mps = MPS(tensor)
    L0 = np.random.randn(2, 2)
    _, L, _ = mps.left_orthonormalize(L0, 10**(-6))

    _, L2 = mps._QRPos(L)
    np.testing.assert_array_almost_equal(L/np.linalg.norm(L),
                                         L2/np.linalg.norm(L2))


def test_mps_left_orthonormalize_gives_fixed_point_mixed_transfer_matrix():
    tensor = np.random.randn(2, 3, 2)
    mps = MPS(tensor)
    L0 = np.random.randn(2, 2)
    AL, L, _ = mps.left_orthonormalize(L0, 10**(-6))

    actual = mps.apply_mixed_transfer_matrix(np.reshape(np.conj(AL), (2, 3, 2)),
                                             L)
    expected = L
    np.testing.assert_array_almost_equal(actual/norm(actual),
                                         expected/norm(expected))


def test_mps_left_orthonormalize_gives_correct_lambda():
    tensor = np.random.randn(2, 3, 2)
    mps = MPS(tensor)
    transfer_matrix = np.reshape(mps.transfer_matrix.tensor, (4, 4))
    L0 = np.random.randn(2, 2)
    AL, L, lamb = mps.left_orthonormalize(L0, 10**(-6))
    val = np.max(eigvals(transfer_matrix))
    np.testing.assert_almost_equal(np.sqrt(val), lamb)


def test_mps_left_orthonormalize_gives_same_uniform_mps():
    tensor = np.random.randn(2, 3, 2)
    mps = MPS(tensor)
    L0 = np.random.randn(2, 2)
    AL, L, lamb = mps.left_orthonormalize(L0, 10**(-10))
    ALt = np.conj(np.reshape(AL, (2, 3, 2)))
    mixed = np.reshape(np.einsum('ijk,ajb->iakb', tensor, ALt), (4, 4))
    val = np.max(eigvals(mixed))
    np.testing.assert_almost_equal(val, lamb, decimal=8)

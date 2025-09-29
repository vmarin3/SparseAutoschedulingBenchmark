import pytest

import numpy as np

from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework


@pytest.fixture
def xp():
    return NumpyFramework()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_basic_addition_with_transpose(xp, rng):
    """Test basic addition with transpose"""
    A = rng.random((5, 5))
    B = rng.random((5, 5))

    C = xp.einsum("C[i,j] = A[i,j] + B[j,i]", A=A, B=B)
    C_ref = A + B.T

    assert np.allclose(C, C_ref)


def test_matrix_multiplication(xp, rng):
    """Test matrix multiplication using += (increment/accumulation)"""
    A = rng.random((3, 4))
    B = rng.random((4, 5))

    C = xp.einsum("C[i,j] += A[i,k] * B[k,j]", A=A, B=B)
    C_ref = A @ B

    assert np.allclose(C, C_ref)


def test_element_wise_multiplication(xp, rng):
    """Test element-wise multiplication"""
    A = rng.random((4, 4))
    B = rng.random((4, 4))

    C = xp.einsum("C[i,j] = A[i,j] * B[i,j]", A=A, B=B)
    C_ref = A * B

    assert np.allclose(C, C_ref)


def test_sum_reduction(xp, rng):
    """Test sum reduction using +="""
    A = rng.random((3, 4))

    C = xp.einsum("C[i] += A[i,j]", A=A)
    C_ref = np.sum(A, axis=1)

    assert np.allclose(C, C_ref)


def test_maximum_reduction(xp, rng):
    """Test maximum reduction using max="""
    A = rng.random((3, 4))

    C = xp.einsum("C[i] max= A[i,j]", A=A)
    C_ref = np.max(A, axis=1)

    assert np.allclose(C, C_ref)


def test_outer_product(xp, rng):
    """Test outer product"""
    A = rng.random(3)
    B = rng.random(4)

    C = xp.einsum("C[i,j] = A[i] * B[j]", A=A, B=B)
    C_ref = np.outer(A, B)

    assert np.allclose(C, C_ref)


def test_batch_matrix_multiplication(xp, rng):
    """Test batch matrix multiplication using +="""
    A = rng.random((2, 3, 4))
    B = rng.random((2, 4, 5))

    C = xp.einsum("C[b,i,j] += A[b,i,k] * B[b,k,j]", A=A, B=B)
    C_ref = np.matmul(A, B)

    assert np.allclose(C, C_ref)


def test_minimum_reduction(xp, rng):
    """Test minimum reduction using min="""
    A = rng.random((3, 4))

    C = xp.einsum("C[i] min= A[i,j]", A=A)
    C_ref = np.min(A, axis=1)

    assert np.allclose(C, C_ref)

@pytest.mark.parametrize("axis",
    [
        (0, 2, 1),
        (3, 0, 1),
        (1, 0, 3, 2),
        (1, 0, 3, 2)
    ]
)
@pytest.mark.parametrize("idxs",
    [
        ("i", "j", "k", "l"),
        ("l", "j", "k", "i"),
        ("l", "k", "j", "i"),
    ]
)
def test_swizzle_in(xp, rng, axis, idxs):
    """Test transpositions with einsum"""
    A = rng.random((4, 4, 4, 4))

    jdxs = [idxs[p] for p in axis]
    xp_idxs = ", ".join(idxs)
    np_idxs = "".join(idxs)
    xp_jdxs = ", ".join(jdxs)
    np_jdxs = "".join(jdxs)

    C = xp.einsum(f"C[{xp_jdxs}] += A[{xp_idxs}]", A=A)
    C_ref = np.einsum(f"{np_idxs}->{np_jdxs}", A)

    assert np.allclose(C, C_ref)


def test_operator_precedence_arithmetic(xp, rng):
    """Test that arithmetic operator precedence follows Python rules"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3))
    
    # Test: A + B * C should be A + (B * C), not (A + B) * C
    result = xp.einsum("D[i,j] = A[i,j] + B[i,j] * C[i,j]", A=A, B=B, C=C)
    expected = A + (B * C)
    
    assert np.allclose(result, expected)


def test_operator_precedence_power_and_multiplication(xp, rng):
    """Test that power has higher precedence than multiplication"""
    A = rng.random((3, 3)) + 1  # Add 1 to avoid numerical issues with powers
    
    # Test: A * A ** 2 should be A * (A ** 2), not (A * A) ** 2
    result = xp.einsum("B[i,j] = A[i,j] * A[i,j] ** 2", A=A)
    expected = A * (A ** 2)
    
    assert np.allclose(result, expected)


def test_operator_precedence_addition_and_multiplication(xp, rng):
    """Test complex arithmetic precedence: A + B * C ** 2"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3)) + 1  # Add 1 to avoid numerical issues
    
    # Test: A + B * C ** 2 should be A + (B * (C ** 2))
    result = xp.einsum("D[i,j] = A[i,j] + B[i,j] * C[i,j] ** 2", A=A, B=B, C=C)
    expected = A + (B * (C ** 2))
    
    assert np.allclose(result, expected)


def test_operator_precedence_logical_and_or(xp, rng):
    """Test that 'and' has higher precedence than 'or'"""
    A = (rng.random((3, 3)) > 0.3).astype(float)  # Boolean-like arrays
    B = (rng.random((3, 3)) > 0.3).astype(float)
    C = (rng.random((3, 3)) > 0.3).astype(float)
    
    # Test: A or B and C should be A or (B and C), not (A or B) and C
    result = xp.einsum("D[i,j] = A[i,j] or B[i,j] and C[i,j]", A=A, B=B, C=C)
    expected = np.logical_or(A, np.logical_and(B, C)).astype(float)
    
    assert np.allclose(result, expected)


def test_operator_precedence_bitwise_operations(xp, rng):
    """Test bitwise operator precedence: | has lower precedence than ^ which has lower than &"""
    # Use integer arrays for bitwise operations
    A = rng.integers(0, 8, size=(3, 3))
    B = rng.integers(0, 8, size=(3, 3))
    C = rng.integers(0, 8, size=(3, 3))
    D = rng.integers(0, 8, size=(3, 3))
    
    # Test: A | B ^ C & D should be A | (B ^ (C & D))
    result = xp.einsum("E[i,j] = A[i,j] | B[i,j] ^ C[i,j] & D[i,j]", A=A, B=B, C=C, D=D)
    expected = A | (B ^ (C & D))
    
    assert np.allclose(result, expected)


def test_operator_precedence_shift_operations(xp, rng):
    """Test shift operator precedence with arithmetic"""
    # Use small integer arrays to avoid overflow in shifts
    A = rng.integers(1, 4, size=(3, 3))
    
    # Test: A << 1 + 1 should be A << (1 + 1), not (A << 1) + 1
    # Since shift has lower precedence than addition
    result = xp.einsum("B[i,j] = A[i,j] << 1 + 1", A=A)
    expected = A << (1 + 1)  # A << 2
    
    assert np.allclose(result, expected)


def test_operator_precedence_comparison_with_arithmetic(xp, rng):
    """Test that arithmetic has higher precedence than comparison"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3))
    
    # Test: A + B == C should be (A + B) == C, not A + (B == C)
    result = xp.einsum("D[i,j] = A[i,j] + B[i,j] == C[i,j]", A=A, B=B, C=C)
    expected = ((A + B) == C).astype(float)
    
    assert np.allclose(result, expected)


def test_operator_precedence_with_parentheses(xp, rng):
    """Test that parentheses override operator precedence"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    C = rng.random((3, 3))
    
    # Test: (A + B) * C should be different from A + B * C
    result_with_parens = xp.einsum("D[i,j] = (A[i,j] + B[i,j]) * C[i,j]", A=A, B=B, C=C)
    result_without_parens = xp.einsum("E[i,j] = A[i,j] + B[i,j] * C[i,j]", A=A, B=B, C=C)
    
    expected_with_parens = (A + B) * C
    expected_without_parens = A + (B * C)
    
    assert np.allclose(result_with_parens, expected_with_parens)
    assert np.allclose(result_without_parens, expected_without_parens)
    
    # Verify they're different (unless by coincidence)
    if not np.allclose(expected_with_parens, expected_without_parens):
        assert not np.allclose(result_with_parens, result_without_parens)


def test_operator_precedence_unary_operators(xp, rng):
    """Test unary operator precedence"""
    A = rng.random((3, 3)) - 0.5  # Some negative values
    
    # Test: -A ** 2 should be -(A ** 2), not (-A) ** 2
    result = xp.einsum("B[i,j] = -A[i,j] ** 2", A=A)
    expected = -(A ** 2)
    
    assert np.allclose(result, expected)


def test_numeric_literals(xp, rng):
    """Test that numeric literals work correctly"""
    A = rng.random((3, 3))
    
    # Test simple addition with literal
    result = xp.einsum("B[i,j] = A[i,j] + 1", A=A)
    expected = A + 1
    
    assert np.allclose(result, expected)
    
    # Test complex expression with literals
    result2 = xp.einsum("C[i,j] = A[i,j] * 2 + 3", A=A)
    expected2 = A * 2 + 3
    
    assert np.allclose(result2, expected2)

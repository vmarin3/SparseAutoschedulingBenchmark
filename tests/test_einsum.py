import pytest
from SparseAutoschedulingBenchmark import NumpyFramework
import numpy as np


@pytest.fixture
def xp():
    return NumpyFramework()


def test_basic_addition_with_transpose(xp):
    """Test basic addition with transpose"""
    A = np.random.rand(5, 5)
    B = np.random.rand(5, 5)
    
    C = xp.einsum("C[i,j] = A[i,j] + B[j,i]", A=A, B=B)
    C_ref = A + B.T
    
    assert np.allclose(C, C_ref)


def test_matrix_multiplication(xp):
    """Test matrix multiplication using += (increment/accumulation)"""
    A = np.random.rand(3, 4)
    B = np.random.rand(4, 5)
    
    C = xp.einsum("C[i,j] += A[i,k] * B[k,j]", A=A, B=B)
    C_ref = A @ B
    
    assert np.allclose(C, C_ref)


def test_element_wise_multiplication(xp):
    """Test element-wise multiplication"""
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    
    C = xp.einsum("C[i,j] = A[i,j] * B[i,j]", A=A, B=B)
    C_ref = A * B
    
    assert np.allclose(C, C_ref)


def test_sum_reduction(xp):
    """Test sum reduction using +="""
    A = np.random.rand(3, 4)
    
    C = xp.einsum("C[i] += A[i,j]", A=A)
    C_ref = np.sum(A, axis=1)
    
    assert np.allclose(C, C_ref)


def test_maximum_reduction(xp):
    """Test maximum reduction using max="""
    A = np.random.rand(3, 4)
    
    C = xp.einsum("C[i] max= A[i,j]", A=A)
    C_ref = np.max(A, axis=1)
    
    assert np.allclose(C, C_ref)


def test_outer_product(xp):
    """Test outer product"""
    A = np.random.rand(3)
    B = np.random.rand(4)
    
    C = xp.einsum("C[i,j] = A[i] * B[j]", A=A, B=B)
    C_ref = np.outer(A, B)
    
    assert np.allclose(C, C_ref)


def test_batch_matrix_multiplication(xp):
    """Test batch matrix multiplication using +="""
    A = np.random.rand(2, 3, 4)
    B = np.random.rand(2, 4, 5)
    
    C = xp.einsum("C[b,i,j] += A[b,i,k] * B[b,k,j]", A=A, B=B)
    C_ref = np.matmul(A, B)
    
    assert np.allclose(C, C_ref)


def test_minimum_reduction(xp):
    """Test minimum reduction using min="""
    A = np.random.rand(3, 4)
    
    C = xp.einsum("C[i] min= A[i,j]", A=A)
    C_ref = np.min(A, axis=1)
    
    assert np.allclose(C, C_ref)
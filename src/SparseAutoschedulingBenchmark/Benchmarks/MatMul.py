import numpy as np
import scipy.sparse as sp
from ..BinsparseFormat import BinsparseFormat

def benchmark_matmul(xp, A_bench, B_bench):
    A = xp.from_benchmark(A_bench)
    B = xp.from_benchmark(B_bench)
    C = xp.dot(A, B)
    D = xp.to_benchmark(C)
    return D

def dg_matmul_dense_small():
    np.random.seed(0)
    A = np.random.rand(32, 32).astype(np.float32)
    B = np.random.rand(32, 32).astype(np.float32)
    A_bin = BinsparseFormat.from_numpy(A)
    B_bin = BinsparseFormat.from_numpy(B)
    return (A_bin, B_bin)

def dg_matmul_dense_large():
    np.random.seed(0)
    A = np.random.rand(4096, 4096).astype(np.float32)
    B = np.random.rand(4096, 4096).astype(np.float32)
    A_bin = BinsparseFormat.from_numpy(A)
    B_bin = BinsparseFormat.from_numpy(B)
    return (A_bin, B_bin)


def dg_matmul_sparse_small():
    np.random.seed(0)
    A = sp.random(32, 32, density=.1, format='coo', dtype=np.float32)
    B = sp.random(32, 32, density=.1, format='coo', dtype=np.float32)
    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    B_bin = BinsparseFormat.from_coo((B.row, B.col), B.data, B.shape)
    return (A_bin, B_bin)


def dg_matmul_sparse_large():
    np.random.seed(0)
    A = sp.random(4096, 4096, density=.01, format='coo', dtype=np.float32)
    B = sp.random(4096, 4096, density=.01, format='coo', dtype=np.float32)
    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    B_bin = BinsparseFormat.from_coo((B.row, B.col), B.data, B.shape)
    return (A_bin, B_bin)

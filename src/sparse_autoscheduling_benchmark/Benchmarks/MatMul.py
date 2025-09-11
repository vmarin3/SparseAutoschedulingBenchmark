import numpy as np
from BinsparseFormat import BinsparseFormat

def benchmark_matmul(xp, A_bench, B_bench):
    A = xp.from_benchmark(A_bench)
    B = xp.from_benchmark(B_bench)
    C = xp.dot(A, B)
    D = xp.to_benchmark(C)
    return D

def datagen_dense_matmul_small():
    np.random.seed(0)
    A = np.random.rand(32, 32).astype(np.float32)
    B = np.random.rand(32, 32).astype(np.float32)
    A_bin = BinsparseFormat.from_numpy(A)
    B_bin = BinsparseFormat.from_numpy(B)
    return (A_bin, B_bin)

def datagen_dense_matmul_large():
    np.random.seed(0)
    A = np.random.rand(4096, 4096).astype(np.float32)
    B = np.random.rand(4096, 4096).astype(np.float32)
    A_bin = BinsparseFormat.from_numpy(A)
    B_bin = BinsparseFormat.from_numpy(B)
    return (A_bin, B_bin)

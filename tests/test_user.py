import os

import numpy as np

import SparseAutoschedulingBenchmark as autobench
from SparseAutoschedulingBenchmark.Benchmarks.MatMul import (
    benchmark_matmul,
    dg_matmul_dense_large,
)
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.AbstractFramework import AbstractFramework


class NumpyTestFramework(AbstractFramework):
    def __init__(self):
        pass

    def from_benchmark(self, array):
        if array.data["format"] == "dense":
            return np.array(array.data["values"]).reshape(array.data["shape"])
        raise ValueError("Unsupported format: " + array.data["format"])

    def to_benchmark(self, array):
        return BinsparseFormat.from_numpy(array)

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def __getattr__(self, name):
        return getattr(np, name)


def test_main(tmp_path):
    autobench.main(
        frameworks=[NumpyTestFramework()],
        framework_names=["NumpyTestFramework"],
        data_generators=[dg_matmul_dense_large],
        data_generator_names=["dense_large"],
        benchmarks=[benchmark_matmul],
        benchmark_names=["matmul"],
        results_folder=tmp_path,
    )
    print(tmp_path / "NumpyTestFramework_matmul_dense_large.csv")
    assert os.path.exists(tmp_path / "NumpyTestFramework_matmul_dense_large.csv")

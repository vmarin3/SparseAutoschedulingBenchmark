import pytest

import numpy as np

from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework


@pytest.fixture
def test_numpy_framework():
    framework = NumpyFramework()

    # Dense array test
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    bsf = BinsparseFormat.from_numpy(arr)
    arr_converted = framework.from_benchmark(bsf)
    assert np.array_equal(arr, arr_converted), "Dense array conversion failed"

    bsf_converted = framework.to_benchmark(arr)
    assert bsf.data == bsf_converted.data, "Dense array to_benchmark failed"

    # Sparse array test (COO format)
    row = np.array([0, 1, 2])
    col = np.array([0, 2, 1])
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    bsf_sparse = BinsparseFormat.from_coo((row, col), data, shape)
    arr_sparse_converted = framework.from_benchmark(bsf_sparse)

    expected_sparse = np.zeros(shape, dtype=np.float32)
    expected_sparse[row, col] = data
    assert np.array_equal(expected_sparse, arr_sparse_converted), (
        "Sparse array conversion failed"
    )

    bsf_sparse_converted = framework.to_benchmark(arr_sparse_converted)
    assert bsf_sparse.data == bsf_sparse_converted.data, (
        "Sparse array to_benchmark failed"
    )

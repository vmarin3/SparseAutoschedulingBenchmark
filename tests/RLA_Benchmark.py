import numpy as np
import scipy.sparse as sp

from ..BinsparseFormat import BinsparseFormat

"""
Name: Random Linear Algebra and Algorithms 
Author: VIlohith Gokarakonda

The purpose of this is to create python tests that are for Randomized Linear Algebra methods.
    Specifically, I will first show the application of the Johnson Lindenstrauss Lemma for Nearest Neighbor. 
"""


def benchmark_johnson_lindenstrauss_nn(xp, data_bench, query_bench, k=5, eps=0.1):
    data = xp.lazy(xp.from_benchmark(data_bench))
    query = xp.lazy(xp.from_benchmark(query_bench))

    n_samples, n_features = data.shape
    query_len = query.shape[0]

    target_dim = max(10, int(4 * xp.log(n_samples) / (eps * eps)))
    if target_dim > n_features:
        target_dim = n_features

    rng = xp.random.RandomState(42)  # Fixed seed for reproducibility
    projection_matrix = rng.normal(0, 1 / xp.sqrt(target_dim), (n_features, target_dim))

    projected_data = xp.matmul(data, projection_matrix)
    projected_query = xp.matmul(query, projection_matrix)

    distances = xp.zeros(n_samples)

    for i in range(n_samples):
        diff = projected_data[i] - projected_query
        distances[i] = xp.sum(diff * diff)  # L2 distance squared

    nearest_indices = xp.zeros(k, dtype=xp.int32)
    nearest_distances = xp.full(k, xp.inf)

    for i in range(n_samples):
        current_dist = distances[i]

        for j in range(k):
            if current_dist < nearest_distances[j]:
                for l in range(k - 1, j, -1):
                    nearest_distances[l] = nearest_distances[l - 1]
                    nearest_indices[l] = nearest_indices[l - 1]

                nearest_distances[j] = current_dist
                nearest_indices[j] = i
                break

    # Convert results back to benchmark format
    result_indices = xp.stack([nearest_indices, xp.arange(k)], axis=0)
    result_values = nearest_distances
    result_shape = (n_samples, k)

    result = xp.stack([result_indices, result_values], axis=0)
    result = xp.lazy(xp.compute(result))

    return xp.to_benchmark(result)


def johnson_lindenstrauss_nn_small():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]])
    query = np.array([1.5, 2.5, 3.5])

    data_bin = BinsparseFormat.from_numpy(data)
    query_bin = BinsparseFormat.from_numpy(query)

    return (data_bin, query_bin)

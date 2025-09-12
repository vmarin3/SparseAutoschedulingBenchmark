import argparse
import time

from .Benchmarks.MatMul import (
    benchmark_matmul,
    dg_matmul_dense_large,
    dg_matmul_dense_small,
    dg_matmul_sparse_large,
    dg_matmul_sparse_small,
)
from .Frameworks.NumpyFramework import NumpyFramework

wrappers = {"numpy": NumpyFramework()}
benchmarks = {"matmul": benchmark_matmul}
data_generators = {
    "matmul": {
        "dense_small": dg_matmul_dense_small,
        "dense_large": dg_matmul_dense_large,
        "sparse_small": dg_matmul_sparse_small,
        "sparse_large": dg_matmul_sparse_large,
    }
}


def run_benchmark(xp, benchmark_function, benchmark_data_generator):
    avg_duration = 0
    for _ in range(5):
        data = benchmark_data_generator()
        start = time.perf_counter()
        result = benchmark_function(xp, *data)
        end = time.perf_counter()
        duration = end - start
        avg_duration += duration
    avg_duration /= 5
    print(f"Benchmark took {avg_duration} seconds")
    return avg_duration, result


def save_benchmark_result(results_folder, duration, wrapper, benchmark, data_generator):
    filename = f"{results_folder}/{wrapper}_{benchmark}_{data_generator}.bin"
    with open(filename, "w") as f:
        f.write(f"Duration: {duration}\n")


def main(
    wrapper=None,
    wrapper_name=None,
    benchmark=None,
    benchmark_name=None,
    data_generator=None,
    data_generator_name=None,
    results_folder=None,
):
    parser = argparse.ArgumentParser(description="Run sparse autoscheduling benchmark")
    parser.add_argument(
        "--wrapper",
        default="numpy",
        choices=list(wrappers.keys()),
        help="Execution wrapper to use",
    )
    parser.add_argument(
        "--benchmark",
        default="matmul",
        choices=list(benchmarks.keys()),
        help="Benchmark to run",
    )
    parser.add_argument(
        "--data-generator", default="default", help="Data generator to use"
    )
    parser.add_argument(
        "--results-folder", default="results", help="Folder to save results"
    )
    args = parser.parse_args()
    if wrapper is None:
        wrapper = wrappers[args.wrapper]
        wrapper_name = args.wrapper
    if benchmark is None:
        benchmark = benchmarks[args.benchmark]
        benchmark_name = args.benchmark
    if data_generator is None:
        data_generator = data_generators[args.benchmark][args.data_generator]
        data_generator_name = args.data_generator
    if results_folder is None:
        results_folder = args.results_folder

    avg_duration, result = run_benchmark(wrapper, benchmark, data_generator)
    save_benchmark_result(
        results_folder, avg_duration, wrapper_name, benchmark_name, data_generator_name
    )

import time
from SparseAutoschedulingBenchmark.Wrappers.NumpyWrapper import NumpyWrapper
from SparseAutoschedulingBenchmark.Benchmarks.MatMul import benchmark_matmul, dg_matmul_dense_small, dg_matmul_dense_large, dg_matmul_sparse_small, dg_matmul_sparse_large
import argparse

wrappers = {"numpy" : NumpyWrapper()}
benchmarks = {"matmul" : benchmark_matmul}
data_generators = {"matmul": {"dense_small" : dg_matmul_dense_small,
                            "dense_large" : dg_matmul_dense_large,
                            "sparse_small" : dg_matmul_sparse_small,
                            "sparse_large" : dg_matmul_sparse_large}}

def run_benchmark(xp, benchmark_function, benchmark_data_generator):
    avg_duration = 0 
    for i in range(5):
        data = benchmark_data_generator()
        start = time.perf_counter()
        result = benchmark_function(xp, *data)
        end = time.perf_counter()
        duration = end - start
        avg_duration += duration
    avg_duration /= 5
    print(f"Benchmark took {avg_duration} seconds")
    return avg_duration, result

def save_benchmark_result(duration, wrapper, benchmark, data_generator):
    filename = f"results/{wrapper}_{benchmark}_{data_generator}.bin"
    with open(filename, "w") as f:
        f.write(f"Duration: {duration}\n")

def main():
    parser = argparse.ArgumentParser(description="Run sparse autoscheduling benchmark")
    parser.add_argument("--wrapper", default="numpy", choices=list(wrappers.keys()), help="Execution wrapper to use")
    parser.add_argument("--benchmark", default="matmul", choices=list(benchmarks.keys()), help="Benchmark to run")
    parser.add_argument("--data-generator", default="sparse_small", help="Data generator to use")
    args = parser.parse_args()

    xp = wrappers[args.wrapper]
    benchmark_func = benchmarks[args.benchmark]
    data_gen = data_generators[args.benchmark][args.data_generator]

    avg_duration, result = run_benchmark(xp, benchmark_func, data_gen)
    save_benchmark_result(avg_duration, args.wrapper, args.benchmark, args.data_generator)

if __name__ == "__main__":
    main()
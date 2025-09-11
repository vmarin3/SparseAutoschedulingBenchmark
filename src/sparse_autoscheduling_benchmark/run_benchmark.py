import time
from Wrappers.NumpyWrapper import NumpyWrapper
from Benchmarks.MatMul import benchmark_matmul, datagen_dense_matmul_small, datagen_dense_matmul_large
import argparse

wrappers = {"numpy" : NumpyWrapper()}
benchmarks = {"matmul" : benchmark_matmul}
data_generators = {"dense_matmul_small" : datagen_dense_matmul_small, "dense_matmul_large" : datagen_dense_matmul_large}

def run_benchmark(xp, benchmark_function, benchmark_data_generator):
    data = benchmark_data_generator()
    start = time.perf_counter()
    result = benchmark_function(xp, *data)
    end = time.perf_counter()
    duration = end - start
    print(f"Benchmark took {duration} seconds")
    return result

def main():
    parser = argparse.ArgumentParser(description="Run sparse autoscheduling benchmark")
    parser.add_argument("--wrapper", default="numpy", choices=list(wrappers.keys()), help="Execution wrapper to use")
    parser.add_argument("--benchmark", default="matmul", choices=list(benchmarks.keys()), help="Benchmark to run")
    parser.add_argument("--data-generator", default="dense_matmul_small", choices=list(data_generators.keys()), help="Data generator to use")
    args = parser.parse_args()

    xp = wrappers[args.wrapper]
    benchmark_func = benchmarks[args.benchmark]
    data_gen = data_generators[args.data_generator]

    run_benchmark(xp, benchmark_func, data_gen)

if __name__ == "__main__":
    main()
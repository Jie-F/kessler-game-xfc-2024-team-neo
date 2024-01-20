import math
import numpy as np
import timeit
import random

# Generating a list of random values
random_values = [random.uniform(-math.pi, math.pi) for _ in range(1000)]

# Benchmarking math.sin()
def benchmark_math_sin():
    return [math.floor(value) for value in random_values]

time_math_sin = timeit.timeit(benchmark_math_sin, number=100)

# Benchmarking np.sin()
def benchmark_np_sin():
    return np.floor(random_values)

time_np_sin = timeit.timeit(benchmark_np_sin, number=100)

# Comparing results
math_sin_results = benchmark_math_sin()
np_sin_results = benchmark_np_sin()

# Since the results are floating-point numbers, we compare them using a tolerance for precision
tolerance = 1e-10
results_match = all(m == n for m, n in zip(math_sin_results, np_sin_results))

time_math_sin, time_np_sin, results_match

print(time_math_sin)
print(time_np_sin)
print(results_match)

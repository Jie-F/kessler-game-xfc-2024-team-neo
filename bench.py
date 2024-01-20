import numpy as np
import math
import timeit

# Generating 10 million random positive floating point numbers
num_samples = 10_000_000
random_numbers = np.random.random_sample(num_samples) * 100

# Function to benchmark np.sqrt
def benchmark_np_sqrt():
    for num in random_numbers:
        np.sqrt(num)

# Function to benchmark math.sqrt
def benchmark_math_sqrt():
    for num in random_numbers:
        math.sqrt(num)

# Number of runs for each benchmark
num_runs = 3

# Using timeit to measure execution time
np_time_taken = timeit.timeit(benchmark_np_sqrt, number=num_runs) / num_runs
math_time_taken = timeit.timeit(benchmark_math_sqrt, number=num_runs) / num_runs

# Print results
print(f"Average time taken for np.sqrt in a loop (over {num_runs} runs): {np_time_taken} seconds")
print(f"Average time taken for math.sqrt in a loop (over {num_runs} runs): {math_time_taken} seconds")

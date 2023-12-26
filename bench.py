import timeit

# Number of repetitions for each operation
repetitions = 1000000

# Operations to benchmark
operations = {
    "Floating Point Addition": "3.14 + 1.59",
    "Floating Point Multiplication": "3.14 * 1.59",
    "Floating Point Division": "3.14 / 1.59",
    "Square Root": "math.sqrt(3.14)",
    "Exponent": "math.exp(1.59)",
    "Logarithm": "math.log(3.14)",
    "Sine": "math.sin(3.14)",
    "Cosine": "math.cos(1.59)"
}

# Importing math module for square root, exponent, logarithm, sine, and cosine
setup_code = "import math"

# Benchmarking each operation
benchmark_results = {}
for operation, code in operations.items():
    time_taken = timeit.timeit(code, setup=setup_code, number=repetitions)
    benchmark_results[operation] = time_taken

print(benchmark_results)


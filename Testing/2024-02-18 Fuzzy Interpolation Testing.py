import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.interpolate import interp1d
import time

# FIS setup
time_until_hit = ctrl.Antecedent(np.arange(0, 101, 1), 'time_until_hit')
fitness = ctrl.Consequent(np.arange(0, 101, 1), 'fitness')

time_until_hit['low'] = fuzz.trimf(time_until_hit.universe, [0, 0, 50])
time_until_hit['medium'] = fuzz.trimf(time_until_hit.universe, [0, 50, 100])
time_until_hit['high'] = fuzz.trimf(time_until_hit.universe, [50, 100, 100])

fitness['low'] = fuzz.trimf(fitness.universe, [0, 0, 50])
fitness['medium'] = fuzz.trimf(fitness.universe, [0, 50, 100])
fitness['high'] = fuzz.trimf(fitness.universe, [50, 100, 100])

rule1 = ctrl.Rule(time_until_hit['low'], fitness['low'])
rule2 = ctrl.Rule(time_until_hit['medium'], fitness['medium'])
rule3 = ctrl.Rule(time_until_hit['high'], fitness['high'])

fitness_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fitness_system = ctrl.ControlSystemSimulation(fitness_ctrl)

start_time_lookup_creation = time.time()
# Creating lookup table
sample_points = 100
times_sampled = np.linspace(0, 100, sample_points)
fitness_values_sampled = []

for time_value in times_sampled:
    fitness_system.input['time_until_hit'] = time_value
    fitness_system.compute()
    fitness_values_sampled.append(fitness_system.output['fitness'])

lookup_table = interp1d(times_sampled, fitness_values_sampled, kind='linear')
end_time_lookup_creation = time.time()

# Benchmark setup
iterations = 10000  # Adjust if necessary
random_times = np.random.uniform(0, 100, iterations)

# Direct computation benchmark
start_time_direct = time.time()
for time_value in random_times:
    fitness_system.input['time_until_hit'] = time_value
    fitness_system.compute()
end_time_direct = time.time()
direct_computation_time = end_time_direct - start_time_direct

# Lookup table computation benchmark
start_time_lookup = time.time()
for time_value in random_times:
    _ = lookup_table(time_value)
end_time_lookup = time.time()
lookup_computation_time = end_time_lookup - start_time_lookup

# Results
print(f"Direct computation time for {iterations} iterations: {direct_computation_time} seconds.")
print(f"Lookup table computation time for {iterations} iterations: {lookup_computation_time} seconds.")
print(f"Lookup table took {end_time_lookup_creation - start_time_lookup_creation} seconds to create.")

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Input variables
asteroids_destroyed = ctrl.Antecedent(np.arange(0, 11, 1), 'asteroids_destroyed')
maneuver_sequence_length = ctrl.Antecedent(np.arange(0, 41, 1), 'maneuver_sequence_length')

# Output variable
fitness = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'fitness')

# Input variables


# Output variable


# Membership functions
'''
asteroids_destroyed['few'] = fuzz.trimf(asteroids_destroyed.universe, [0, 0, 2])
asteroids_destroyed['moderate'] = fuzz.trimf(asteroids_destroyed.universe, [1, 4, 10])
asteroids_destroyed['many'] = fuzz.trimf(asteroids_destroyed.universe, [4, 10, 10])

maneuver_sequence_length['short'] = fuzz.trimf(maneuver_sequence_length.universe, [0, 0, 12])
maneuver_sequence_length['medium'] = fuzz.trimf(maneuver_sequence_length.universe, [6, 20, 40])
maneuver_sequence_length['long'] = fuzz.trimf(maneuver_sequence_length.universe, [20, 40, 40])

fitness['low'] = fuzz.trimf(fitness.universe, [0, 0, 0.50])
fitness['medium'] = fuzz.trimf(fitness.universe, [0.25, 0.50, 0.75])
fitness['high'] = fuzz.trimf(fitness.universe, [0.50, 1, 1])
'''

# Switching to Gaussian membership functions
asteroids_destroyed['few'] = fuzz.gaussmf(asteroids_destroyed.universe, 0, 1.5)
asteroids_destroyed['moderate'] = fuzz.gaussmf(asteroids_destroyed.universe, 5, 2)
asteroids_destroyed['many'] = fuzz.gaussmf(asteroids_destroyed.universe, 10, 1.5)

maneuver_sequence_length['short'] = fuzz.gaussmf(maneuver_sequence_length.universe, 0, 6)
maneuver_sequence_length['medium'] = fuzz.gaussmf(maneuver_sequence_length.universe, 20, 10)
maneuver_sequence_length['long'] = fuzz.gaussmf(maneuver_sequence_length.universe, 40, 6)

# For the output variable, since it spans a 0-1 range, we need to adjust the parameters accordingly.
# Assuming the universe of discourse for fitness is 0 to 100 (to keep it consistent with your setup),
# we scale the Gaussian parameters accordingly.
fitness['low'] = fuzz.gaussmf(fitness.universe, 0, 0.1667)
fitness['medium'] = fuzz.gaussmf(fitness.universe, 0.50, 0.1667)
fitness['high'] = fuzz.gaussmf(fitness.universe, 1, 0.1667)


rule1 = ctrl.Rule(asteroids_destroyed['many'] & maneuver_sequence_length['short'], fitness['high'])
rule2 = ctrl.Rule(asteroids_destroyed['many'] & maneuver_sequence_length['medium'], fitness['high'])
rule3 = ctrl.Rule(asteroids_destroyed['many'] & maneuver_sequence_length['long'], fitness['medium'])

rule4 = ctrl.Rule(asteroids_destroyed['moderate'] & maneuver_sequence_length['short'], fitness['high'])
rule5 = ctrl.Rule(asteroids_destroyed['moderate'] & maneuver_sequence_length['medium'], fitness['medium'])
rule6 = ctrl.Rule(asteroids_destroyed['moderate'] & maneuver_sequence_length['long'], fitness['low'])

rule7 = ctrl.Rule(asteroids_destroyed['few'] & maneuver_sequence_length['short'], fitness['medium'])
rule8 = ctrl.Rule(asteroids_destroyed['few'] & maneuver_sequence_length['medium'], fitness['low'])
rule9 = ctrl.Rule(asteroids_destroyed['few'] & maneuver_sequence_length['long'], fitness['low'])


fitness_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
fitness_sim = ctrl.ControlSystemSimulation(fitness_ctrl)



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skfuzzy import control as ctrl, trimf, interp_membership

# Define the range of inputs for asteroids destroyed and maneuver sequence length
asteroids_range = np.linspace(0, 10, 10)  # Assuming a range from 0 to 100 for asteroids destroyed
maneuver_range = np.linspace(0, 40, 40)  # Assuming a range from 0 to 100 for maneuver sequence length

# Initialize the grid for inputs
X, Y = np.meshgrid(asteroids_range, maneuver_range)

# Initialize the outputs for both methods
Z_fuzzy = np.zeros_like(X)
Z_non_fuzzy = np.zeros_like(X)

# Define the non-fuzzy fitness calculation as a function
def non_fuzzy_fitness_jagg(asteroids_shot, move_sequence_length_s):
    if asteroids_shot < 0:
        asteroids_fitness = 0
    else:
        asteroids_shot += 0.1
        time_per_asteroids_shot = move_sequence_length_s / asteroids_shot
        asteroids_fitness = max(0, 1 - time_per_asteroids_shot / 15)
    return asteroids_fitness

def non_fuzzy_fitness(asteroids_shot, move_sequence_length_s):
    if asteroids_shot < 0:
        return 0
    else:
        asteroids_shot += 0.1  # Avoid division by zero
        time_per_asteroids_shot = move_sequence_length_s / asteroids_shot
        
        # Sigmoid parameters
        x0 = 13  # Midpoint of the sigmoid, adjust as needed
        k = 0.3  # Steepness of the sigmoid, negative for a decreasing function
        
        # Applying the sigmoid function to smooth the transition
        asteroids_fitness = 1 / (1 + math.exp(k * (time_per_asteroids_shot - x0)))
        
        # Optionally, scale and shift the sigmoid output if needed
        # This example directly uses the sigmoid output, assuming it provides a suitable range
        
    return asteroids_fitness

# Loop over each point in the grid to compute the fuzzy and non-fuzzy fitness
#print(len(asteroids_range) - 2)
print(f"Size of x: {X}")
for j in range(len(asteroids_range)):
    for i in range(len(maneuver_range)):
        print(i, j)
        asteroids_shot = X[i, j]
        move_sequence_length_s = Y[i, j]
        
        # Compute non-fuzzy fitness
        Z_non_fuzzy[i, j] = non_fuzzy_fitness(asteroids_shot, move_sequence_length_s)
        
        # For the fuzzy fitness, assume the fuzzy system (fitness_sim) is already set up as per previous instructions
        print(f"asts: {asteroids_shot}, seq len {move_sequence_length_s}")
        fitness_sim.input['asteroids_destroyed'] = asteroids_shot
        fitness_sim.input['maneuver_sequence_length'] = move_sequence_length_s
        fitness_sim.compute()
        Z_fuzzy[i, j] = fitness_sim.output['fitness']

# Plotting the results
fig = plt.figure(figsize=(20, 10))

# Plot for non-fuzzy approach
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_non_fuzzy, cmap='viridis', edgecolor='none')
ax1.set_title('Non-Fuzzy Fitness Evaluation')
ax1.set_xlabel('Asteroids Destroyed')
ax1.set_ylabel('Maneuver Sequence Length')
ax1.set_zlabel('Fitness')
fig.colorbar(surf1, shrink=0.5, aspect=5, ax=ax1)

# Plot for fuzzy approach
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_fuzzy, cmap='viridis', edgecolor='none')
ax2.set_title('Fuzzy Fitness Evaluation')
ax2.set_xlabel('Asteroids Destroyed')
ax2.set_ylabel('Maneuver Sequence Length')
ax2.set_zlabel('Fitness')
fig.colorbar(surf2, shrink=0.5, aspect=5, ax=ax2)
print(non_fuzzy_fitness(100000000, 0.1))
plt.show()

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time

mines_left = ctrl.Antecedent(np.arange(0, 4, 1), 'mines_left')
lives_left = ctrl.Antecedent(np.arange(1, 4, 1), 'lives_left')
asteroids_hit = ctrl.Antecedent(np.arange(0, 51, 1), 'asteroids_hit')
drop_mine = ctrl.Consequent(np.arange(0, 11, 1), 'drop_mine')

# Defining the membership functions
#mines_left.automf(2, names=['few', 'many'])
mines_left['few'] = fuzz.trimf(mines_left.universe, [1, 1, 2])
mines_left['many'] = fuzz.trimf(mines_left.universe, [1.5, 3, 3])
#lives_left.automf(2, names=['few', 'many'])
lives_left['few'] = fuzz.trimf(lives_left.universe, [1, 1, 2])
lives_left['many'] = fuzz.trimf(lives_left.universe, [1.5, 3, 3])
#asteroids_hit.automf(3, names=['few', 'okay', 'many'])
asteroids_hit_okay_center = 15
asteroids_hit['few'] = fuzz.trimf(asteroids_hit.universe, [0, 0, asteroids_hit_okay_center])
asteroids_hit['okay'] = fuzz.trimf(asteroids_hit.universe, [0, asteroids_hit_okay_center, 50])
asteroids_hit['many'] = fuzz.trimf(asteroids_hit.universe, [asteroids_hit_okay_center, 50, 50])

drop_mine['no'] = fuzz.trimf(drop_mine.universe, [0, 0, 5])
drop_mine['yes'] = fuzz.trimf(drop_mine.universe, [5, 10, 10])

rules = [
    ctrl.Rule(mines_left['few'] & lives_left['few'] & asteroids_hit['few'], drop_mine['no']),
    ctrl.Rule(mines_left['few'] & lives_left['few'] & asteroids_hit['okay'], drop_mine['no']),
    ctrl.Rule(mines_left['few'] & lives_left['few'] & asteroids_hit['many'], drop_mine['yes']),
    ctrl.Rule(mines_left['few'] & lives_left['many'] & asteroids_hit['few'], drop_mine['no']),
    ctrl.Rule(mines_left['few'] & lives_left['many'] & asteroids_hit['okay'], drop_mine['yes']),
    ctrl.Rule(mines_left['few'] & lives_left['many'] & asteroids_hit['many'], drop_mine['yes']),
    ctrl.Rule(mines_left['many'] & lives_left['few'] & asteroids_hit['few'], drop_mine['no']),
    ctrl.Rule(mines_left['many'] & lives_left['few'] & asteroids_hit['okay'], drop_mine['no']),
    ctrl.Rule(mines_left['many'] & lives_left['few'] & asteroids_hit['many'], drop_mine['yes']),
    ctrl.Rule(mines_left['many'] & lives_left['many'] & asteroids_hit['few'], drop_mine['yes']),
    ctrl.Rule(mines_left['many'] & lives_left['many'] & asteroids_hit['okay'], drop_mine['yes']),
    ctrl.Rule(mines_left['many'] & lives_left['many'] & asteroids_hit['many'], drop_mine['yes']),
]

mine_dropping_control = ctrl.ControlSystem(rules)
mine_dropping_fis = ctrl.ControlSystemSimulation(mine_dropping_control)

# Example game state
start = time.time()
mine_dropping_fis.input['mines_left'] = 1
mine_dropping_fis.input['lives_left'] = 1
mine_dropping_fis.input['asteroids_hit'] = 500

# Compute the output
mine_dropping_fis.compute()
end = time.time()
# Interpreting the output
drop_decision = mine_dropping_fis.output['drop_mine']
should_drop_mine = drop_decision > 5  # True for drop, False for don't drop
print(should_drop_mine)
print(f"Took {end - start} s to compute")

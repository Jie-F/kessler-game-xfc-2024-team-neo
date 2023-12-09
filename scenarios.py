from src.kesslergame import Scenario

import numpy as np

# "Simple" Scenarios --------------------------------------------------------------------------------------------------#
# Threat priority tests
threat_test_1 = Scenario(
    name="threat_test_1",
    asteroid_states=[{"position": (0, 300), "angle": -90.0, "speed": 40},
                     {"position": (700, 300), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (600, 300)}],
    seed=0
)

threat_test_2 = Scenario(
    name="threat_test_2",
    asteroid_states=[{"position": (800, 300), "angle": 90.0, "speed": 40},
                     {"position": (100, 300), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (200, 300)}],
    seed=0
)

threat_test_3 = Scenario(
    name="threat_test_3",
    asteroid_states=[{"position": (400, 0), "angle": 0.0, "speed": 40},
                     {"position": (400, 550), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (400, 450)}],
    seed=0
)

threat_test_4 = Scenario(
    name="threat_test_4",
    asteroid_states=[{"position": (400, 600), "angle": 180.0, "speed": 40},
                     {"position": (400, 50), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (400, 150)}],
    seed=0
)

# Accuracy tests

accuracy_test_1 = Scenario(
    name="accuracy_test_1",
    asteroid_states=[{"position": (400, 500), "angle": 90.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (400, 100)}],
    seed=0
)

accuracy_test_2 = Scenario(
    name="accuracy_test_2",
    asteroid_states=[{"position": (400, 500), "angle": -90.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (400, 100)}],
    seed=0
)

accuracy_test_3 = Scenario(
    name="accuracy_test_3",
    asteroid_states=[{"position": (100, 100), "angle": 0.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (400, 100)}],
    seed=0
)

accuracy_test_4 = Scenario(
    name="accuracy_test_4",
    asteroid_states=[{"position": (700, 100), "angle": 0.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (400, 100)}],
    seed=0
)

accuracy_test_5 = Scenario(
    name="accuracy_test_5",
    asteroid_states=[{"position": (100, 500), "angle": 180.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (400, 100)}],
    seed=0
)

accuracy_test_6 = Scenario(
    name="accuracy_test_6",
    asteroid_states=[{"position": (700, 500), "angle": 180.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (400, 100)}],
    seed=0
)

accuracy_test_7 = Scenario(
    name="accuracy_test_7",
    asteroid_states=[{"position": (400, 500), "angle": 180.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (400, 100), "angle": 90.0}],
    seed=0
)

accuracy_test_8 = Scenario(
    name="accuracy_test_8",
    asteroid_states=[{"position": (400, 500), "angle": 180.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (400, 100), "angle": -90.0}],
    seed=0
)

accuracy_test_9 = Scenario(
    name="accuracy_test_9",
    asteroid_states=[{"position": (100, 500), "angle": -135.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (700, 100), "angle": -90.0}],
    seed=0
)

accuracy_test_10 = Scenario(
    name="accuracy_test_10",
    asteroid_states=[{"position": (700, 500), "angle": 135.0, "speed": 120, "size": 1},
                     ],
    ship_states=[{"position": (100, 100), "angle": 90.0}],
    seed=0
)

# "Easy" wall scenario with default ship state, starts on left and moves right
wall_left_easy = Scenario(
    name="wall_left_easy",
    asteroid_states=[{"position": (0, 100), "angle": -90.0, "speed": 60},
                     {"position": (0, 200), "angle": -90.0, "speed": 60},
                     {"position": (0, 300), "angle": -90.0, "speed": 60},
                     {"position": (0, 400), "angle": -90.0, "speed": 60},
                     {"position": (0, 500), "angle": -90.0, "speed": 60},
                     ],
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# "Easy" wall scenario with default ship state, starts on right and moves left
wall_right_easy = Scenario(
    name="wall_right_easy",
    asteroid_states=[{"position": (800, 100), "angle": 90.0, "speed": 60},
                     {"position": (800, 200), "angle": 90.0, "speed": 60},
                     {"position": (800, 300), "angle": 90.0, "speed": 60},
                     {"position": (800, 400), "angle": 90.0, "speed": 60},
                     {"position": (800, 500), "angle": 90.0, "speed": 60},
                     ],
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# "Easy" wall scenario with default ship state, starts at the top and moves downward
wall_top_easy = Scenario(
    name="wall_top_easy",
    asteroid_states=[{"position": (100, 600), "angle": 180.0, "speed": 60},
                     {"position": (200, 600), "angle": 180.0, "speed": 60},
                     {"position": (300, 600), "angle": 180.0, "speed": 60},
                     {"position": (400, 600), "angle": 180.0, "speed": 60},
                     {"position": (500, 600), "angle": 180.0, "speed": 60},
                     {"position": (600, 600), "angle": 180.0, "speed": 60},
                     {"position": (700, 600), "angle": 180.0, "speed": 60},
                     ],
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# "Easy" wall scenario with default ship state, starts at the top and moves downward
wall_bottom_easy = Scenario(
    name="wall_bottom_easy",
    asteroid_states=[{"position": (100, 0), "angle": 0.0, "speed": 60},
                     {"position": (200, 0), "angle": 0.0, "speed": 60},
                     {"position": (300, 0), "angle": 0.0, "speed": 60},
                     {"position": (400, 0), "angle": 0.0, "speed": 60},
                     {"position": (500, 0), "angle": 0.0, "speed": 60},
                     {"position": (600, 0), "angle": 0.0, "speed": 60},
                     {"position": (700, 0), "angle": 0.0, "speed": 60},
                     ],
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# Ring scenarios ------------------------------------------------------------------------------------------------------#
# Scenario where a ring of asteroids close in on the vehicle
# calculating initial states
R = 300
theta = np.linspace(0, 2 * np.pi, 17)[:-1]
ast_x = [R * np.cos(angle) + 400 for angle in theta]
ast_y = [R * np.sin(angle) + 300 for angle in theta]

init_angle = [90 + val * 180 / np.pi for val in theta]
ast_states = []
for ii in range(len(init_angle)):
    ast_states.append({"position": (ast_x[ii], ast_y[ii]), "angle": init_angle[ii], "speed": 30})

ring_closing = Scenario(
    name="ring_closing",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# Static ring scenarios
# Static ring left
R = 150
theta = np.linspace(0, 2 * np.pi, 17)[1:-2]
ast_x = [R * np.cos(angle + np.pi) + 400 for angle in theta]
ast_y = [R * np.sin(angle + np.pi) + 300 for angle in theta]

init_angle = [90 + val * 180 / np.pi for val in theta]
ast_states = []
for ii in range(len(init_angle)):
    ast_states.append({"position": (ast_x[ii], ast_y[ii]), "angle": init_angle[ii], "speed": 0})

ring_static_left = Scenario(
    name="ring_static_left",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# Static ring right
R = 150
theta = np.linspace(0, 2 * np.pi, 17)[1:-2]
ast_x = [R * np.cos(angle) + 400 for angle in theta]
ast_y = [R * np.sin(angle) + 300 for angle in theta]

init_angle = [90 + val * 180 / np.pi for val in theta]
ast_states = []
for ii in range(len(init_angle)):
    ast_states.append({"position": (ast_x[ii], ast_y[ii]), "angle": init_angle[ii], "speed": 0})

ring_static_right = Scenario(
    name="ring_static_right",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# Static ring top
R = 150
theta = np.linspace(0, 2 * np.pi, 17)[1:-2]
ast_x = [R * np.cos(angle + np.pi / 2) + 400 for angle in theta]
ast_y = [R * np.sin(angle + np.pi / 2) + 300 for angle in theta]

init_angle = [90 + val * 180 / np.pi for val in theta]
ast_states = []
for ii in range(len(init_angle)):
    ast_states.append({"position": (ast_x[ii], ast_y[ii]), "angle": init_angle[ii], "speed": 0})

ring_static_top = Scenario(
    name="ring_static_top",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# Static ring bottom
R = 150
theta = np.linspace(0, 2 * np.pi, 17)[1:-2]
ast_x = [R * np.cos(angle + 3 * np.pi / 2) + 400 for angle in theta]
ast_y = [R * np.sin(angle + 3 * np.pi / 2) + 300 for angle in theta]

init_angle = [90 + val * 180 / np.pi for val in theta]
ast_states = []
for ii in range(len(init_angle)):
    ast_states.append({"position": (ast_x[ii], ast_y[ii]), "angle": init_angle[ii], "speed": 0})

ring_static_bottom = Scenario(
    name="ring_static_bottom",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 300)}],
    seed=0
)
# ---------------------------------------------------------------------------------------------------------------------#


# Normal corridor scenarios -------------------------------------------------------------------------------------------#
# Scenario where ship is in a corridor and forced to shoot its way through
# calculating corridor states
num_x = 17
num_y = 10
x = np.linspace(0, 800, num_x)
y = np.concatenate((np.linspace(0, 200, int(num_y / 2)), np.linspace(400, 600, int(num_y / 2))))

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": 0.0, "speed": 0})

# calculate wall asteroid states
ast_states.append({"position": (50, 266), "angle": -90.0, "speed": 0})
ast_states.append({"position": (50, 332), "angle": -90.0, "speed": 0})

corridor_left = Scenario(
    name="corridor_left",
    asteroid_states=ast_states,
    ship_states=[{"position": (700, 300)}],
    seed=0
)

# calculate wall asteroid states
ast_states = ast_states[:-2]
ast_states.append({"position": (800, 266), "angle": 90.0, "speed": 20})
ast_states.append({"position": (800, 332), "angle": 90.0, "speed": 20})

corridor_right = Scenario(
    name="corridor_right",
    asteroid_states=ast_states,
    ship_states=[{"position": (100, 300)}],
    seed=0
)

# Corridor top scenario
num_x = 14
num_y = 13

x = np.concatenate((np.linspace(0, 300, int(num_x / 2)), np.linspace(500, 800, int(num_x / 2))))
y = np.linspace(0, 600, num_y)

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": 0.0, "speed": 0})

# calculate wall asteroid states
ast_states.append({"position": (366, 600), "angle": 180.0, "speed": 20})
ast_states.append({"position": (432, 600), "angle": 180.0, "speed": 20})

corridor_top = Scenario(
    name="corridor_top",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 100)}],
    seed=0
)

# Corridor bottom scenario
# calculate wall asteroid states
ast_states = ast_states[:-2]
ast_states.append({"position": (366, 0), "angle": 0.0, "speed": 20})
ast_states.append({"position": (432, 0), "angle": 0.0, "speed": 20})

corridor_bottom = Scenario(
    name="corridor_bottom",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 500)}],
    seed=0
)
# ---------------------------------------------------------------------------------------------------------------------#


# Moving Corridor Scenarios -------------------------------------------------------------------------------------------#
# Corridor moving right
# calculating corridor states
num_x = 17
num_y = 10
x = np.linspace(0, 800, num_x)
y = np.concatenate((np.linspace(0, 200, int(num_y / 2)), np.linspace(400, 600, int(num_y / 2))))

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": -90.0, "speed": 120})

moving_corridor_1 = Scenario(
    name="moving_corridor_1",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 300), "angle": 90}],
    seed=0
)

# Corridor moving left
# calculating corridor states
num_x = 17
num_y = 10
x = np.linspace(0, 800, num_x)
y = np.concatenate((np.linspace(0, 200, int(num_y / 2)), np.linspace(400, 600, int(num_y / 2))))

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": 90.0, "speed": 120})

moving_corridor_2 = Scenario(
    name="moving_corridor_2",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 300), "angle": -90}],
    seed=0
)

# Corridor moving down
# calculating corridor states
num_x = 14
num_y = 13

x = np.concatenate((np.linspace(0, 300, int(num_x / 2)), np.linspace(500, 800, int(num_x / 2))))
y = np.linspace(0, 600, num_y)

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": 180.0, "speed": 120})

moving_corridor_3 = Scenario(
    name="moving_corridor_3",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 300), "angle": 0}],
    seed=0
)

# Corridor moving up
# calculating corridor states
num_x = 14
num_y = 13

x = np.concatenate((np.linspace(0, 300, int(num_x / 2)), np.linspace(500, 800, int(num_x / 2))))
y = np.linspace(0, 600, num_y)

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": 0.0, "speed": 120})

moving_corridor_4 = Scenario(
    name="moving_corridor_4",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 300), "angle": 180, "lives": 30, "mines_remaining": 30}],
    seed=0
)

# Angled corridor scenario 1
# calculating corridor states
num_x = 17
num_y = 13
x = np.linspace(0, 800, num_x)
y = np.linspace(0, 600, num_y)

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        if not (abs(1.5 * ast_x[ii, jj] - ast_y[ii, jj]) <= 160) and not (
                abs(-1.5 * ast_x[ii, jj] + 1200 - ast_y[ii, jj]) <= 160):
            ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": -90.0, "speed": 30})

moving_corridor_angled_1 = Scenario(
    name="moving_corridor_angled_1",
    asteroid_states=ast_states,
    ship_states=[{"position": (750, 50), "angle": 90, "lives": 30, "mines_remaining": 30}],
    seed=0
)

# Angled corridor scenario 2
# calculating corridor states
num_x = 17
num_y = 13
x = np.linspace(0, 800, num_x)
y = np.linspace(0, 600, num_y)

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        if not (abs(-1.5 * ast_x[ii, jj] + 600 - ast_y[ii, jj]) <= 160) and not (
                abs(1.5 * ast_x[ii, jj] - 600 - ast_y[ii, jj]) <= 160):
            ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": -90.0, "speed": 30})

moving_corridor_angled_2 = Scenario(
    name="moving_corridor_angled_2",
    asteroid_states=ast_states,
    ship_states=[{"position": (750, 550), "angle": 90, "lives": 30, "mines_remaining": 30}],
    seed=0
)

# Curved corridor scenario 1
# calculating corridor states
num_x = 17
num_y = 13
x = np.linspace(0, 800, num_x)
y = np.linspace(0, 600, num_y)

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        if not (abs(-(1 / 300) * (ast_x[ii, jj] - 400) ** 2 + 600 - ast_y[ii, jj]) <= 200):
            ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": -90.0, "speed": 30})

moving_corridor_curve_1 = Scenario(
    name="moving_corridor_curve_1",
    asteroid_states=ast_states,
    ship_states=[{"position": (550, 500), "angle": 90, "lives": 30, "mines_remaining": 30}],
    seed=0
)

# Curved corridor scenario 2
# calculating corridor states
num_x = 30
num_y = 45
x = np.linspace(0, 800, num_x)
y = np.linspace(0, 600, num_y)

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        if not (abs((1 / 300) * (ast_x[ii, jj] - 400) ** 2 - ast_y[ii, jj]) <= 200) and not (
                abs((1 / 300) * (ast_x[ii, jj] - 400) ** 2 - ast_y[ii, jj]) >= 300):
            ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]), "angle": -90.0, "speed": 120, "size": 1})

moving_corridor_curve_2 = Scenario(
    name="moving_corridor_curve_2",
    asteroid_states=ast_states,
    ship_states=[{"position": (550, 100), "angle": 90, "lives": 30, "mines_remaining": 30}],
    seed=0
)
# ---------------------------------------------------------------------------------------------------------------------#


# Apocalypse scenarios-------------------------------------------------------------------------------------------------#
# Scenario meant to be difficult, probably can't be totally cleared
# currently the vehicle spawns on top of asteroids. It won't kill the vehicle until you fire though
scenario_apocalypse_1 = Scenario(name="apocalypse_1", num_asteroids=50, seed=1)
# ---------------------------------------------------------------------------------------------------------------------#


# Forcing wrap scenarios-----------------------------------------------------------------------------------------------#
# Wrap right scenarios
wall_right_wrap_1 = Scenario(
    name="wall_right_wrap_1",
    asteroid_states=[{"position": (600, 0), "angle": -90.0, "speed": 80},
                     {"position": (600, 100), "angle": -90.0, "speed": 80},
                     {"position": (600, 200), "angle": -90.0, "speed": 80},
                     {"position": (600, 300), "angle": -90.0, "speed": 80},
                     {"position": (600, 400), "angle": -90.0, "speed": 80},
                     {"position": (600, 500), "angle": -90.0, "speed": 80},
                     {"position": (600, 600), "angle": -90.0, "speed": 80},
                     ],
    ship_states=[{"position": (750, 300), "lives": 30, "mines_remaining": 30}],
    seed=0
)

wall_right_wrap_2 = Scenario(
    name="wall_right_wrap_2",
    asteroid_states=[{"position": (750, 0), "angle": -90.0, "speed": 80},
                     {"position": (750, 100), "angle": -90.0, "speed": 80},
                     {"position": (750, 200), "angle": -90.0, "speed": 80},
                     {"position": (750, 300), "angle": -90.0, "speed": 80},
                     {"position": (750, 400), "angle": -90.0, "speed": 80},
                     {"position": (750, 500), "angle": -90.0, "speed": 80},
                     {"position": (750, 600), "angle": -90.0, "speed": 80},
                     ],
    ship_states=[{"position": (50, 300), "lives": 30, "mines_remaining": 30}],
    seed=0
)

wall_right_wrap_3 = Scenario(
    name="wall_right_wrap_3",
    asteroid_states=[{"position": (600, 0), "angle": -90.0, "speed": 80},
                     {"position": (600, 100), "angle": -90.0, "speed": 80},
                     {"position": (600, 200), "angle": -90.0, "speed": 80},
                     {"position": (600, 300), "angle": -90.0, "speed": 80},
                     {"position": (600, 400), "angle": -90.0, "speed": 80},
                     {"position": (600, 500), "angle": -90.0, "speed": 80},
                     {"position": (600, 600), "angle": -90.0, "speed": 80},
                     {"position": (200, 0), "angle": -90.0, "speed": 0},
                     {"position": (200, 100), "angle": -90.0, "speed": 0},
                     {"position": (200, 200), "angle": -90.0, "speed": 0},
                     {"position": (200, 300), "angle": -90.0, "speed": 0},
                     {"position": (200, 400), "angle": -90.0, "speed": 0},
                     {"position": (200, 500), "angle": -90.0, "speed": 0},
                     {"position": (200, 600), "angle": -90.0, "speed": 0},
                     ],
    ship_states=[{"position": (750, 300), "lives": 30, "mines_remaining": 30}],
    seed=0
)

wall_right_wrap_4 = Scenario(
    name="wall_right_wrap_4",
    asteroid_states=[{"position": (750, 0), "angle": -90.0, "speed": 80},
                     {"position": (750, 100), "angle": -90.0, "speed": 80},
                     {"position": (750, 200), "angle": -90.0, "speed": 80},
                     {"position": (750, 300), "angle": -90.0, "speed": 80},
                     {"position": (750, 400), "angle": -90.0, "speed": 80},
                     {"position": (750, 500), "angle": -90.0, "speed": 80},
                     {"position": (750, 600), "angle": -90.0, "speed": 80},
                     {"position": (200, 0), "angle": -90.0, "speed": 0},
                     {"position": (200, 100), "angle": -90.0, "speed": 0},
                     {"position": (200, 200), "angle": -90.0, "speed": 0},
                     {"position": (200, 300), "angle": -90.0, "speed": 0},
                     {"position": (200, 400), "angle": -90.0, "speed": 0},
                     {"position": (200, 500), "angle": -90.0, "speed": 0},
                     {"position": (200, 600), "angle": -90.0, "speed": 0},
                     ],
    ship_states=[{"position": (50, 300), "lives": 30, "mines_remaining": 30}],
    seed=0
)

# Wrap left scenarios
wall_left_wrap_1 = Scenario(
    name="wall_left_wrap_1",
    asteroid_states=[{"position": (200, 0), "angle": 90.0, "speed": 80},
                     {"position": (200, 100), "angle": 90.0, "speed": 80},
                     {"position": (200, 200), "angle": 90.0, "speed": 80},
                     {"position": (200, 300), "angle": 90.0, "speed": 80},
                     {"position": (200, 400), "angle": 90.0, "speed": 80},
                     {"position": (200, 500), "angle": 90.0, "speed": 80},
                     {"position": (200, 600), "angle": 90.0, "speed": 80},
                     ],
    ship_states=[{"position": (50, 300)}],
    seed=0
)

wall_left_wrap_2 = Scenario(
    name="wall_left_wrap_2",
    asteroid_states=[{"position": (50, 0), "angle": 90.0, "speed": 80},
                     {"position": (50, 100), "angle": 90.0, "speed": 80},
                     {"position": (50, 200), "angle": 90.0, "speed": 80},
                     {"position": (50, 300), "angle": 90.0, "speed": 80},
                     {"position": (50, 400), "angle": 90.0, "speed": 80},
                     {"position": (50, 500), "angle": 90.0, "speed": 80},
                     {"position": (50, 600), "angle": 90.0, "speed": 80},
                     ],
    ship_states=[{"position": (750, 300)}],
    seed=0
)

wall_left_wrap_3 = Scenario(
    name="wall_left_wrap_3",
    asteroid_states=[{"position": (200, 0), "angle": 90.0, "speed": 80},
                     {"position": (200, 100), "angle": 90.0, "speed": 80},
                     {"position": (200, 200), "angle": 90.0, "speed": 80},
                     {"position": (200, 300), "angle": 90.0, "speed": 80},
                     {"position": (200, 400), "angle": 90.0, "speed": 80},
                     {"position": (200, 500), "angle": 90.0, "speed": 80},
                     {"position": (200, 600), "angle": 90.0, "speed": 80},
                     {"position": (600, 0), "angle": -90.0, "speed": 0},
                     {"position": (600, 100), "angle": -90.0, "speed": 0},
                     {"position": (600, 200), "angle": -90.0, "speed": 0},
                     {"position": (600, 300), "angle": -90.0, "speed": 0},
                     {"position": (600, 400), "angle": -90.0, "speed": 0},
                     {"position": (600, 500), "angle": -90.0, "speed": 0},
                     {"position": (600, 600), "angle": -90.0, "speed": 0},
                     ],
    ship_states=[{"position": (50, 300)}],
    seed=0
)

wall_left_wrap_4 = Scenario(
    name="wall_left_wrap_4",
    asteroid_states=[{"position": (50, 0), "angle": 90.0, "speed": 80},
                     {"position": (50, 100), "angle": 90.0, "speed": 80},
                     {"position": (50, 200), "angle": 90.0, "speed": 80},
                     {"position": (50, 300), "angle": 90.0, "speed": 80},
                     {"position": (50, 400), "angle": 90.0, "speed": 80},
                     {"position": (50, 500), "angle": 90.0, "speed": 80},
                     {"position": (50, 600), "angle": 90.0, "speed": 80},
                     {"position": (600, 0), "angle": -90.0, "speed": 0},
                     {"position": (600, 100), "angle": -90.0, "speed": 0},
                     {"position": (600, 200), "angle": -90.0, "speed": 0},
                     {"position": (600, 300), "angle": -90.0, "speed": 0},
                     {"position": (600, 400), "angle": -90.0, "speed": 0},
                     {"position": (600, 500), "angle": -90.0, "speed": 0},
                     {"position": (600, 600), "angle": -90.0, "speed": 0},
                     ],
    ship_states=[{"position": (750, 300)}],
    seed=0
)

# Wrap top scenarios
wall_top_wrap_1 = Scenario(
    name="wall_top_wrap_1",
    asteroid_states=[{"position": (0, 400), "angle": 0.0, "speed": 80},
                     {"position": (100, 400), "angle": 0.0, "speed": 80},
                     {"position": (200, 400), "angle": 0.0, "speed": 80},
                     {"position": (300, 400), "angle": 0.0, "speed": 80},
                     {"position": (400, 400), "angle": 0.0, "speed": 80},
                     {"position": (500, 400), "angle": 0.0, "speed": 80},
                     {"position": (600, 400), "angle": 0.0, "speed": 80},
                     {"position": (700, 400), "angle": 0.0, "speed": 80},
                     {"position": (800, 400), "angle": 0.0, "speed": 80},
                     ],
    ship_states=[{"position": (400, 550)}],
    seed=0
)

wall_top_wrap_2 = Scenario(
    name="wall_top_wrap_2",
    asteroid_states=[{"position": (0, 400), "angle": 0.0, "speed": 80},
                     {"position": (100, 400), "angle": 0.0, "speed": 80},
                     {"position": (200, 400), "angle": 0.0, "speed": 80},
                     {"position": (300, 400), "angle": 0.0, "speed": 80},
                     {"position": (400, 400), "angle": 0.0, "speed": 80},
                     {"position": (500, 400), "angle": 0.0, "speed": 80},
                     {"position": (600, 400), "angle": 0.0, "speed": 80},
                     {"position": (700, 400), "angle": 0.0, "speed": 80},
                     {"position": (800, 400), "angle": 0.0, "speed": 80},
                     ],
    ship_states=[{"position": (400, 50)}],
    seed=0
)

wall_top_wrap_3 = Scenario(
    name="wall_top_wrap_3",
    asteroid_states=[{"position": (0, 400), "angle": 0.0, "speed": 80},
                     {"position": (100, 400), "angle": 0.0, "speed": 80},
                     {"position": (200, 400), "angle": 0.0, "speed": 80},
                     {"position": (300, 400), "angle": 0.0, "speed": 80},
                     {"position": (400, 400), "angle": 0.0, "speed": 80},
                     {"position": (500, 400), "angle": 0.0, "speed": 80},
                     {"position": (600, 400), "angle": 0.0, "speed": 80},
                     {"position": (700, 400), "angle": 0.0, "speed": 80},
                     {"position": (800, 400), "angle": 0.0, "speed": 80},
                     {"position": (0, 200), "angle": 0.0, "speed": 0},
                     {"position": (100, 200), "angle": 0.0, "speed": 0},
                     {"position": (200, 200), "angle": 0.0, "speed": 0},
                     {"position": (300, 200), "angle": 0.0, "speed": 0},
                     {"position": (400, 200), "angle": 0.0, "speed": 0},
                     {"position": (500, 200), "angle": 0.0, "speed": 0},
                     {"position": (600, 200), "angle": 0.0, "speed": 0},
                     {"position": (700, 200), "angle": 0.0, "speed": 0},
                     {"position": (800, 200), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (400, 550)}],
    seed=0
)

wall_top_wrap_4 = Scenario(
    name="wall_top_wrap_4",
    asteroid_states=[{"position": (0, 400), "angle": 0.0, "speed": 80},
                     {"position": (100, 400), "angle": 0.0, "speed": 80},
                     {"position": (200, 400), "angle": 0.0, "speed": 80},
                     {"position": (300, 400), "angle": 0.0, "speed": 80},
                     {"position": (400, 400), "angle": 0.0, "speed": 80},
                     {"position": (500, 400), "angle": 0.0, "speed": 80},
                     {"position": (600, 400), "angle": 0.0, "speed": 80},
                     {"position": (700, 400), "angle": 0.0, "speed": 80},
                     {"position": (800, 400), "angle": 0.0, "speed": 80},
                     {"position": (0, 200), "angle": 0.0, "speed": 0},
                     {"position": (100, 200), "angle": 0.0, "speed": 0},
                     {"position": (200, 200), "angle": 0.0, "speed": 0},
                     {"position": (300, 200), "angle": 0.0, "speed": 0},
                     {"position": (400, 200), "angle": 0.0, "speed": 0},
                     {"position": (500, 200), "angle": 0.0, "speed": 0},
                     {"position": (600, 200), "angle": 0.0, "speed": 0},
                     {"position": (700, 200), "angle": 0.0, "speed": 0},
                     {"position": (800, 200), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (400, 50)}],
    seed=0
)

# Wrap bottom scenarios
wall_bottom_wrap_1 = Scenario(
    name="wall_bottom_wrap_1",
    asteroid_states=[{"position": (0, 200), "angle": 180.0, "speed": 80},
                     {"position": (100, 200), "angle": 180.0, "speed": 80},
                     {"position": (200, 200), "angle": 180.0, "speed": 80},
                     {"position": (300, 200), "angle": 180.0, "speed": 80},
                     {"position": (400, 200), "angle": 180.0, "speed": 80},
                     {"position": (500, 200), "angle": 180.0, "speed": 80},
                     {"position": (600, 200), "angle": 180.0, "speed": 80},
                     {"position": (700, 200), "angle": 180.0, "speed": 80},
                     {"position": (800, 200), "angle": 180.0, "speed": 80},
                     ],
    ship_states=[{"position": (400, 50)}],
    seed=0
)

wall_bottom_wrap_2 = Scenario(
    name="wall_bottom_wrap_2",
    asteroid_states=[{"position": (0, 200), "angle": 180.0, "speed": 80},
                     {"position": (100, 200), "angle": 180.0, "speed": 80},
                     {"position": (200, 200), "angle": 180.0, "speed": 80},
                     {"position": (300, 200), "angle": 180.0, "speed": 80},
                     {"position": (400, 200), "angle": 180.0, "speed": 80},
                     {"position": (500, 200), "angle": 180.0, "speed": 80},
                     {"position": (600, 200), "angle": 180.0, "speed": 80},
                     {"position": (700, 200), "angle": 180.0, "speed": 80},
                     {"position": (800, 200), "angle": 180.0, "speed": 80},
                     ],
    ship_states=[{"position": (400, 550)}],
    seed=0
)

wall_bottom_wrap_3 = Scenario(
    name="wall_bottom_wrap_3",
    asteroid_states=[{"position": (0, 200), "angle": 180.0, "speed": 80},
                     {"position": (100, 200), "angle": 180.0, "speed": 80},
                     {"position": (200, 200), "angle": 180.0, "speed": 80},
                     {"position": (300, 200), "angle": 180.0, "speed": 80},
                     {"position": (400, 200), "angle": 180.0, "speed": 80},
                     {"position": (500, 200), "angle": 180.0, "speed": 80},
                     {"position": (600, 200), "angle": 180.0, "speed": 80},
                     {"position": (700, 200), "angle": 180.0, "speed": 80},
                     {"position": (800, 200), "angle": 180.0, "speed": 80},
                     {"position": (0, 400), "angle": 0.0, "speed": 0},
                     {"position": (100, 400), "angle": 0.0, "speed": 0},
                     {"position": (200, 400), "angle": 0.0, "speed": 0},
                     {"position": (300, 400), "angle": 0.0, "speed": 0},
                     {"position": (400, 400), "angle": 0.0, "speed": 0},
                     {"position": (500, 400), "angle": 0.0, "speed": 0},
                     {"position": (600, 400), "angle": 0.0, "speed": 0},
                     {"position": (700, 400), "angle": 0.0, "speed": 0},
                     {"position": (800, 400), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (400, 50)}],
    seed=0
)

wall_bottom_wrap_4 = Scenario(
    name="wall_bottom_wrap_4",
    asteroid_states=[{"position": (0, 200), "angle": 180.0, "speed": 80},
                     {"position": (100, 200), "angle": 180.0, "speed": 80},
                     {"position": (200, 200), "angle": 180.0, "speed": 80},
                     {"position": (300, 200), "angle": 180.0, "speed": 80},
                     {"position": (400, 200), "angle": 180.0, "speed": 80},
                     {"position": (500, 200), "angle": 180.0, "speed": 80},
                     {"position": (600, 200), "angle": 180.0, "speed": 80},
                     {"position": (700, 200), "angle": 180.0, "speed": 80},
                     {"position": (800, 200), "angle": 180.0, "speed": 80},
                     {"position": (0, 400), "angle": 0.0, "speed": 0},
                     {"position": (100, 400), "angle": 0.0, "speed": 0},
                     {"position": (200, 400), "angle": 0.0, "speed": 0},
                     {"position": (300, 400), "angle": 0.0, "speed": 0},
                     {"position": (400, 400), "angle": 0.0, "speed": 0},
                     {"position": (500, 400), "angle": 0.0, "speed": 0},
                     {"position": (600, 400), "angle": 0.0, "speed": 0},
                     {"position": (700, 400), "angle": 0.0, "speed": 0},
                     {"position": (800, 400), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (400, 550)}],
    seed=0
)

# A scenario with a big non moving box
scenario_big_box = Scenario(
    name="big_box",
    asteroid_states=[{"position": (100, 600), "angle": 0.0, "speed": 0},
                     {"position": (200, 600), "angle": 0.0, "speed": 0},
                     {"position": (300, 600), "angle": 0.0, "speed": 0},
                     {"position": (400, 600), "angle": 0.0, "speed": 0},
                     {"position": (500, 600), "angle": 0.0, "speed": 0},
                     {"position": (600, 600), "angle": 0.0, "speed": 0},
                     {"position": (700, 600), "angle": 0.0, "speed": 0},
                     {"position": (100, 0), "angle": 0.0, "speed": 0},
                     {"position": (200, 0), "angle": 0.0, "speed": 0},
                     {"position": (300, 0), "angle": 0.0, "speed": 0},
                     {"position": (400, 0), "angle": 0.0, "speed": 0},
                     {"position": (500, 0), "angle": 0.0, "speed": 0},
                     {"position": (600, 0), "angle": 0.0, "speed": 0},
                     {"position": (700, 0), "angle": 0.0, "speed": 0},
                     {"position": (800, 0), "angle": 0.0, "speed": 0},
                     {"position": (0, 0), "angle": 0.0, "speed": 0},
                     {"position": (0, 100), "angle": 0.0, "speed": 0},
                     {"position": (0, 200), "angle": 0.0, "speed": 0},
                     {"position": (0, 300), "angle": 0.0, "speed": 0},
                     {"position": (0, 400), "angle": 0.0, "speed": 0},
                     {"position": (0, 500), "angle": 0.0, "speed": 0},
                     {"position": (0, 600), "angle": 0.0, "speed": 0},
                     {"position": (800, 100), "angle": 0.0, "speed": 0},
                     {"position": (800, 200), "angle": 0.0, "speed": 0},
                     {"position": (800, 300), "angle": 0.0, "speed": 0},
                     {"position": (800, 400), "angle": 0.0, "speed": 0},
                     {"position": (800, 500), "angle": 0.0, "speed": 0},
                     {"position": (800, 600), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# A scenario with a little non moving box
scenario_small_box = Scenario(
    name="small_box",
    asteroid_states=[{"position": (200, 500), "angle": 0.0, "speed": 0},
                     {"position": (300, 500), "angle": 0.0, "speed": 0},
                     {"position": (400, 500), "angle": 0.0, "speed": 0},
                     {"position": (500, 500), "angle": 0.0, "speed": 0},

                     {"position": (200, 100), "angle": 0.0, "speed": 0},
                     {"position": (300, 100), "angle": 0.0, "speed": 0},
                     {"position": (400, 100), "angle": 0.0, "speed": 0},
                     {"position": (500, 100), "angle": 0.0, "speed": 0},
                     {"position": (600, 100), "angle": 0.0, "speed": 0},

                     {"position": (200, 200), "angle": 0.0, "speed": 0},
                     {"position": (200, 300), "angle": 0.0, "speed": 0},
                     {"position": (200, 400), "angle": 0.0, "speed": 0},

                     {"position": (600, 200), "angle": 0.0, "speed": 0},
                     {"position": (600, 300), "angle": 0.0, "speed": 0},
                     {"position": (600, 400), "angle": 0.0, "speed": 0},
                     {"position": (600, 500), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (400, 300)}],
    seed=0
)

# A scenario with a big non moving box

scenario_2_still_corridors = Scenario(
    name="scenario_2_still_corridors",
    asteroid_states=[{"position": (0, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (50, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (100, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (150, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (200, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (250, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (300, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 250), "angle": 0.0, "speed": 0, "size": 2},

                     {"position": (0, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (50, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (100, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (150, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (200, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (250, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (300, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 350), "angle": 0.0, "speed": 0, "size": 2},

                     {"position": (450, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (500, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (550, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (600, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (650, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (700, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (750, 250), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (800, 250), "angle": 0.0, "speed": 0, "size": 2},

                     {"position": (450, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (500, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (550, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (600, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (650, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (700, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (750, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (800, 350), "angle": 0.0, "speed": 0, "size": 2},

                     {"position": (350, 0), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 50), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 100), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 150), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 200), "angle": 0.0, "speed": 0, "size": 2},

                     {"position": (450, 0), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (450, 50), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (450, 100), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (450, 150), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (450, 200), "angle": 0.0, "speed": 0, "size": 2},

                     {"position": (350, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 400), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 450), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 500), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 550), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (350, 600), "angle": 0.0, "speed": 0, "size": 2},

                     {"position": (450, 350), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (450, 400), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (450, 450), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (450, 500), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (450, 550), "angle": 0.0, "speed": 0, "size": 2},
                     {"position": (450, 600), "angle": 0.0, "speed": 0, "size": 2},
                     ],
    ship_states=[{"position": (400, 300)}],
    seed=0
)

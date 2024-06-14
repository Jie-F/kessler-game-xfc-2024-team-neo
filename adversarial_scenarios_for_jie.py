from kesslergame.scenario import Scenario
import numpy as np

adv_random_small_1 = Scenario(
    name="adv_random_small_1",
    asteroid_states=[{"position": (232, 464), "angle": 159, "speed": 32, "size": 4},
                     {"position": (846, 413), "angle": 257, "speed": 109, "size": 2},
                     {"position": (101, 657), "angle": 1, "speed": 151, "size": 1},
                     ],
    ship_states=[{"position": (100, 400), "angle": 180, "lives": 3, "team": 1, "mines_remaining": 3},
                 {"position": (700, 300), "angle": 0, "lives": 3, "team": 2, "mines_remaining": 3},
                 ],
    ammo_limit_multiplier=0.75,
    stop_if_no_ammo=True,
    time_limit=30
)

adv_random_small_1_2 = Scenario(
    name="adv_random_small_1_2",
    asteroid_states=[{"position": (232, 464), "angle": 159, "speed": 32, "size": 4},
                     {"position": (846, 413), "angle": 257, "speed": 109, "size": 2},
                     {"position": (101, 657), "angle": 1, "speed": 151, "size": 1},
                     ],
    ship_states=[{"position": (700, 300), "angle": 180, "lives": 3, "team": 1, "mines_remaining": 3},
                 {"position": (100, 400), "angle": 0, "lives": 3, "team": 2, "mines_remaining": 3},
                 ],
    ammo_limit_multiplier=0.75,
    stop_if_no_ammo=True,
    time_limit=30
)

adv_multi_wall_left_easy = Scenario(
    name="multi_wall_left_easy",
    asteroid_states=[{"position": (0, 100), "angle": 0.0, "speed": 60},
                     {"position": (0, 200), "angle": 0.0, "speed": 60},
                     {"position": (0, 300), "angle": 0.0, "speed": 60},
                     {"position": (0, 400), "angle": 0.0, "speed": 60},
                     {"position": (0, 500), "angle": 0.0, "speed": 60},
                     {"position": (0, 600), "angle": 0.0, "speed": 60},
                     {"position": (0, 700), "angle": 0.0, "speed": 60},
                     ],
    ship_states=[{"position": (600, 200), "team": 1, "mines_remaining": 3},
                 {"position": (600, 600), "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=30, ammo_limit_multiplier=1.2, stop_if_no_ammo=True
)

adv_multi_four_corners = Scenario(
    name="multi_four_corners",
    asteroid_states=[{"position": (50, 50), "angle": 270.0, "speed": 0, "size": 4},
                     {"position": (50, 750), "angle": 270.0, "speed": 0, "size": 4},
                     {"position": (950, 750), "angle": 270.0, "speed": 0, "size": 4},
                     {"position": (950, 50), "angle": 270.0, "speed": 0, "size": 4},
                     ],
    ship_states=[{"position": (200, 400), "angle": 0.0, "team": 1, "mines_remaining": 3},
                 {"position": (600, 400), "angle": 0.0, "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=30, ammo_limit_multiplier=0.75, stop_if_no_ammo=True)

adv_multi_wall_top_easy = Scenario(
    name="multi_wall_top_easy",
    asteroid_states=[{"position": (100, 800), "angle": 90.0, "speed": 60},
                     {"position": (200, 800), "angle": 90.0, "speed": 60},
                     {"position": (300, 800), "angle": 90.0, "speed": 60},
                     {"position": (400, 800), "angle": 90.0, "speed": 60},
                     {"position": (500, 800), "angle": 90.0, "speed": 60},
                     {"position": (600, 800), "angle": 90.0, "speed": 60},
                     {"position": (700, 800), "angle": 90.0, "speed": 60},
                     {"position": (800, 800), "angle": 90.0, "speed": 60},
                     {"position": (900, 800), "angle": 90.0, "speed": 60},
                     ],
    ship_states=[{"position": (250, 200), "team": 1, "mines_remaining": 3},
                 {"position": (750, 200), "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=30
)

adv_multi_2wall_closing = Scenario(
    name="adv_multi_2wall_closing",
    asteroid_states=[{"position": (0, 50), "angle": 0.0, "speed": 60},
                     {"position": (0, 150), "angle": 0.0, "speed": 60},
                     {"position": (0, 250), "angle": 0.0, "speed": 60},
                     {"position": (0, 350), "angle": 0.0, "speed": 60},
                     {"position": (0, 450), "angle": 0.0, "speed": 60},
                     {"position": (0, 550), "angle": 0.0, "speed": 60},
                     {"position": (0, 650), "angle": 0.0, "speed": 60},
                     {"position": (0, 750), "angle": 0.0, "speed": 60},
                     {"position": (1000, 50), "angle": 180.0, "speed": 60},
                     {"position": (1000, 150), "angle": 180.0, "speed": 60},
                     {"position": (1000, 250), "angle": 180.0, "speed": 60},
                     {"position": (1000, 350), "angle": 180.0, "speed": 60},
                     {"position": (1000, 450), "angle": 180.0, "speed": 60},
                     {"position": (1000, 550), "angle": 180.0, "speed": 60},
                     {"position": (1000, 650), "angle": 180.0, "speed": 60},
                     {"position": (1000, 750), "angle": 180.0, "speed": 60},
                     ],
    ship_states=[{"position": (500, 300), "angle": 90, "team": 1, "mines_remaining": 3},
                 {"position": (500, 500), "angle": 270, "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=30, ammo_limit_multiplier=1.2, stop_if_no_ammo=True
)

adv_wall_bottom_staggered = Scenario(
    name="adv_wall_staggered",
    asteroid_states=[{"position": (100, 0), "angle": -90.0, "speed": 60},
                     {"position": (200, 50), "angle": -90.0, "speed": 60},
                     {"position": (300, 0), "angle": -90.0, "speed": 60},
                     {"position": (400, 50), "angle": -90.0, "speed": 60},
                     {"position": (500, 0), "angle": -90.0, "speed": 60},
                     {"position": (600, 50), "angle": -90.0, "speed": 60},
                     {"position": (700, 0), "angle": -90.0, "speed": 60},
                     {"position": (800, 50), "angle": -90.0, "speed": 60},
                     {"position": (900, 0), "angle": -90.0, "speed": 60},
                     ],
    ship_states=[{"position": (100, 400), "angle": 180, "lives": 3, "team": 1, "mines_remaining": 3},
                 {"position": (700, 400), "angle": 0, "lives": 3, "team": 2, "mines_remaining": 3},
                 ],
    ammo_limit_multiplier=0.1,
    stop_if_no_ammo=True,
    time_limit=45,
)

adv_multi_wall_right_hard = Scenario(
    name="multi_wall_right_hard",
    asteroid_states=[{"position": (800, 100), "angle": 180.0, "speed": 150},
                     {"position": (800, 200), "angle": 180.0, "speed": 150},
                     {"position": (800, 300), "angle": 180.0, "speed": 150},
                     {"position": (800, 400), "angle": 180.0, "speed": 150},
                     {"position": (800, 500), "angle": 180.0, "speed": 150},
                     {"position": (800, 600), "angle": 180.0, "speed": 150},
                     {"position": (800, 700), "angle": 180.0, "speed": 150},
                     ],
    ship_states=[{"position": (200, 200), "team": 1, "mines_remaining": 3},
                 {"position": (200, 600), "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=30
)

# Angled corridor scenario 1
# calculating corridor states
num_x = 17
num_y = 13
x = np.linspace(0, 1000, num_x)
y = np.linspace(0, 800, num_y)

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        if not (abs(1.6 * ast_x[ii, jj] - ast_y[ii, jj]) <= 160) and not (
                abs(-1.6 * ast_x[ii, jj] + 1600 - ast_y[ii, jj]) <= 160):
            ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]+50), "angle": 0.0, "speed": 90, "size": 2})

adv_moving_corridor_angled_1 = Scenario(
    name="adv_moving_corridor_angled_1",
    asteroid_states=ast_states,
    ship_states=[{"position": (230, 400), "angle": 180, "team": 1, "mines_remaining": 3},
                 {"position": (770, 400), "angle": 180, "team": 2, "mines_remaining": 3}],
    seed=0, time_limit=40,
)

# Angled corridor scenario 2
# calculating corridor states
num_x = 17
num_y = 13
x = np.linspace(0, 1000, num_x)
y = np.linspace(0, 800, num_y)

ast_x, ast_y = np.meshgrid(x, y, sparse=False, indexing='ij')

ast_states = []
for ii in range(num_x):
    for jj in range(num_y):
        if not (abs(1.6 * ast_x[ii, jj] - ast_y[ii, jj]) <= 160) and not (
                abs(-1.6 * ast_x[ii, jj] + 1600 - ast_y[ii, jj]) <= 160):
            ast_states.append({"position": (ast_x[ii, jj], ast_y[ii, jj]+50), "angle": 0.0, "speed": 90, "size": 2})

adv_moving_corridor_angled_1_mines = Scenario(
    name="adv_moving_corridor_angled_1_mines",
    asteroid_states=ast_states,
    ship_states=[{"position": (230, 400), "angle": 180, "team": 1, "mines_remaining": 10},
                 {"position": (770, 400), "angle": 180, "team": 2, "mines_remaining": 10}],
    seed=0, ammo_limit_multiplier=0.05, time_limit=40,
)

# ring closing left
R = 300
theta = np.linspace(0, 2 * np.pi, 17)[:-1]
ast_x = [R * np.cos(angle) + 200 for angle in theta]
ast_y = [R * np.sin(angle) + 400 for angle in theta]

init_angle = [180 + val * 180 / np.pi for val in theta]
ast_states = []
for ii in range(len(init_angle)):
    ast_states.append({"position": (ast_x[ii], ast_y[ii]), "angle": init_angle[ii], "speed": 30})

adv_multi_ring_closing_left = Scenario(
    name="multi_ring_closing_left",
    asteroid_states=ast_states,
    ship_states=[{"position": (200, 400), "team": 1, "mines_remaining": 3},
                 {"position": (600, 400), "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=30
)

adv_multi_ring_closing_left2 = Scenario(
    name="multi_ring_closing_left",
    asteroid_states=ast_states,
    ship_states=[{"position": (600, 400), "team": 1, "mines_remaining": 3},
                 {"position": (200, 400), "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=30
)

# ring closing both 2
R1 = 150
R2 = 200
R3 = 250
R4 = 300
R = [300, 310]
theta = np.linspace(0, 2 * np.pi, 49)[:-1]
ast_x = [[r * np.cos(angle) + 250 for angle in theta] for r in R]
ast_x2 = [[r * np.cos(angle) + 750 for angle in theta] for r in R]
ast_y = [[r * np.sin(angle) + 400 for angle in theta] for r in R]

init_angle = [180 + val * 180 / np.pi for val in theta]
ast_states = []
for jj in range(len(R)):
    for ii in range(len(init_angle)):
        ast_states.append({"position": (ast_x[jj][ii], ast_y[jj][ii]), "angle": init_angle[ii], "speed": 30, "size": 1})
        ast_states.append({"position": (ast_x2[jj][ii], ast_y[jj][ii]), "angle": init_angle[ii], "speed": 30, "size": 1})

adv_multi_ring_closing_both2 = Scenario(
    name="multi_ring_closing_both2",
    asteroid_states=ast_states,
    ship_states=[{"position": (250, 400), "team": 1, "mines_remaining": 3},
                 {"position": (750, 400), "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=30
)

# ring closing both inside fast
R = 400
theta = np.linspace(0, 2 * np.pi, 27)[:-1]
ast_x = [R * np.cos(angle) + 500 for angle in theta]
ast_y = [R * np.sin(angle) + 400 for angle in theta]

init_angle = [180 + val * 180 / np.pi for val in theta]
ast_states = []
for ii in range(len(init_angle)):
    ast_states.append({"position": (ast_x[ii], ast_y[ii]), "angle": init_angle[ii], "speed": 150, "size":2})

adv_multi_ring_closing_both_inside_fast = Scenario(
    name="multi_ring_closing_both_inside_fast",
    asteroid_states=ast_states,
    ship_states=[{"position": (400, 400), "team": 1, "mines_remaining": 3},
                 {"position": (600, 400), "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=30
)

# ring closing both
R = 300
theta = np.linspace(0, 2 * np.pi, 17)[:-1]
ast_x = [R * np.cos(angle) + 250 for angle in theta]
ast_x2 = [R * np.cos(angle) + 750 for angle in theta]
ast_y = [R * np.sin(angle) + 400 for angle in theta]

init_angle = [180 + val * 180 / np.pi for val in theta]
ast_states = []
for ii in range(len(init_angle)):
    ast_states.append({"position": (ast_x[ii], ast_y[ii]), "angle": init_angle[ii], "speed": 30})
    ast_states.append({"position": (ast_x2[ii], ast_y[ii]), "angle": init_angle[ii], "speed": 30})

adv_multi_two_rings_closing = Scenario(
    name="multi_two_rings_closing",
    asteroid_states=ast_states,
    ship_states=[{"position": (200, 400), "team": 1, "mines_remaining": 3},
                 {"position": (750, 400), "team": 2, "mines_remaining": 3}],
    seed=1, time_limit=45
)
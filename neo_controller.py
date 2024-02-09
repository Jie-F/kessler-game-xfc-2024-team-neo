# Neo
# XFC 2024 Kessler controller
# Jie Fan (jie.f@pm.me)
# Feel free to reach out if you have questions or want to discuss anything!

# TODO: ASCII art for ship
# TODO: Add heuristic FIS for maneuvering
# TODO: Show stats at the end
# TODO: Use mine fis before creating the sim, not inside of it
# TODO: Move recording

import random
import math
from math import sin, cos, sqrt, floor, ceil, radians, degrees, atan2, asin, pi, dist
import heapq
import time
import bisect
from functools import lru_cache
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skfuzzy as fuzz
from skfuzzy import control as ctrl
#from scalene import scalene_profiler
import line_profiler

from src.kesslergame import KesslerController

# IMPORTANT: if multiple scenarios are run back-to-back, this controller doesn't get freshly initialized in the subsequent runs.
# If any global variables are changed during execution, make sure to reset them when the timestep is 0.

# Output config
DEBUG_MODE = False
PRINT_EXPLANATIONS = True
EXPLANATION_MESSAGE_SILENCE_INTERVAL_S = 5 # Repeated messages within this time window get silenced

# State dumping for debug
REALITY_STATE_DUMP = False
SIMULATION_STATE_DUMP = False
KEY_STATE_DUMP = False
gamestate_plotting = False
bullet_sim_plotting = False
next_target_plotting = False
maneuver_sim_plotting = False
start_gamestate_plotting_at_second = None
new_target_plot_pause_time_s = 0.5
slow_down_game_after_second = math.inf
slow_down_game_pause_time = 2
#move_recording = True
#recorded_list_of_moves

# These can trade off to get better performance at the expense of safety
ENABLE_ASSERTIONS = True
PRUNE_SIM_STATE_SEQUENCE = True
VALIDATE_SIMULATED_KEY_STATES = True
VALIDATE_ALL_SIMULATED_STATES = False

# Strategic variables
UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON = 8
UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON = 3 # 1 second to turn, 2 seconds for bullet travel time
asteroid_size_shot_priority = [math.nan, 1, 2, 3, 4] # Index i holds the priority of shooting an asteroid of size i (the first element is not important)

# Initialization
# Global variable to store messages and their last printed timestep
explanation_messages_with_timestamps = {} # Make sure to clear this when timestep is 0 so back-to-back runs work properly!

# Quantities
TAD = 0.1
GRAIN = 0.001
EPS = 0.0000000001

# Kessler game constants
DELTA_TIME = 1/30 # s/ts
#fire_time = 1/10  # seconds
BULLET_SPEED = 800.0 # px/s
BULLET_MASS = 1 # kg
BULLET_LENGTH = 12 # px
SHIP_MAX_TURN_RATE = 180.0 # deg/s
SHIP_MAX_THRUST = 480.0 # px/s^2
SHIP_DRAG = 80.0 # px/s^2
SHIP_MAX_SPEED = 240.0 # px/s
SHIP_RADIUS = 20.0 # px
SHIP_MASS = 300 # kg
TIMESTEPS_UNTIL_SHIP_ACHIEVES_MAX_SPEED = ceil(SHIP_MAX_SPEED/(SHIP_MAX_THRUST - SHIP_DRAG)/DELTA_TIME) # Should be 18 timesteps
COLLISION_CHECK_PAD = EPS # px
ASTEROID_AIM_BUFFER_PIXELS = 7 # px
COORDINATE_BOUND_CHECK_PADDING = 1 # px
MINE_BLAST_RADIUS = 150 # px
MINE_RADIUS = 12 # px
MINE_BLAST_PRESSURE = 2000
MINE_FUSE_TIME = 3 # s
MINE_MASS = 25 # kg
ASTEROID_RADII_LOOKUP = [8*size for size in range(5)] # asteroid.py
ASTEROID_AREA_LOOKUP = [pi*r*r for r in ASTEROID_RADII_LOOKUP]
ASTEROID_MASS_LOOKUP = [0.25*pi*(8*size)**2 for size in range(5)] # asteroid.py
RESPAWN_INVINCIBILITY_TIME_S = 3 # s
SHIP_AVOIDANCE_PADDING = 25
SHIP_AVOIDANCE_SPEED_PADDING_RATIO = 1/100
ASTEROID_COUNT_LOOKUP = (0, 1, 4, 13, 40) # A size 2 asteroid is 4 asteroids, size 4 is 30, etc. Each asteroid splits into 3, and itself is counted as well. Explicit formula is count(n) = (3^n - 1)/2

# Set up FIS as global variables
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

def mine_fis(num_mines_left: int, num_lives_left: int, num_asteroids_hit: int):
    if num_mines_left == 0 or num_asteroids_hit < 8:
        return False
    debug_print(f"Mine fis: Mines left: {num_mines_left}, lives left: {num_lives_left}, asteroids hit: {num_asteroids_hit}")
    mine_dropping_fis.input['mines_left'] = num_mines_left
    mine_dropping_fis.input['lives_left'] = num_lives_left
    mine_dropping_fis.input['asteroids_hit'] = num_asteroids_hit
    # Compute the output
    mine_dropping_fis.compute()
    drop_decision = mine_dropping_fis.output['drop_mine']
    # Interpreting the output
    should_drop_mine = drop_decision > 5  # True for drop, False for don't drop
    return should_drop_mine

def compare_asteroids(ast_a, ast_b):
    for i in range(2):
        if ast_a['position'][i] != ast_b['position'][i]:
            return False
        if ast_a['velocity'][i] != ast_b['velocity'][i]:
            return False
        #if not math.isclose(ast_a['velocity'][i], ast_b['velocity'][i], abs_tol=EPS):
        #    return False
    if ast_a['size'] != ast_b['size']:
        return False
    if ast_a['mass'] != ast_b['mass']:
        return False
    if ast_a['radius'] != ast_b['radius']:
        return False
    return True

def compare_bullets(bul_a, bul_b):
    for i in range(2):
        if bul_a['position'][i] != bul_b['position'][i]:
            return False
        if bul_a['velocity'][i] != bul_b['velocity'][i]:
            return False
    if bul_a['heading'] != bul_b['heading']:
        return False
    if bul_a['mass'] != bul_b['mass']:
        return False
    return True

def compare_mines(mine_a, mine_b):
    for i in range(2):
        if not mine_a['position'][i] == mine_b['position'][i]:
            return False
    if not mine_a['mass'] == mine_b['mass']:
        return False
    if not mine_a['fuse_time'] == mine_b['fuse_time']:
        return False
    if not mine_a['remaining_time'] == mine_b['remaining_time']:
        return False
    return True

def compare_gamestates(gamestate_a, gamestate_b):
    # The game state consists of asteroids, ships, bullets, mines
    asteroids_a = gamestate_a['asteroids']
    asteroids_b = gamestate_b['asteroids']
    if len(asteroids_a) != len(asteroids_b):
        print("Asteroids lists are different lengths!")
        return False
    for i in range(len(asteroids_a)):
        if not compare_asteroids(asteroids_a[i], asteroids_b[i]):
            print(f"Asteroids don't match! {ast_to_string(asteroids_a[i])} vs {ast_to_string(asteroids_b[i])}")
            return False

    bullets_a = gamestate_a['bullets']
    bullets_b = gamestate_b['bullets']
    if len(bullets_a) != len(bullets_b):
        print("Bullet lists are different lengths!")
        return False
    for i in range(len(bullets_a)):
        if bullets_a[i]['velocity'][0] == 796.5596791827362:
            print('THIS IS THE BULLET')
        if not compare_bullets(bullets_a[i], bullets_b[i]):
            print(f"Bullets don't match! {bullets_a[i]} vs {bullets_b[i]}")
            return False

    mines_a = gamestate_a['mines']
    mines_b = gamestate_b['mines']
    if len(mines_a) != len(mines_b):
        print("Mine lists are different lengths!")
        return False
    for i in range(len(mines_a)):
        if not compare_mines(mines_a[i], mines_b[i]):
            print("Mines don't match!")
            return False
    # No need to compare trivial stuff like timesteps that are in the game state
    return True

def compare_shipstates(ship_a, ship_b):
    # Compare booleans and integers
    if ship_a['is_respawning'] != ship_b['is_respawning']:
        return False
    if ship_a['id'] != ship_b['id']:
        print("Ship ID's don't match!")
        return False
    if ship_a['team'] != ship_b['team']:
        print("Ship teams don't match!")
        return False
    if ship_a['lives_remaining'] != ship_b['lives_remaining']:
        print("Ship lives remaining don't match!")
        return False
    if ship_a['bullets_remaining'] != ship_b['bullets_remaining']:
        print("Ship bullets remaining don't match!")
        print(ship_a['bullets_remaining'])
        print(ship_b['bullets_remaining'])
        return False
    if ship_a['mines_remaining'] != ship_b['mines_remaining']:
        print("Ship mines remaining don't match!")
        return False
    # TODO: Enable this
    #if 'can_fire' in ship_a and 'can_fire' in ship_b:
    #    if ship_a['can_fire'] != ship_b['can_fire']:
    #        print("Ship can fire don't match!")
    #        return False
    if ship_a['max_speed'] != ship_b['max_speed']:
        print("Ship max speeds don't match!")
        return False

    # Compare positions and velocities
    for i in range(2):
        if ship_a['position'][i] != ship_b['position'][i]:
            print("Ship positions don't match!")
            return False
        if ship_a['velocity'][i] != ship_b['velocity'][i]:
            print("Ship velocities don't match!")
            return False

    # Compare other floating-point values
    if ship_a['speed'] != ship_b['speed']:
        print("Ship speeds don't match!")
        return False
    if ship_a['heading'] != ship_b['heading']:
        print("Ship headings don't match!")
        return False
    if ship_a['mass'] != ship_b['mass']:
        print("Ship masses don't match!")
        return False
    if ship_a['radius'] != ship_b['radius']:
        print("Ship radii don't match!")
        return False
    if ship_a['fire_rate'] != ship_b['fire_rate']:
        print("Ship fire rates don't match!")
        return False
    if ship_a['drag'] != ship_b['drag']:
        print("Ship drags don't match!")
        return False

    # Compare tuple values
    for i in range(2):
        if ship_a['thrust_range'][i] != ship_b['thrust_range'][i]:
            print("Ship thrust ranges don't match!")
            return False
        if ship_a['turn_rate_range'][i] != ship_b['turn_rate_range'][i]:
            print("Ship turn rates don't match!")
            return False
    return True

def wrap_position(position: tuple, bounds: tuple):
    x, y = position
    x_bound, y_bound = bounds

    if x > x_bound:
        x -= x_bound
    elif x + x_bound < x_bound: # This statement can be mathematically simplified to subtract x_bound from both sides, but Kessler is silly and does it this way so we have to match it
        x += x_bound

    if y > y_bound:
        y -= y_bound
    elif y + y_bound < y_bound: # This statement can be mathematically simplified to subtract y_bound from both sides, but Kessler is silly and does it this way so we have to match it
        y += y_bound

    return x, y

def wrap_position_slow(position: tuple, bounds: tuple):
    position = list(position)
    for idx, pos in enumerate(position):
        bound = bounds[idx]
        offset = bound - pos
        if offset < 0 or offset > bound:
            position[idx] += bound * np.sign(offset)
    return tuple(position)

def preprocess_bullets(bullets):
    for b in bullets:
        bullet_tail_delta = (-BULLET_LENGTH*cos(radians(b['heading'])), -BULLET_LENGTH*sin(radians(b['heading'])))
        b['tail_delta'] = bullet_tail_delta
    return bullets

def preprocess_bullets_in_gamestate(game_state):
    game_state['bullets'] = preprocess_bullets(game_state['bullets'])
    return game_state

def print_explanation(message, current_timestep, time_threshold=EXPLANATION_MESSAGE_SILENCE_INTERVAL_S/DELTA_TIME):
    if not PRINT_EXPLANATIONS:
        return
    global explanation_messages_with_timestamps

    # Check if the message was printed within the time threshold
    last_timestep_printed = explanation_messages_with_timestamps.get(message, -math.inf)
    if current_timestep - last_timestep_printed >= time_threshold:
        print(message)
        explanation_messages_with_timestamps[message] = current_timestep

def debug_print(*messages):
    if DEBUG_MODE:
        print(*messages)

def evaluate_scenario(game_state, ship_state):
    asteroids = game_state['asteroids']
    width = game_state['map_size'][0]
    height = game_state['map_size'][1]

    def asteroid_density():
        total_asteroid_coverage_area = 0
        for a in asteroids:
            total_asteroid_coverage_area += ASTEROID_AREA_LOOKUP[a['size']]
        total_screen_size = width*height
        return total_asteroid_coverage_area/total_screen_size

    def average_velocity():
        total_x_velocity = 0
        total_y_velocity = 0
        for a in asteroids:
            total_x_velocity += a['velocity'][0]
            total_y_velocity += a['velocity'][1]
        num_asteroids = len(asteroids)
        return (total_x_velocity/num_asteroids, total_y_velocity/num_asteroids)

    def average_speed():
        total_speed = 0
        for a in asteroids:
            total_speed += sqrt(a['velocity'][0]*a['velocity'][0] + a['velocity'][1]*a['velocity'][1])
        return total_speed/len(asteroids)

    average_density = asteroid_density()
    current_asteroids, total_asteroids = asteroid_counter(asteroids)
    average_vel = average_velocity()
    avg_speed = average_speed()
    print(f"Average asteroid density: {average_density}, average vel: {average_vel}, average speed: {avg_speed}")

def asteroid_counter(asteroids: list=None):
    if asteroids is None:
        return 0
    current_count = len(asteroids)
    total_count = 0
    for a in asteroids:
        total_count += ASTEROID_COUNT_LOOKUP[a['size']]
    return total_count, current_count

class GameStatePlotter:
    def __init__(self, game_state):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, game_state['map_size'][0]])
        self.ax.set_ylim([0, game_state['map_size'][1]])
        self.ax.set_aspect('equal', adjustable='box')
        self.game_state = game_state

    def fuse_time_to_color(self, remaining_time):
        if remaining_time >= 3:
            return "#00FF00"  # Green
        elif remaining_time <= 0:
            return "#FF0000"  # Red
        else:
            # Linearly interpolate between green and red
            # From green to yellow
            if remaining_time > 1.5:
                # Interpolate between green and yellow
                green_to_yellow_ratio = (remaining_time - 1.5) / 1.5
                red = int(255 * (1 - green_to_yellow_ratio))
                green = 255
            # From yellow to red
            else:
                # Interpolate between yellow and red
                yellow_to_red_ratio = remaining_time / 1.5
                red = 255
                green = int(255 * yellow_to_red_ratio)

            # Convert to hex
            return f"#{red:02x}{green:02x}00"

    def update_plot(self, asteroids: dict=None, ship_state: dict | None=None, bullets: list=None, special_bullets: list=None, circled_asteroids: list=None, ghost_asteroids: list=None, forecasted_asteroids: list=None, mines: list=None, clear_plot=True, pause_time=EPS, plot_title=""):
        if asteroids is None:
            asteroids = []
        if bullets is None:
            bullets = []
        if special_bullets is None:
            special_bullets = []
        if circled_asteroids is None:
            circled_asteroids = []
        if ghost_asteroids is None:
            ghost_asteroids = []
        if forecasted_asteroids is None:
            forecasted_asteroids = []
        if mines is None:
            mines = []
        if clear_plot:
            self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.set_xlim([0, self.game_state['map_size'][0]])
        self.ax.set_ylim([0, self.game_state['map_size'][1]])
        self.ax.set_aspect('equal', adjustable='box')

        self.ax.set_title(plot_title, fontsize=14, color='black')

        # Draw asteroids and their velocities
        for a in asteroids:
            if a:
                asteroid_circle = patches.Circle(a['position'], a['radius'], color='gray', fill=True)
                self.ax.add_patch(asteroid_circle)
                self.ax.arrow(a['position'][0], a['position'][1], a['velocity'][0]*DELTA_TIME, a['velocity'][1]*DELTA_TIME, head_width=3, head_length=5, fc='white', ec='white')
        for a in ghost_asteroids:
            if a:
                asteroid_circle = patches.Circle(a['position'], a['radius'], color='#333333', fill=True, zorder=-100)
                self.ax.add_patch(asteroid_circle)
                #self.ax.arrow(a['position'][0], a['position'][1], a['velocity'][0]*delta_time, a['velocity'][1]*delta_time, head_width=3, head_length=5, fc='white', ec='white')
        for a in forecasted_asteroids:
            if a:
                asteroid_circle = patches.Circle(a['position'], a['radius'], color='#440000', fill=True, zorder=100, alpha=0.4)
                self.ax.add_patch(asteroid_circle)
        #print(highlighted_asteroids)
        for a in circled_asteroids:
            if a:
                #print('asteroid', a)
                highlight_circle = patches.Circle(a['position'], a['radius'] + 5, color='orange', fill=False)
                self.ax.add_patch(highlight_circle)

        for m in mines:
            if m:
                highlight_circle = patches.Circle(m['position'], MINE_BLAST_RADIUS, color=self.fuse_time_to_color(m['remaining_time']), fill=False)
                self.ax.add_patch(highlight_circle)

        # Hard code a circle so I can see what a coordinate on screen is for debugging
        circle_hardcoded_coordinate = False
        if circle_hardcoded_coordinate:
            highlight_circle = patches.Circle((1017.3500530032204, 423.2001881178426), 25, color='red', fill=False)
            self.ax.add_patch(highlight_circle)

        if ship_state:
            # Draw the ship as an elongated triangle
            ship_size_base = SHIP_RADIUS
            ship_size_tip = SHIP_RADIUS
            ship_heading = ship_state['heading']
            ship_position = ship_state['position']
            angle_rad = radians(ship_heading)
            ship_vertices = [
                (ship_position[0] + ship_size_tip*cos(angle_rad), ship_position[1] + ship_size_tip*sin(angle_rad)),
                (ship_position[0] + ship_size_base*cos(angle_rad + pi*3/4), ship_position[1] + ship_size_base*sin(angle_rad + pi*3/4)),
                (ship_position[0] + ship_size_base*cos(angle_rad - pi*3/4), ship_position[1] + ship_size_base*sin(angle_rad - pi*3/4)),
            ]
            ship = patches.Polygon(ship_vertices, color='green', fill=True)
            self.ax.add_patch(ship)

            # Draw the ship's hitbox as a blue circle
            ship_circle = patches.Circle(ship_position, SHIP_RADIUS, color='blue', fill=False)
            self.ax.add_patch(ship_circle)

        # Draw arrow line segments for bullets
        for b in bullets:
            if b:
                bullet_tail = (b['position'][0] - BULLET_LENGTH*cos(radians(b['heading'])), b['position'][1] - BULLET_LENGTH*sin(radians(b['heading'])))
                self.ax.arrow(bullet_tail[0], bullet_tail[1], b['position'][0] - bullet_tail[0], b['position'][1] - bullet_tail[1], head_width=3, head_length=5, fc='red', ec='red')
        for b in special_bullets:
            if b:
                bullet_tail = (b['position'][0] - BULLET_LENGTH*cos(radians(b['heading'])), b['position'][1] - BULLET_LENGTH*sin(radians(b['heading'])))
                self.ax.arrow(bullet_tail[0], bullet_tail[1], b['position'][0] - bullet_tail[0], b['position'][1] - bullet_tail[1], head_width=3, head_length=5, fc='green', ec='green')
        plt.draw()
        plt.pause(pause_time)

    def is_asteroid_in_list(self, asteroid_list, asteroid):
        # Assuming you have a function to check if an asteroid is in the list
        return asteroid in asteroid_list

    def start_animation(self):
        # This can be an empty function if you are calling update_plot manually in your game loop
        pass

def write_to_json(data, filename):
    import json
    """
    Serialize a Python list or dictionary to JSON and write it to a file.

    Args:
    data (list or dict): The Python list or dictionary to be serialized.
    filename (str): The name of the file where the JSON will be saved.

    Returns:
    None
    """
    try:
        with open(filename, 'a') as file:
            json.dump(data, file, indent=4)
        print(f"Data successfully written to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

def append_dict_to_file(dict_data, file_path):
    """
    This function takes a dictionary and appends it as a string in a JSON-like format to a text file.

    :param dict_data: Dictionary to be converted to a string and appended.
    :param file_path: Path of the text file to append the data to.
    """
    import json

    # Convert the dictionary to a JSON-like string
    dict_string = json.dumps(dict_data, indent=4)

    # Append the string to the file
    with open(file_path, 'a') as file:
        file.write(dict_string + "\n")

def log_tuple_to_file(tuple_of_numbers, file_path):
    """
    Logs a tuple of numbers to a text file, appending to the file.
    Each tuple is logged on a new line, with its elements separated by commas.

    :param tuple_of_numbers: Tuple containing numbers to be logged.
    :param file_path: Path of the file to which the data will be logged.
    """
    with open(file_path, 'a') as file:
        # Joining the tuple elements with commas and appending a newline
        file.write(','.join(map(str, tuple_of_numbers)) + '\n')

def ast_to_string(a):
    if 'timesteps_until_appearance' in a:
        return f"Pos: ({a['position'][0]}, {a['position'][1]}), Vel: ({a['velocity'][0]}, {a['velocity'][1]}), Size: {a['size']}, Appears in {a['timesteps_until_appearance']*DELTA_TIME} s"
    else:
        return f"Pos: ({a['position'][0]}, {a['position'][1]}), Vel: ({a['velocity'][0]}, {a['velocity'][1]}), Size: {a['size']}"

def angle_difference_rad(angle1, angle2):
    # Calculate the raw difference
    raw_diff = angle1 - angle2

    # Adjust for wraparound using modulo
    adjusted_diff = raw_diff % (2*pi)

    # If the difference is greater than pi, adjust to keep within -pi to pi
    if adjusted_diff > pi:
        adjusted_diff -= 2*pi

    return adjusted_diff

def angle_difference_deg(angle1, angle2):
    # Calculate the raw difference
    raw_diff = angle1 - angle2

    # Adjust for wraparound using modulo
    adjusted_diff = raw_diff % 360

    # If the difference is greater than 180 degrees, adjust to keep within -180 to 180
    if adjusted_diff > 180:
        adjusted_diff -= 360

    return adjusted_diff

def check_collision(a_x, a_y, a_r, b_x, b_y, b_r, pixel_padding=0):
    delta_x = a_x - b_x
    delta_y = a_y - b_y
    separation = a_r + b_r + pixel_padding
    if delta_x*delta_x + delta_y*delta_y < separation*separation:
        return True
    else:
        return False

def collision_prediction(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_radius, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius):
    # https://stackoverflow.com/questions/11369616/circle-circle-collision-prediction/
    Oax, Oay = ship_pos_x, ship_pos_y
    Dax, Day = ship_vel_x, ship_vel_y
    Obx, Oby = ast_pos_x, ast_pos_y
    Dbx, Dby = ast_vel_x, ast_vel_y
    ra = ship_radius
    rb = ast_radius
    separation = ra + rb
    # If both objects are stationary, then we only have to check the collision right now and not do any fancy math
    # This should speed up scenarios where most asteroids are stationary
    if math.isclose(Dax, 0, abs_tol=EPS) and math.isclose(Day, 0, abs_tol=EPS) and math.isclose(Dbx, 0, abs_tol=EPS) and math.isclose(Dby, 0, abs_tol=EPS):
        if check_collision(Oax, Oay, ra, Obx, Oby, rb, COLLISION_CHECK_PAD):
            t1 = -math.inf
            t2 = math.inf
        else:
            t1 = math.nan
            t2 = math.nan
        return t1, t2
    A = Dax*Dax + Dbx*Dbx + Day*Day + Dby*Dby - 2*(Dax*Dbx + Day*Dby)
    B = 2*(Oax*Dax - Oax*Dbx - Obx*Dax + Obx*Dbx + Oay*Day - Oay*Dby - Oby*Day + Oby*Dby)
    C = Oax*Oax + Obx*Obx + Oay*Oay + Oby*Oby - 2*(Oax*Obx + Oay*Oby) - separation*separation
    t1, t2 = solve_quadratic(A, B, C)
    if t1 is None:
        t1 = math.nan
    if t2 is None:
        t2 = math.nan
    if t1 and not math.isnan(t1):
        pass
        #debug_print(f"Running collision prediction with ship pos ({ship_pos_x}, {ship_pos_y}) ship vel ({ship_vel_x}, {ship_vel_y}), ast pos ({ast_pos_x}, {ast_pos_y}), ast vel ({ast_vel_x}, {ast_vel_y})")
        #debug_print(f"Found imminent collision of time {t1}, a: {A}, b: {B}, c: {C}")
    return t1, t2

def predict_next_imminent_collision_time_with_asteroid(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius):
    #print("ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius")
    #print(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius)
    t1, t2 = collision_prediction(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius)
    #print(f"Quadratic equation gave t1 {t1} t2: {t2} and chatgpt got me {t3} and {t4}")
    # If we're already colliding with something, then return 0 as the next imminent collision time
    if math.isnan(t1) or math.isnan(t2):
        next_imminent_collision_time = math.inf
    else:
        start_collision_time = min(t1, t2)
        end_collision_time = max(t1, t2)
        if end_collision_time < 0:
            next_imminent_collision_time = math.inf
        elif start_collision_time <= 0 <= end_collision_time:
            # 0 <= end_collision_time is for a given, but included in the statement for clarity
            next_imminent_collision_time = 0
        else:
            # start_collision_time > 0 and 0 <= end_collision_time
            next_imminent_collision_time = start_collision_time
    return next_imminent_collision_time

def calculate_border_crossings(x0, y0, vx, vy, W, H, c):
    # Initialize lists to hold crossing times
    x_crossings_times = []
    y_crossings_times = []

    # Calculate crossing times for x (if vx is not zero to avoid division by zero)
    if abs(vx) > EPS:
        # Calculate time to first x-boundary crossing based on direction of vx
        x_crossing_interval = W/abs(vx)
        #print(f"x_crossing_interval: {x_crossing_interval}")
        time_to_first_x_crossing = ((W - x0)/vx if vx > 0 else x0/-vx)
        x_crossings_times.append(time_to_first_x_crossing)
        # Add additional crossings until time c is reached
        while x_crossings_times[-1] + x_crossing_interval <= c:
            x_crossings_times.append(x_crossings_times[-1] + x_crossing_interval)
    #print(f"x crossing times: {x_crossings_times}")
    # Calculate crossing times for y (if vy is not zero)
    if abs(vy) > EPS:
        # Calculate time to first y-boundary crossing based on direction of vy
        y_crossing_interval = H/abs(vy)
        #print(f"y_crossing_interval: {y_crossing_interval}")
        time_to_first_y_crossing = ((H - y0)/vy if vy > 0 else y0/-vy)
        y_crossings_times.append(time_to_first_y_crossing)
        # Add additional crossings until time c is reached
        while y_crossings_times[-1] + y_crossing_interval <= c:
            y_crossings_times.append(y_crossings_times[-1] + y_crossing_interval)
    #print(f"y crossing times: {y_crossings_times}")
    # Merge the two lists while tracking the origin of each time
    merged_times = []
    sequence = []
    i = j = 0

    while i < len(x_crossings_times) and j < len(y_crossings_times):
        if x_crossings_times[i] < y_crossings_times[j]:
            merged_times.append(x_crossings_times[i])
            sequence.append('x')
            i += 1
        else:
            merged_times.append(y_crossings_times[j])
            sequence.append('y')
            j += 1

    # Add any remaining times from the x_crossings_times list
    while i < len(x_crossings_times):
        merged_times.append(x_crossings_times[i])
        sequence.append('x')
        i += 1

    # Add any remaining times from the y_crossings_times list
    while j < len(y_crossings_times):
        merged_times.append(y_crossings_times[j])
        sequence.append('y')
        j += 1

    # Initialize current universe coordinates and list of visited universes
    current_universe_x, current_universe_y = 0, 0
    universes = [(current_universe_x, current_universe_y)]

    # Iterate through merged crossing times and sequence
    for time, crossing in zip(merged_times, sequence):
        if time <= c:
            if crossing == 'x':
                current_universe_x += 1 if vx > 0 else -1
            else:  # crossing == 'y'
                current_universe_y += 1 if vy > 0 else -1
            universes.append((current_universe_x, current_universe_y))
    return universes

def unwrap_asteroid(asteroid: dict, max_x: float, max_y: float, time_horizon_s: float=10) -> list:
    if abs(asteroid['velocity'][0]) < EPS and abs(asteroid['velocity'][1]) < EPS:
        return [asteroid]

    unwrapped_asteroids = []
    # The idea is to track which universes the asteroid visits from t=t_0 until t=t_0 + time_horizon_s.
    # The current universe is (0, 0) and if the asteroid wraps to the right, it visits (1, 0). If it wraps down, it visits (0, -1). If it wraps right and then down, it starts in (0, 0), visits (1, 0), and finally (1, -1).
    border_crossings = calculate_border_crossings(asteroid['position'][0], asteroid['position'][1], asteroid['velocity'][0], asteroid['velocity'][1], max_x, max_y, time_horizon_s)
    for universe in border_crossings:
        # We negate the directions because we're using the frame of reference of the ship now, not the asteroid
        dx = -universe[0]*max_x
        dy = -universe[1]*max_y
        unwrapped_asteroid = dict(asteroid)
        unwrapped_asteroid['position'] = (unwrapped_asteroid['position'][0] + dx, unwrapped_asteroid['position'][1] + dy)
        unwrapped_asteroids.append(unwrapped_asteroid)
    #print(f"Returning unwrapped asteroids: {unwrapped_asteroids}")
    return unwrapped_asteroids

def check_coordinate_bounds(game_state, x, y):
    if 0 <= x <= game_state['map_size'][0] and 0 <= y <= game_state['map_size'][1]:
        return True
    else:
        return False

def check_coordinate_bounds_exact(game_state, x, y):
    wrapped = wrap_position((x, y), game_state['map_size'])
    if math.isclose(x, wrapped[0]) and math.isclose(y, wrapped[1]):
        return True
    else:
        return False

def solve_quadratic(a, b, c):
    # This solves a*x*x + b*x + c = 0 for x
    # This handles the case where a, b, or c are 0.
    d = b*b - 4*a*c
    if d < 0:
        # No real solutions.
        r1 = None
        r2 = None
    elif a == 0:
        # This is a linear equation. Handle this case separately.
        r1 = -c/b
        r2 = None
    else:
        # This handles the case where b or c are 0
        # If d is 0, technically there's only one solution but this will give two duplicated solutions. It's not worth checking each time for this since it's so rare
        if b > 0:
            u = -b - sqrt(d)
        else:
            u = -b + sqrt(d)
        r1 = u/(2*a)
        r2 = 2*c/u
    return r1, r2

def calculate_interception(ship_pos_x, ship_pos_y, asteroid_pos_x, asteroid_pos_y, asteroid_vel_x, asteroid_vel_y, asteroid_r, ship_heading, game_state, future_shooting_timesteps=0):
    # The bullet's head originates from the edge of the ship's radius.
    # We want to set the position of the bullet to the center of the bullet, so we have to do some fanciness here so that at t=0, the bullet's center is where it should be
    t_0 = (SHIP_RADIUS - BULLET_LENGTH/2)/BULLET_SPEED
    # Positions are relative to the ship. We set the origin to the ship's position. Remember to translate back!
    origin_x = ship_pos_x
    origin_y = ship_pos_y
    avx = asteroid_vel_x
    avy = asteroid_vel_y
    ax = asteroid_pos_x - origin_x + avx*DELTA_TIME # We project the asteroid one timestep ahead, since by the time we shoot our bullet, the asteroid would have moved one more timestep!
    ay = asteroid_pos_y - origin_y + avy*DELTA_TIME

    vb = BULLET_SPEED
    #tr = radians(ship_max_turn_rate) # rad/s
    vb_sq = vb*vb
    theta_0 = radians(ship_heading)

    # Calculate constants for naive_desired_heading_calc
    A = avx*avx + avy*avy - vb_sq

    time_until_can_fire_s = future_shooting_timesteps*DELTA_TIME
    ax_delayed = ax + time_until_can_fire_s*avx # We add a delay to account for the timesteps until we can fire delay
    ay_delayed = ay + time_until_can_fire_s*avy

    B = 2*(ax_delayed*avx + ay_delayed*avy - vb_sq*t_0)
    C = ax_delayed*ax_delayed + ay_delayed*ay_delayed - vb_sq*t_0*t_0

    t1, t2 = solve_quadratic(A, B, C)
    positive_interception_times = []
    if t1 and t1 >= 0:
        positive_interception_times.append(t1)
    if t2 and t2 >= 0:
        positive_interception_times.append(t2)
    solutions = []
    feasible = False
    for t in positive_interception_times:
        x = ax_delayed + t*avx
        y = ay_delayed + t*avy
        theta = atan2(y, x)
        # If the asteroid is out of bounds, then it will wrap around and this shot isn't feasible
        # However, if an unwrapped asteroid was passed into this function and the interception is inbounds, then it's a feasible shot
        intercept_x = x + origin_x
        intercept_y = y + origin_y
        feasible = check_coordinate_bounds(game_state, intercept_x, intercept_y)
        asteroid_dist = sqrt(x*x + y*y)
        if asteroid_r < asteroid_dist:
            shot_heading_tolerance_rad = asin((asteroid_r - ASTEROID_AIM_BUFFER_PIXELS)/asteroid_dist)
        else:
            shot_heading_tolerance_rad = pi/4
        solutions.append((feasible, angle_difference_rad(theta, theta_0), shot_heading_tolerance_rad, t, intercept_x, intercept_y, asteroid_dist))

    return solutions
    # feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time_s + 0*future_shooting_timesteps*delta_time, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception

def forecast_asteroid_bullet_splits(a, timesteps_until_appearance, bullet_heading_deg=None, bullet_velocity=None, game_state=None, wrap=False):
    if a['size'] == 1:
        # Asteroids of size 1 don't split
        return []
    # Look at asteroid.py in the Kessler game's code
    if bullet_heading_deg is not None:
        bullet_vel_x = cos(radians(bullet_heading_deg))*BULLET_SPEED
        bullet_vel_y = sin(radians(bullet_heading_deg))*BULLET_SPEED
    elif bullet_velocity is not None:
        bullet_vel_x, bullet_vel_y = bullet_velocity
        #print(f"Bullet vel: {bullet_vel_x} {bullet_vel_y}")
    else:
        raise Exception("No bullet heading or velocity provided")
    vfx = (1/(BULLET_MASS + a['mass']))*(BULLET_MASS*bullet_vel_x + a['mass']*a['velocity'][0])
    vfy = (1/(BULLET_MASS + a['mass']))*(BULLET_MASS*bullet_vel_y + a['mass']*a['velocity'][1])
    return forecast_asteroid_splits(a, timesteps_until_appearance, vfx, vfy, game_state, wrap)

def forecast_asteroid_mine_splits(asteroid, timesteps_until_appearance, mine, game_state=None, wrap=False):
    if asteroid['size'] == 1:
        # Asteroids of size 1 don't split
        return []
    #dist_slow = sqrt((mine['position'][0] - asteroid['position'][0])**2 + (mine['position'][1] - asteroid['position'][1])**2)
    #dist = dist(mine['position'], asteroid['position'])
    #dist = dist_slow
    delta_x = mine['position'][0] - asteroid['position'][0]
    delta_y = mine['position'][1] - asteroid['position'][1]
    dist = sqrt(delta_x*delta_x + delta_y*delta_y)
    #if enable_assertions:
    #    assert math.isclose(dist, dist_slow)
    F = (-dist/MINE_BLAST_RADIUS + 1)*MINE_BLAST_PRESSURE*2*asteroid['radius']
    a = F/asteroid['mass']
    # calculate "impulse" based on acc
    if dist == 0:
        debug_print(f"Dist is 0! Kessler will spit out a runtime warning, and this asteroid will disappear without splitting.")
        return []
    vfx = asteroid['velocity'][0] + a*(asteroid['position'][0] - mine['position'][0])/dist
    vfy = asteroid['velocity'][1] + a*(asteroid['position'][1] - mine['position'][1])/dist
    #debug_print(f"{asteroid['velocity'][0]} + {a}*({asteroid['position'][0]} - {mine['position'][0]})/{dist}")
    return forecast_asteroid_splits(asteroid, timesteps_until_appearance, vfx, vfy, game_state, wrap)

def forecast_asteroid_ship_splits(asteroid, timesteps_until_appearance, ship_velocity, game_state=None, wrap=False):
    if asteroid['size'] == 1:
        # Asteroids of size 1 don't split
        return []
    #print(f"Ship vel: {ship_velocity}")
    vfx = (1/(SHIP_MASS + asteroid['mass']))*(SHIP_MASS*ship_velocity[0] + asteroid['mass']*asteroid['velocity'][0])
    vfy = (1/(SHIP_MASS + asteroid['mass']))*(SHIP_MASS*ship_velocity[1] + asteroid['mass']*asteroid['velocity'][1])
    return forecast_asteroid_splits(asteroid, timesteps_until_appearance, vfx, vfy, game_state, wrap)

def forecast_asteroid_splits(a, timesteps_until_appearance, vfx, vfy, game_state=None, wrap=False):
    # Calculate speed of resultant asteroid(s) based on velocity vector
    v = sqrt(vfx**2 + vfy**2) # TODO: Should be safe to revert to x*x
    # Calculate angle of center asteroid for split (degrees)
    theta = atan2(vfy, vfx)*180/pi#degrees(atan2(vfy, vfx))
    # Split angle is the angle off of the new velocity vector for the two asteroids to the sides, the center child
    # asteroid continues on the new velocity path
    split_angle = 15
    angles = [theta + split_angle, theta, theta - split_angle]
    # This is wacky because we're back-extrapolation the position of the asteroid BEFORE IT WAS BORN!!!!11!

    '''
    for angle in angles:
        while angle < 0:
            angle += 360
        while angle > 360:
            angle -= 360
    '''

    forecasted_asteroids = [{'position': wrap_position((a['position'][0] + a['velocity'][0]*DELTA_TIME*timesteps_until_appearance - timesteps_until_appearance*cos(radians(angle))*v*DELTA_TIME, a['position'][1] + a['velocity'][1]*DELTA_TIME*timesteps_until_appearance - timesteps_until_appearance*sin(radians(angle))*v*DELTA_TIME), game_state['map_size']) if game_state and wrap else (a['position'][0] + a['velocity'][0]*DELTA_TIME*timesteps_until_appearance - timesteps_until_appearance*cos(radians(angle))*v*DELTA_TIME, a['position'][1] + a['velocity'][1]*DELTA_TIME*timesteps_until_appearance - timesteps_until_appearance*sin(radians(angle))*v*DELTA_TIME),
                             'velocity': (v*cos(radians(angle)), v*sin(radians(angle))),
                             'size': a['size'] - 1,
                             'mass': ASTEROID_MASS_LOOKUP[a['size'] - 1],
                             'radius': ASTEROID_RADII_LOOKUP[a['size'] - 1],
                             'timesteps_until_appearance': timesteps_until_appearance}
                               for angle in angles]
    for a in forecasted_asteroids:
        if math.isclose(a['velocity'][0], 8290563106):
            print("\n\n\n\n\nBAMMO IN NEO")
            print(f"vfx: {vfx} vfy: {vfy}")
            #raise Exception("BAMMO IN NEO")
    return forecasted_asteroids

def maintain_forecasted_asteroids(forecasted_asteroid_splits, game_state=None, wrap=False):
    # Maintain the list of projected split asteroids by advancing the position, decreasing the timestep, and facilitate removal
    new_forecasted_asteroids = []
    for forecasted_asteroid in forecasted_asteroid_splits:
        if forecasted_asteroid['timesteps_until_appearance'] > 1:
            new_a = {
                'position': wrap_position((forecasted_asteroid['position'][0] + forecasted_asteroid['velocity'][0]*DELTA_TIME, forecasted_asteroid['position'][1] + forecasted_asteroid['velocity'][1]*DELTA_TIME), game_state['map_size']) if wrap and game_state is not None else (forecasted_asteroid['position'][0] + forecasted_asteroid['velocity'][0]*DELTA_TIME, forecasted_asteroid['position'][1] + forecasted_asteroid['velocity'][1]*DELTA_TIME),
                'velocity': forecasted_asteroid['velocity'],
                'size': forecasted_asteroid['size'],
                'mass': forecasted_asteroid['mass'],
                'radius': forecasted_asteroid['radius'],
                'timesteps_until_appearance': forecasted_asteroid['timesteps_until_appearance'] - 1,
            }
            new_forecasted_asteroids.append(new_a)
    return new_forecasted_asteroids

def is_asteroid_in_list(list_of_asteroids, a, tolerance=1e-9):
    # Since floating point comparison isn't a good idea, break apart the asteroid dict and compare each element manually in a fuzzy way
    for asteroid in list_of_asteroids:
        if math.isclose(a['position'][0], asteroid['position'][0], abs_tol=tolerance) and math.isclose(a['position'][1], asteroid['position'][1], abs_tol=tolerance) and math.isclose(a['velocity'][0], asteroid['velocity'][0], abs_tol=tolerance) and math.isclose(a['velocity'][1], asteroid['velocity'][1], abs_tol=tolerance) and math.isclose(a['size'], asteroid['size'], abs_tol=tolerance) and math.isclose(a['mass'], asteroid['mass'], abs_tol=tolerance) and math.isclose(a['radius'], asteroid['radius'], abs_tol=tolerance):
            #print(f"INSIDE COMP FUNCTION, ASTEROID {ast_to_string(a)} IS CLOSE TO {ast_to_string(asteroid)}")
            return True
    return False

def count_asteroids_in_mine_blast_radius(game_state, mine_x, mine_y, future_check_timesteps):
    count = 0
    for a in game_state['asteroids']:
        # Extrapolate the asteroid position into the time of the mine detonation to check its bounds
        asteroid_future_x = a['position'][0] + future_check_timesteps*a['velocity'][0]*DELTA_TIME
        asteroid_future_y = a['position'][1] + future_check_timesteps*a['velocity'][1]*DELTA_TIME
        if check_coordinate_bounds(game_state, asteroid_future_x, asteroid_future_y):
            # Use the same collision prediction function as we use with the ship
            t1, t2 = collision_prediction(mine_x, mine_y, 0, 0, MINE_BLAST_RADIUS - 80, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'])
            #print(t1, t2)
            # Assuming the two times exist, the first time is when the collision starts, and the second time is when the collision ends
            # All in between times is where the circles are inside of each other (intersects)
            # We want to check whether the mine's blast radius is intersecting with the asteroid at the future time
            if not math.isnan(t1) and not math.isnan(t2) and min(t1, t2) < future_check_timesteps*DELTA_TIME < max(t1, t2):
                # A collision exists, and it'll happen when the mine is detonating
                count += 1
    return count

def predict_ship_mine_collision(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, mine, future_timesteps=0):
    if mine['remaining_time'] >= future_timesteps*DELTA_TIME:
        # Project the ship to its future location when the mine is blowing up
        ship_pos_x += ship_vel_x*(mine['remaining_time'] - future_timesteps*DELTA_TIME)
        ship_pos_y += ship_vel_y*(mine['remaining_time'] - future_timesteps*DELTA_TIME)
        if check_collision(ship_pos_x, ship_pos_y, SHIP_RADIUS, mine['position'][0], mine['position'][1], MINE_BLAST_RADIUS, COLLISION_CHECK_PAD):
            #print(f"\nMINE WILL COLLID IN {mine['remaining_time']} s")
            return mine['remaining_time']
        else:
            return math.inf
    else:
        # This mine exploded in the past, so won't ever collide
        return math.inf

def calculate_timesteps_until_bullet_hits_asteroid(time_until_asteroid_center_s, asteroid_radius):
    # time_until_asteroid_center is the time it takes for the bullet to travel from the center of the ship to the center of the asteroid
    # The bullet originates from the ship's edge, and the collision can happen as early as it touches the radius of the asteroid
    # We have to add 1, because it takes 1 timestep for the bullet to originate at the start, before it starts moving
    return 1 + ceil((time_until_asteroid_center_s*BULLET_SPEED - asteroid_radius - SHIP_RADIUS)/BULLET_SPEED/DELTA_TIME)

#@line_profiler.profile
def asteroid_bullet_collision(bullet_head_position, bullet_tail_position, asteroid_center, asteroid_radius):
    # This is an optimized version of circle_line_collision() from the Kessler source code
    # First, do a rough check if there's no chance the collision can occur
    # Avoid the use of min/max because it should be a bit faster
    if bullet_head_position[0] < bullet_tail_position[0]:
        x_min = bullet_head_position[0] - asteroid_radius
        if asteroid_center[0] < x_min:
            return False
        x_max = bullet_tail_position[0] + asteroid_radius
    else:
        x_min = bullet_tail_position[0] - asteroid_radius
        if asteroid_center[0] < x_min:
            return False
        x_max = bullet_head_position[0] + asteroid_radius
    if asteroid_center[0] > x_max:
        return False

    if bullet_head_position[1] < bullet_tail_position[1]:
        y_min = bullet_head_position[1] - asteroid_radius
        if asteroid_center[1] < y_min:
            return False
        y_max = bullet_tail_position[1] + asteroid_radius
    else:
        y_min = bullet_tail_position[1] - asteroid_radius
        if asteroid_center[1] < y_min:
            return False
        y_max = bullet_head_position[1] + asteroid_radius
    if asteroid_center[1] > y_max:
        return False

    # A collision is possible.
    # Create a triangle between the center of the asteroid, and the two ends of the bullet.
    a = round(dist(bullet_head_position, asteroid_center), 4)
    b = round(dist(bullet_tail_position, asteroid_center), 4)
    c = BULLET_LENGTH

    # Heron's formula to calculate area of triangle and resultant height (distance from circle center to line segment)
    s = 0.5*(a + b + c)

    squared_area = s*(s - a)*(s - b)*(s - c)
    triangle_height = 2/c*sqrt(squared_area)

    # If triangle's height is less than the asteroid's radius, the bullet is colliding with it
    return triangle_height < asteroid_radius

@lru_cache() # This function gets called with the same params all the time, so just cache the return value the first time
def get_simulated_ship_max_range(max_cruise_seconds):
    dummy_game_state = {}
    dummy_ship_state = {'speed': 0, 'position': (0, 0), 'velocity': (0, 0), 'heading': 0, 'bullets_remaining': 0, 'lives_remaining': 1}
    max_ship_range_test = Simulation(dummy_game_state, dummy_ship_state, 0)
    max_ship_range_test.accelerate(SHIP_MAX_SPEED)
    max_ship_range_test.cruise(round(max_cruise_seconds/DELTA_TIME))
    max_ship_range_test.accelerate(0)
    state_sequence = max_ship_range_test.get_state_sequence()
    #print(state_sequence[0])
    ship_random_range = dist(state_sequence[0]['position'], state_sequence[-1]['position'])
    ship_random_max_maneuver_length = len(state_sequence)
    return ship_random_range, ship_random_max_maneuver_length

def simulate_ship_movement_with_inputs(game_state, ship_state, move_sequence):
    dummy_game_state = {'asteroids': [], 'mines': [], 'map_size': game_state['map_size']}
    ship_movement_sim = Simulation(dummy_game_state, ship_state, 0, 0, {}, [], -math.inf, -math.inf, False, False)
    ship_movement_sim.apply_move_sequence(move_sequence)
    return ship_movement_sim.get_ship_state()

def get_adversary_interception_time_lower_bound(asteroid, adversary_ships, game_state):
    if not adversary_ships:
        return math.inf
    feasible, _, aiming_timesteps_required, interception_time_s, _, _, _ = solve_interception(asteroid, adversary_ships[0], game_state, 0)
    if feasible:
        #print(f"ADVERSARY INTERCEPT TIME: {interception_time_s + aiming_timesteps_required*DELTA_TIME}")
        return interception_time_s + aiming_timesteps_required*DELTA_TIME
    else:
        return math.inf

def solve_interception(asteroid, ship_state, game_state, timesteps_until_can_fire: int=0):
    #debug_print(f"\nChecking whether the asteroid {ast_to_string(asteroid)} is feasible to intercept!")
    # The bullet's head originates from the edge of the ship's radius.
    # We want to set the position of the bullet to the center of the bullet, so we have to do some fanciness here so that at t=0, the bullet's center is where it should be
    t_0 = (SHIP_RADIUS - BULLET_LENGTH/2)/BULLET_SPEED
    # Positions are relative to the ship. We set the origin to the ship's position. Remember to translate back!
    origin_x = ship_state['position'][0]
    origin_y = ship_state['position'][1]
    avx = asteroid['velocity'][0]
    avy = asteroid['velocity'][1]
    ax = asteroid['position'][0] - origin_x + avx*DELTA_TIME # We project the asteroid one timestep ahead, since by the time we shoot our bullet, the asteroid would have moved one more timestep!
    ay = asteroid['position'][1] - origin_y + avy*DELTA_TIME

    vb = BULLET_SPEED
    #tr = radians(SHIP_MAX_TURN_RATE) # rad/s THIS IS JUST PI
    vb_sq = vb*vb
    theta_0 = radians(ship_state['heading'])

    # Calculate constants for naive_desired_heading_calc
    A = avx*avx + avy*avy - vb_sq

    # Calculate constants for root_function, root_function_derivative, root_function_second_derivative
    k1 = ay*vb - avy*vb*t_0
    k2 = ax*vb - avx*vb*t_0
    k3 = avy*ax - avx*ay

    def naive_desired_heading_calc(timesteps_until_can_fire: int=0):
        time_until_can_fire_s = timesteps_until_can_fire*DELTA_TIME
        ax_delayed = ax + time_until_can_fire_s*avx # We add a delay to account for the timesteps until we can fire delay
        ay_delayed = ay + time_until_can_fire_s*avy

        # A is calculated outside of this function since it's a constant
        B = 2*(ax_delayed*avx + ay_delayed*avy - vb_sq*t_0)
        C = ax_delayed*ax_delayed + ay_delayed*ay_delayed - vb_sq*t_0*t_0

        t1, t2 = solve_quadratic(A, B, C)
        positive_interception_times = []
        if t1 and t1 >= 0:
            positive_interception_times.append(t1)
        if t2 and t2 >= 0:
            positive_interception_times.append(t2)
        solutions = []
        for t in positive_interception_times:
            x = ax_delayed + t*avx
            y = ay_delayed + t*avy
            theta = atan2(y, x)
            # If the asteroid is out of bounds, then it will wrap around and this shot isn't feasible
            # However, if an unwrapped asteroid was passed into this function and the interception is inbounds, then it's a feasible shot
            intercept_x = x + origin_x
            intercept_y = y + origin_y
            solutions.append((t, angle_difference_rad(theta, theta_0), timesteps_until_can_fire, None, intercept_x, intercept_y, None))
        return solutions
        # Returned tuple is (interception time in seconds from firing to hit, delta theta rad, timesteps until can fire, None, intercept_x, intercept_y, None)

    def naive_root_function(theta, time_until_can_fire_s: int=0):
        # Can be optimized more by expanding out the terms
        # Convert heading error to absolute heading
        theta += theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        ax_delayed = ax + time_until_can_fire_s*avx # We add a delay to account for the timesteps until we can fire delay
        ay_delayed = ay + time_until_can_fire_s*avy
        return (vb*cos_theta - avx)*(ay_delayed - vb*t_0*sin_theta) - (vb*sin_theta - avy)*(ax_delayed - vb*t_0*cos_theta)

    def naive_time_function(theta):
        # Convert heading error to absolute heading
        theta += theta_0
        return (ax + ay - vb*t_0*(sin(theta) + cos(theta)))/(vb*(sin(theta) + cos(theta)) - avx - avy)

    def naive_time_function_for_plotting(theta):
        return max(-200000, min(naive_time_function(theta)*100000, 200000))

    def root_function(theta):
        # Convert heading error to absolute heading
        theta += theta_0
        # Domain of this function is theta_0 - pi to theta_0 + pi
        # Make this function periodic by wrapping inputs outside this range, to within the range
        if not (theta_0 - pi <= theta <= theta_0 + pi):
            theta = (theta - theta_0 + pi)%(2*pi) - pi + theta_0
        abs_delta_theta = abs(theta - theta_0)
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        sinusoidal_component = k1*cos_theta - k2*sin_theta + k3
        wacky_component = vb*abs_delta_theta/pi*(avy*cos_theta - avx*sin_theta)
        return sinusoidal_component + wacky_component
        
    def root_function_derivative(theta):
        # Convert heading error to absolute heading
        theta += theta_0
        # Domain of this function is theta_0 - pi to theta_0 + pi
        # Make this function periodic by wrapping inputs outside this range, to within the range
        if not (theta_0 - pi <= theta <= theta_0 + pi):
            theta = (theta - theta_0 + pi)%(2*pi) - pi + theta_0
        
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        sinusoidal_component = -k1*sin_theta - k2*cos_theta
        wacky_component = -vb*np.sign(theta - theta_0)/pi*(avx*sin_theta - avy*cos_theta + (theta - theta_0)*(avx*cos_theta + avy*sin_theta))
        return sinusoidal_component + wacky_component

    def root_function_second_derivative(theta):
        # Convert heading error to absolute heading
        theta += theta_0
        # Domain of this function is theta_0 - pi to theta_0 + pi
        # Make this function periodic by wrapping inputs outside this range, to within the range
        if not (theta_0 - pi <= theta <= theta_0 + pi):
            theta = (theta - theta_0 + pi)%(2*pi) - pi + theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        sinusoidal_component = -k1*cos_theta + k2*sin_theta
        wacky_component = -vb*np.sign(theta - theta_0)/pi*(2*avx*cos_theta + 2*avy*sin_theta - (theta - theta_0)*(avx*sin_theta - avy*cos_theta))
        return sinusoidal_component + wacky_component

    def turbo_rootinator_5000(initial_guess, function, derivative_function, second_derivative_function, tolerance=EPS, max_iterations=4):
        # theta_new = theta_old - f(theta_old)/f'(theta_old)
        #theta_old = (initial_guess + pi)%(2*pi) - pi
        theta_old = initial_guess
        #debug_print(f"Our initial guess is {initial_guess} which gives function value {function(initial_guess)}")
        initial_func_value = None
        for iteration in range(max_iterations):
            func_value = function(theta_old)
            if abs(func_value) < TAD:
                return theta_old, iteration
            if not initial_func_value:
                initial_func_value = func_value
            derivative_value = derivative_function(theta_old)
            second_derivative_value = second_derivative_function(theta_old)
            
            # Avoid division by zero
            if derivative_value == 0:
                raise ValueError("Derivative is zero. Rootinator fails :(")

            # Update the estimate using Halley's method
            denominator = 2*derivative_value*derivative_value - func_value*second_derivative_value
            if denominator == 0:
                return None, 0
            theta_new = theta_old - (2*func_value*derivative_value)/denominator
            # The value has jumped past the periodic boundary. Clamp it to right past the boundary just so things don't get too crazy.
            if theta_new < -pi:
                theta_new = pi - GRAIN
            elif pi < theta_new:
                theta_new = -pi + GRAIN
            elif -pi <= theta_old <= 0 and 0 <= theta_new <= pi:
                # The value jumped past the kink in the middle of the graph. Set it to right past the kink so the value doesn't jump around like crazy
                theta_new = GRAIN
            elif 0 <= theta_old <= pi and -pi <= theta_new <= 0:
                theta_new = -GRAIN
            
            #debug_print(f"After iteration {iteration + 1}, our new theta value is {theta_new}. Func value is {func_value}")
            # Check for convergence
            # It converged if the theta value isn't changing much, and the function itself takes a value that is close to zero (magnitude at most 1% of the original func value)
            if abs(theta_new - theta_old) < tolerance and abs(func_value) < abs(initial_func_value)/10:
                return theta_new, (iteration + 1)

            theta_old = theta_new
        return None, 0

    def rotation_time(delta_theta_rad):
        return abs(delta_theta_rad)/radians(SHIP_MAX_TURN_RATE)

    def bullet_travel_time(theta, t_rot):
        # Convert heading error to absolute heading
        theta += theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        denom_x = avx - vb*cos_theta
        denom_y = avy - vb*sin_theta
        if denom_x == 0 and denom_y == 0:
            return math.inf
        if abs(denom_x) > abs(denom_y):
            t_bul = (vb*t_0*cos_theta - ax - avx*t_rot)/denom_x
        else:
            t_bul = (vb*t_0*sin_theta - ay - avy*t_rot)/denom_y
        return t_bul

    def bullet_travel_time_for_plot(theta):
        # Convert heading error to absolute heading
        theta += theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        t_rot = rotation_time(theta - theta_0)

        denom = (avx*sin_theta - avy*cos_theta)
        if denom == 0:
            return math.inf
        else:
            return max(-200000, min(((cos_theta*(ay + avy*t_rot) - sin_theta*(ax + avx*t_rot))/denom)*100000, 200000))

    def plot_function():
        naive_theta_ans_list = naive_desired_heading_calc(timesteps_until_can_fire)  # Assuming this function returns a list of angles
        theta_0 = radians(ship_state['heading'])
        theta_range = np.linspace(theta_0 - pi, theta_0 + pi, 400)
        theta_delta_range = np.linspace(-pi, pi, 400)

        # Vectorize the functions for numpy compatibility
        vectorized_function = np.vectorize(root_function)
        vectorized_derivative = np.vectorize(root_function_derivative)
        vectorized_second_derivative = np.vectorize(root_function_second_derivative)
        vectorized_bullet_time = np.vectorize(bullet_travel_time_for_plot)
        vectorized_naive_function = np.vectorize(naive_root_function)
        vectorized_naive_time = np.vectorize(naive_time_function_for_plotting)

        # Calculate function values
        function_values = vectorized_function(theta_delta_range)
        derivative_values = vectorized_derivative(theta_delta_range)
        alt_derivative_values = vectorized_second_derivative(theta_delta_range)
        bullet_times = vectorized_bullet_time(theta_delta_range)
        naive_function_values = vectorized_naive_function(theta_delta_range, timesteps_until_can_fire*DELTA_TIME)
        naive_times = vectorized_naive_time(theta_delta_range)

        plt.figure(figsize=(12, 6))

        # Plot the function and its derivatives
        plt.plot(theta_delta_range, function_values, label="Function")
        plt.plot(theta_delta_range, derivative_values, label="Derivative", color="orange")
        #plt.plot(theta_delta_range, alt_derivative_values, label="Second Derivative", color="blue", linestyle=':')
        plt.plot(theta_delta_range, bullet_times, label="Bullet Time", color="green", linestyle='-')
        plt.plot(theta_delta_range, naive_function_values, label="Naive Function", color="magenta", linestyle='-')
        plt.plot(theta_delta_range, naive_times, label="Naive Times", color="purple", linestyle='-')

        # Add vertical lines for each naive_theta_ans
        fudge = 0
        for theta_ans in naive_theta_ans_list:
            plt.axvline(x=theta_ans[1] + fudge, color='yellow', linestyle='--', label=f"Naive Theta Ans at {theta_ans[1]:.2f}")

            zero, iterations = turbo_rootinator_5000(theta_ans[1] + fudge, root_function, root_function_derivative, root_function_second_derivative, TAD, 15)
            if zero:
                delta_theta_solution = zero
                if not (-pi <= delta_theta_solution <= pi):
                    #print(f"SOLUTION WAS OUT OUT BOUNDS AT {delta_theta_solution} AND WRAPPED TO -pi, pi")
                    delta_theta_solution = (delta_theta_solution + pi)%(2*pi) - pi
                plt.axvline(x=delta_theta_solution, color='green', linestyle='--', label=f"Theta Ans Converged at {delta_theta_solution:.2f} after {iterations} iterations")
            else:
                pass
                #print('Root finder gave up rip')

        # Add a horizontal line at y=0
        plt.axhline(y=0, color='black', linewidth=1.5, label="y=0")

        plt.xlabel("Theta")
        plt.ylabel("Values")
        plt.title("Function and Derivatives Plot")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    #print('PLOTTING FUNCTION!')
    #plot_function()
    valid_solutions = []
    #print_debug = False
    naive_solutions = naive_desired_heading_calc(timesteps_until_can_fire)
    #debug_print(f'\n\nALL OUR NAIVE SOLUTIONS:, keep in mind timesteps until can fire is {timesteps_until_can_fire} so we cant go below that', naive_solutions)
    amount_we_can_turn_before_we_can_shoot_rad = radians(timesteps_until_can_fire*DELTA_TIME*SHIP_MAX_TURN_RATE)
    for naive_solution in naive_solutions:
        #debug_print("Evaluating naive solution:", naive_solution)
        if abs(naive_solution[1]) <= amount_we_can_turn_before_we_can_shoot_rad + EPS:
            # The naive solution works because there's no turning delay
            #debug_print('Naive solution works!', naive_solution)
            if check_coordinate_bounds(game_state, naive_solution[4], naive_solution[5]):
                valid_solutions.append((True, degrees(naive_solution[1]), timesteps_until_can_fire, naive_solution[0], naive_solution[4], naive_solution[5], None))
        else:
            if abs(avx) < GRAIN and abs(avy) < GRAIN:
                # The asteroid is pretty much stationary. Naive solution works fine.
                #debug_print("The asteroid is pretty much stationary. Naive solution works fine.")
                sol = naive_solution[1]
            else:
                # Use more advanced solution
                #debug_print('Using more advanced root finder')
                sol, _ = turbo_rootinator_5000(naive_solution[1], root_function, root_function_derivative, root_function_second_derivative, TAD, 4)
                #debug_print('Root finder gave us:', sol)
            if not sol:
                continue
            delta_theta_solution = sol
            absolute_theta_solution = delta_theta_solution + theta_0
            if ENABLE_ASSERTIONS:
                assert (-pi <= delta_theta_solution <= pi)
            #if not (-pi <= delta_theta_solution <= pi):
                #debug_print(f"SOLUTION WAS OUT OUT BOUNDS AT {delta_theta_solution} AND WRAPPED TO -pi, pi")
                #delta_theta_solution = (delta_theta_solution + pi)%(2*pi) - pi
            # Check validity of solution to make sure time is positive and stuff
            delta_theta_solution_deg = degrees(delta_theta_solution)
            t_rot = rotation_time(delta_theta_solution)
            if ENABLE_ASSERTIONS:
                assert math.isclose(t_rot, abs(delta_theta_solution_deg)/SHIP_MAX_TURN_RATE, abs_tol=EPS)
            t_bullet = bullet_travel_time(delta_theta_solution, t_rot)
            #debug_print(f't_bullet: {t_bullet}')
            if t_bullet < 0:
                continue
            #t_total = t_rot + t_bullet
            
            intercept_x = origin_x + vb*cos(absolute_theta_solution)*(t_bullet + t_0)
            intercept_y = origin_y + vb*sin(absolute_theta_solution)*(t_bullet + t_0)
            #debug_print(f"Intercept_x ({intercept_x}) = origin_x ({origin_x}) + vb*cos({absolute_theta_solution})*(t_bullet ({t_bullet}) + t_0 ({t_0}))")
            #debug_print(f"Intercept_y ({intercept_y}) = origin_y ({origin_y}) + vb*sin({absolute_theta_solution})*(t_bullet ({t_bullet}) + t_0 ({t_0}))")

            feasible = check_coordinate_bounds(game_state, intercept_x, intercept_y)
            if feasible:
                #debug_print(f"The coordinates of {intercept_x}, {intercept_y} are GUCCI! We'd have to turn this many ts: {t_rot/delta_time}")
                # Since half timesteps don't exist, we need to discretize this solution by rounding up the amount of timesteps, and now we can use the naive method to confirm and get the exact angle
                # We max this with ts until can fire, because that's the floor and we can't go below it
                t_rot_ts = max(timesteps_until_can_fire, ceil(t_rot/DELTA_TIME))
                #debug_print(f"The rotation timesteps we've calculated is {t_rot_ts}, from a t_rot of {t_rot}")
                #valid_solutions.append((True, delta_theta_solution_deg, t_rot_ts, None, intercept_x, intercept_y, None))
                proper_discretized_solutions = naive_desired_heading_calc(t_rot_ts)
                for disc_sol in proper_discretized_solutions:
                    # Only expecting there to be one
                    if not abs(degrees(disc_sol[1])) - EPS <= t_rot_ts*DELTA_TIME*SHIP_MAX_TURN_RATE:
                        continue
                    #    print_debug = True
                    #    #print(f"Good tbullet: {t_bullet}, t_bullet_1: {t_bullet_1} with denom {(avx - vb*cos(sol))}, t_bullet_2: {t_bullet_2} with denom {(avy - vb*sin(sol))}")
                    #    print(f"About to fail assertion! degrees required to turn: {abs(degrees(disc_sol[1]))} isn't at most the amount we can rotate: {t_rot_ts*delta_time*ship_max_turn_rate}")
                    #    print('New sol:')
                    #    print((True, degrees(disc_sol[1]), t_rot_ts, disc_sol[0], disc_sol[4], disc_sol[5], None))
                    #    print('Old continuous sol:')
                    #    print((feasible, delta_theta_solution_deg, t_rot/delta_time, t_total, intercept_x, intercept_y, None))
                    #    plot_function()
                    #if not t_rot_ts == disc_sol[2]:
                    #    print_debug = True
                    #    print(f"About to fail assertion! TS we need to rotate ceiled: {t_rot_ts} isn't equal to our discretized solution of {disc_sol[2]}")
                    #    print((True, degrees(disc_sol[1]), t_rot_ts, disc_sol[0], disc_sol[4], disc_sol[5], None))
                    #    #plot_function()
                    #assert abs(degrees(disc_sol[1])) - eps <= t_rot_ts*delta_time*ship_max_turn_rate
                    # disc_sol tuple is (interception time in seconds from firing to hit, delta theta rad, timesteps until can fire, None, intercept_x, intercept_y, None)
                    if ENABLE_ASSERTIONS:
                        assert t_rot_ts == disc_sol[2]
                    if check_coordinate_bounds(game_state, disc_sol[4], disc_sol[5]):
                        #debug_print('Valid solution found!', disc_sol)
                        valid_solutions.append((True, degrees(disc_sol[1]), t_rot_ts, disc_sol[0], disc_sol[4], disc_sol[5], None))
            else:
                pass
                #debug_print(f"The coordinates of {intercept_x}, {intercept_y} are outside the bounds! Invalid solution. We'd have to turn this many ts: {t_rot/DELTA_TIME}")

    sorted_solutions = sorted(valid_solutions, key=lambda x: x[2]) # Sort by aiming timesteps required
    #print('Sorted solutions at the end:')
    #print(sorted_solutions)
    if sorted_solutions:
        #if print_debug:
        #    pass
            #print('ALL SORTED SOLUTIONS:')
            #print(sorted_solutions)
            #print('Finally, printing the inputs to this problem:')
            #print("asteroid, ship_state, game_state, timesteps_until_can_fire")
            #print(asteroid, ship_state, game_state, timesteps_until_can_fire)
            #print()
            #plot_function()
        # TODO: Maybe return all solutions just so we have more options
        return sorted_solutions[0]
    else:
        return False, None, None, None, None, None, None
    # return feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception

def track_asteroid_we_shot_at(asteroids_pending_death, current_timestep, game_state, bullet_travel_timesteps, asteroid, wrap=True):
    asteroid = dict(asteroid)
    # Project the asteroid into the future, to where it would be on the timestep of its death
    
    for future_timesteps in range(0, bullet_travel_timesteps + 1):
        if wrap:
            # Wrap asteroid position to get the canonical asteroid
            asteroid['position'] = wrap_position(asteroid['position'], game_state['map_size'])
        #print(asteroid)
        timestep = current_timestep + future_timesteps
        if timestep not in asteroids_pending_death:
            asteroids_pending_death[timestep] = [dict(asteroid)]
        else:
            #debug_print(f"Future ts: {future_timesteps}")
            #if is_asteroid_in_list(asteroids_pending_death[timestep], asteroid):
                #debug_print(f"WARNING: ASTEREOID ALREWADY  IN LIST ")
            if ENABLE_ASSERTIONS:
                if is_asteroid_in_list(asteroids_pending_death[timestep], asteroid):
                    print(f'ABOUT TO FAIL ASSERTION, we are in the future by {future_timesteps} timesteps, LIST FOR THIS TS IS:')
                    print(asteroids_pending_death[timestep])
                    pass
                assert not is_asteroid_in_list(asteroids_pending_death[timestep], asteroid)
            asteroids_pending_death[timestep].append(dict(asteroid))
        # Advance the asteroid to the next position
        if future_timesteps != bullet_travel_timesteps:
            # Skip this operation on the last iteration
            asteroid['position'] = (asteroid['position'][0] + asteroid['velocity'][0]*DELTA_TIME, asteroid['position'][1] + asteroid['velocity'][1]*DELTA_TIME)
    return asteroids_pending_death

def check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(asteroids_pending_death, current_timestep, game_state, asteroid, wrap=False):
    # Check whether the asteroid has already been shot at, or if we can shoot at it again
    if wrap:
        asteroid = dict(asteroid)
        asteroid['position'] = wrap_position(asteroid['position'], game_state['map_size'])
    else:
        if ENABLE_ASSERTIONS:
            assert check_coordinate_bounds(game_state, asteroid['position'][0], asteroid['position'][1])
    if current_timestep not in asteroids_pending_death:
        #print(f"This asteroid was NOT shot at and IS IN THE CLEAR! {ast_to_string(asteroid)}")
        return True
    else:
        if not is_asteroid_in_list(asteroids_pending_death[current_timestep], asteroid):
            #print(f"This asteroid was NOT shot at and IS IN THE CLEAR! {ast_to_string(asteroid)}")
            pass
        return not is_asteroid_in_list(asteroids_pending_death[current_timestep], asteroid)

def time_travel_asteroid(asteroid, timesteps, game_state=None, wrap=False):
    asteroid = dict(asteroid) # TODO: Unsure whether this copy is necessary
    if wrap and game_state is not None:
        asteroid['position'] = wrap_position((asteroid['position'][0] + asteroid['velocity'][0]*timesteps*DELTA_TIME, asteroid['position'][1] + asteroid['velocity'][1]*timesteps*DELTA_TIME), game_state['map_size'])
    else:
        asteroid['position'] = (asteroid['position'][0] + asteroid['velocity'][0]*timesteps*DELTA_TIME, asteroid['position'][1] + asteroid['velocity'][1]*timesteps*DELTA_TIME)
    return asteroid

def check_mine_opportunity(ship_state, game_state):
    if ship_state['mines_remaining'] == 0 or len(game_state['mines']) > 1:# or 'lives_remaining' not in ship_state:
        return False
    #average_asteroid_density = len(game_state['asteroids'])/(game_state['map_size'][0]*game_state['map_size'][1])
    #average_asteroids_inside_blast_radius = average_asteroid_density*pi*MINE_BLAST_RADIUS**2
    #if average_asteroids_inside_blast_radius > 5:
    #    return True
    mine_ast_count = count_asteroids_in_mine_blast_radius(game_state, ship_state['position'][0], ship_state['position'][1], round(MINE_FUSE_TIME/DELTA_TIME))
    #debug_print(f"Mine count inside: {mine_ast_count} compared to average density amount inside: {average_asteroids_inside_blast_radius}")
    return mine_fis(ship_state['mines_remaining'], ship_state['lives_remaining'], mine_ast_count)
    #if (len(game_state['asteroids']) > 40 and mine_ast_count > 1.5*average_asteroids_inside_blast_radius or mine_ast_count > 20 or mine_ast_count > 2*average_asteroids_inside_blast_radius) and len(game_state['mines']) == 0:
    #    return True
    #else:
    #    return False

class Simulation():
    # Simulates kessler_game.py and ship.py and other game mechanics
    def __init__(self, game_state: dict, ship_state: dict, initial_timestep: int, respawn_timer: float=0, asteroids_pending_death: dict=None, forecasted_asteroid_splits: list=None, last_timestep_fired = -math.inf, last_timestep_mined = -math.inf, halt_shooting: bool=False, fire_first_timestep: bool=False, game_state_plotter: GameStatePlotter | None=None):
        #print(f"INITIALIZING SIMULATION, fire first timestep is: {fire_first_timestep}")
        #print(f"STARTING SIMULATION ON TIMESTEP {initial_timestep}")
        #if initial_timestep == 42:
        #    print(f"On TS 42 we have the bullets being:", bullets)
        debug_print(f"Starting sim on ts {initial_timestep} with ship state:", ship_state)
        if asteroids_pending_death is None:
            asteroids_pending_death = {}
        if forecasted_asteroid_splits is None:
            forecasted_asteroid_splits = []
        self.initial_timestep = initial_timestep
        self.future_timesteps = 0
        self.last_timestep_fired = last_timestep_fired
        self.last_timestep_mined = last_timestep_mined
        #debug_print('Heres the asteroid list when coming in:')
        #debug_print(asteroids)
        #self.game_state['mines'] = [dict(m) for m in mines]
        #print(other_ships)
        #self.other_ships = [dict(s) for s in other_ships]
        #self.game_state['bullets'] = [dict(b) for b in bullets]
        self.game_state = dict(game_state)
        self.ship_state = dict(ship_state)
        self.game_state['asteroids'] = [dict(a) for a in game_state['asteroids']]
        self.game_state['ships'] = [dict(s) for s in game_state['ships']]
        self.game_state['bullets'] = [dict(b) for b in game_state['bullets']]
        self.game_state['mines'] = [dict(m) for m in game_state['mines']]
        # TODO: Comprehension
        self.other_ships = []
        for ship in game_state['ships']:
            if ship['id'] != ship_state['id']:
                self.other_ships.append(ship)
        if ENABLE_ASSERTIONS:
            assert 0 <= len(self.other_ships) <= 1
        self.ship_move_sequence = []
        self.ship_pending_moves = []
        self.state_sequence = []
        self.asteroids_shot = 0
        self.asteroids_pending_death = copy.deepcopy(asteroids_pending_death)#{timestep: list[:] for timestep, list in asteroids_pending_death.items()}
        self.forecasted_asteroid_splits = copy.deepcopy(forecasted_asteroid_splits)#[dict(a) for a in forecasted_asteroid_splits]
        #self.timesteps_to_not_check_collision_for = timesteps_to_not_check_collision_for
        self.halt_shooting = halt_shooting
        self.fire_next_timestep_flag = False
        self.fire_first_timestep = fire_first_timestep
        self.game_state_plotter = game_state_plotter
        self.sim_id = random.randint(1, 100000)
        self.explanation_messages = []
        self.safety_messages = []
        self.respawn_timer = respawn_timer

    def get_explanations(self):
        return self.explanation_messages

    def get_safety_messages(self):
        return self.safety_messages

    def get_sim_id(self):
        return self.sim_id

    def get_respawn_timer(self):
        return self.respawn_timer

    def get_ship_state(self):
        #return {'position': self.ship_state['position'], 'velocity': self.ship_state['velocity'], 'speed': self.ship_state['speed'], 'heading': self.ship_state['heading'], 'bullets_remaining': self.ship_state['bullets_remaining'], 'mines_remaining': self.ship_state['mines_remaining'], 'lives_remaining': self.ship_state['lives_remaining']}
        return dict(self.ship_state)
    
    def get_game_state(self):
        # TODO: Might need a deepcopy for this to not bug out
        #return {key: copy.copy(val) for key, val in self.game_state.items()}
        return dict(self.game_state)

    def get_fire_next_timestep_flag(self):
        return self.fire_next_timestep_flag

    def set_fire_next_timestep_flag(self, fire_next_timestep_flag):
        self.fire_next_timestep_flag = fire_next_timestep_flag

    def get_asteroids_pending_death(self):
        return self.asteroids_pending_death

    def get_forecasted_asteroid_splits(self):
        return self.forecasted_asteroid_splits

    def get_instantaneous_asteroid_collision(self, asteroids: list=None, ship_position: tuple=None):
        if ship_position is not None:
            position = ship_position
        else:
            position = self.ship_state['position']
        for a in (asteroids if asteroids is not None else self.game_state['asteroids']):
            if check_collision(position[0], position[1], SHIP_RADIUS, a['position'][0], a['position'][1], a['radius'], COLLISION_CHECK_PAD):
                return True
        return False

    def get_instantaneous_ship_collision(self):
        for ship in self.other_ships:
            # The faster the other ship is going, the bigger of a bubble around it I'm going to draw, since they can deviate from their path very quickly and run into me even though I thought I was in the clear
            if check_collision(self.ship_state['position'][0], self.ship_state['position'][1], SHIP_RADIUS, ship['position'][0], ship['position'][1], ship['radius'] + SHIP_AVOIDANCE_PADDING + sqrt(ship['velocity'][0]**2 + ship['velocity'][1]**2)*SHIP_AVOIDANCE_SPEED_PADDING_RATIO, COLLISION_CHECK_PAD):
                return True
        return False

    def get_instantaneous_mine_collision(self):
        mine_collision = False
        mine_remove_idxs = []
        for i, m in enumerate(self.game_state['mines']):
            if m['remaining_time'] < EPS:
                if check_collision(self.ship_state['position'][0], self.ship_state['position'][1], SHIP_RADIUS, m['position'][0], m['position'][1], MINE_BLAST_RADIUS, COLLISION_CHECK_PAD):
                    mine_collision = True
                mine_remove_idxs.append(i)
        if mine_remove_idxs:
            self.game_state['mines'] = [mine for idx, mine in enumerate(self.game_state['mines']) if idx not in mine_remove_idxs]
        return mine_collision

    def get_next_extrapolated_asteroid_collision_time(self):
        #debug_print(f"Inside get fitness, we shot {self.asteroids_shot} asteroids. getting extrapolated collision time. The ship's velocity is: ({self.ship_state['velocity'][0]}, {self.ship_state['velocity'][1]})")
        # Assume constant velocity from here
        next_imminent_asteroid_collision_time = math.inf
        #print('Extrapolating stuff at rest in end')
        for asteroid in (self.game_state['asteroids'] + self.forecasted_asteroid_splits):
            #debug_print(f"Checking collision with asteroid: {ast_to_string(asteroid)}")
            #debug_print(f"Future timesteps: {self.future_timesteps}, timesteps to not check collision for: {self.timesteps_to_not_check_collision_for}")
            unwrapped_asteroids = unwrap_asteroid(asteroid, self.game_state['map_size'][0], self.game_state['map_size'][1], UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON)
            for a in unwrapped_asteroids:
                #if self.future_timesteps >= self.timesteps_to_not_check_collision_for:
                predicted_collision_time = predict_next_imminent_collision_time_with_asteroid(self.ship_state['position'][0], self.ship_state['position'][1], self.ship_state['velocity'][0], self.ship_state['velocity'][1], SHIP_RADIUS, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'])
                #else:
                    # The asteroids position was never updated, so we need to extrapolate it right now in one step
                    #debug_print(f"The asteroids position was never updated, so we need to extrapolate it right now in one step. My velocity is {self.ship_state['velocity']}")
                #    predicted_collision_time = predict_next_imminent_collision_time_with_asteroid(self.ship_state['position'][0], self.ship_state['position'][1], self.ship_state['velocity'][0], self.ship_state['velocity'][1], SHIP_RADIUS, a['position'][0] + self.future_timesteps*DELTA_TIME*a['velocity'][0], a['position'][1] + self.future_timesteps*DELTA_TIME*a['velocity'][1], a['velocity'][0], a['velocity'][1], a['radius'])
                #debug_print(f"The predicted collision time is {predicted_collision_time} s")
                if ENABLE_ASSERTIONS:
                    assert predicted_collision_time >= 0
                if math.isinf(predicted_collision_time):
                    continue
                # The predicted collision time is finite and non-negative
                if 'timesteps_until_appearance' in asteroid and asteroid['timesteps_until_appearance']*DELTA_TIME > predicted_collision_time + EPS:
                    # TODO: Probably off by one error, gotta verify this
                    # There is no collision since the asteroid is born after our predicted collision time, and an unborn asteroid can't collide with anything
                    debug_print("There is no collision since the unborn asteroid can't collide with anything")
                else:
                    if not check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, asteroid, True):
                        # We're already shooting the asteroid. Check whether the imminent collision time is before or after the asteroid is eliminated
                        predicted_collision_ts = floor(predicted_collision_time/DELTA_TIME)
                        future_asteroid_during_imminent_collision_time = time_travel_asteroid(a, predicted_collision_ts, self.game_state, True)
                        if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + predicted_collision_ts, self.game_state, future_asteroid_during_imminent_collision_time, True):
                            # In the future time the asteroid has already been eliminated, so there won't actually be a collision
                            debug_print("In the future time the asteroid has already been eliminated, so there won't actually be a collision")
                            continue
                        else:
                            next_imminent_asteroid_collision_time = min(next_imminent_asteroid_collision_time, predicted_collision_time)
                    else:
                        # We're not eliminating this asteroid, and if it was forecasted, it comes into existence before our collision time. Therefore our collision is real and should be considered.
                        next_imminent_asteroid_collision_time = min(next_imminent_asteroid_collision_time, predicted_collision_time)
            #else:
            #    debug_print(f"Inside extrapolated coll time checker. We already have a pending shot for this so we'll ignore this asteroid: {ast_to_string(asteroid)}")
        return next_imminent_asteroid_collision_time

    def get_next_extrapolated_mine_collision_time(self):
        next_imminent_mine_collision_time = math.inf
        for m in self.game_state['mines']:
            next_imminent_mine_collision_time = min(next_imminent_mine_collision_time, predict_ship_mine_collision(self.ship_state['position'][0], self.ship_state['position'][1], self.ship_state['velocity'][0], self.ship_state['velocity'][1], m, 0))
        return next_imminent_mine_collision_time

    def get_next_extrapolated_collision_time(self):
        next_imminent_asteroid_collision_time = self.get_next_extrapolated_asteroid_collision_time()
        next_imminent_mine_collision_time = self.get_next_extrapolated_mine_collision_time()
        next_imminent_collision_time = min(next_imminent_asteroid_collision_time, next_imminent_mine_collision_time)
        return next_imminent_collision_time

    def coordinates_in_same_wrap(self, pos1, pos2):
        # Checks whether the coordinates are in the same universe
        max_width = self.game_state['map_size'][0]
        max_height = self.game_state['map_size'][1]
        return pos1[0]//max_width == pos2[0]//max_width and pos1[1]//max_height == pos2[1]//max_height

    def get_fitness(self):
        # This will return a scalar number representing how good of an action/state sequence we just went through
        # If these moves will keep us alive for a long time and shoot many asteroids along the way, then the fitness is good
        # If these moves result in us getting into a dangerous spot, or if we don't shoot many asteroids at all, then the fitness will be bad
        # The LOWER the fitness score, the BETTER!
        move_sequence_length_s = (self.get_sequence_length() - 1)*DELTA_TIME
        next_extrapolated_asteroid_collision_time = self.get_next_extrapolated_asteroid_collision_time()
        next_extrapolated_mine_collision_time = self.get_next_extrapolated_mine_collision_time()
        safe_time_after_maneuver_s = min(next_extrapolated_asteroid_collision_time, next_extrapolated_mine_collision_time)
        asteroids_shot = self.asteroids_shot

        if asteroids_shot < 0:
            # This is to signal that we won't hit anything ever if we're staying here, so we should defer to the maneuver subcontroller to force a move
            debug_print(f"Deferring to maneuver subcontroller! Forcing a move.")
            asteroids_score = 1
        else:
            if asteroids_shot == 0:
                asteroids_shot = 0.5
            time_per_asteroids_shot = move_sequence_length_s/asteroids_shot
            asteroids_score = time_per_asteroids_shot/3
        
        states = self.get_state_sequence()
        
        ship_start_position = states[0]['ship_state']['position']
        ship_end_position = states[-1]['ship_state']['position']

        if len(states) >= 2:
            displacement = dist(ship_start_position, ship_end_position)
        else:
            displacement = 0
        displacement_score = displacement/1000 # TODO: If wrapped, this score will be disporportionately big. Ideally account for that by unwrapping somehow
        displacement_score = 0

        if displacement < EPS:
            # Stationary
            # Only has to be safe for 3 seconds to get the max score, to encourage staying put and eliminating threats by shooting rather than by maneuvering and missing shot opportunities
            asteroid_safe_time_score = max(0, min(5, 5 - 5/3*next_extrapolated_asteroid_collision_time))
        else:
            # Maneuvering
            asteroid_safe_time_score = max(0, min(5, 5 - next_extrapolated_asteroid_collision_time))
        # Regardless of stationary or maneuvering, the mine safe time score is calculated the same way
        if math.isinf(next_extrapolated_mine_collision_time):
            mine_safe_time_score = 0
        else:
            next_extrapolated_mine_collision_time = max(0, min(3, next_extrapolated_mine_collision_time))
            assert -EPS <= next_extrapolated_mine_collision_time <= 3 + EPS
            mine_safe_time_score = 5 - (5 - 1.5)/3*next_extrapolated_mine_collision_time

        other_ship_proximity_score = 0
        ship_proximity_max_penalty = 6
        ship_proximity_detection_radius = 400
        ship_prox_exponent = 3
        for other_ship in self.other_ships:
            other_ship_pos_x, other_ship_pos_y = other_ship['position']
            self_pos_x, self_pos_y = ship_end_position#self.ship_state['position']
            # Account for wrap. We know that the farthest two objects can be separated within the screen in the x and y axes, is by half the width, and half the height
            abs_sep_x = abs(self_pos_x - other_ship_pos_x)
            abs_sep_y = abs(self_pos_y - other_ship_pos_y)
            sep_x = min(abs_sep_x, self.game_state['map_size'][0] - abs_sep_x)
            sep_y = min(abs_sep_y, self.game_state['map_size'][1] - abs_sep_y)
            separation_dist = sqrt(sep_x*sep_x + sep_y*sep_y)
            if separation_dist < ship_proximity_detection_radius:
                other_ship_proximity_score += ship_proximity_max_penalty/(ship_proximity_detection_radius**ship_prox_exponent)*(ship_proximity_detection_radius - separation_dist)**ship_prox_exponent
        #print(f"Prox score: {other_ship_proximity_score}")
        sequence_length_score = move_sequence_length_s/10

        # Slightly discourage being too close to the edges
        edge_proximity_score_cap = 0.1
        # Find the edge I'm the closest to
        if ship_end_position[0]*2 < self.game_state['map_size'][0]:
            sep_x = ship_end_position[0]
        else:
            sep_x = self.game_state['map_size'][0] - ship_end_position[0]
        if ship_end_position[1]*2 < self.game_state['map_size'][1]:
            sep_y = ship_end_position[1]
        else:
            sep_y = self.game_state['map_size'][1] - ship_end_position[1]
        separation_dist = min(sep_x, sep_y)
        max_separation_dist = min(self.game_state['map_size'][0], self.game_state['map_size'][1])/2
        edge_proximity_score = edge_proximity_score_cap*(1 - separation_dist/max_separation_dist)

        
        debug_print(f"Fitness: {asteroid_safe_time_score + mine_safe_time_score + asteroids_score + sequence_length_score + displacement_score}, Ast safe time score: {asteroid_safe_time_score} (safe time after maneuver is {safe_time_after_maneuver_s} s, and current sim mode is {'stationary' if displacement < EPS else 'maneuver'}), asteroids score: {asteroids_score}, sequence length score: {sequence_length_score}, displacement score: {displacement_score}, other ship prox score: {other_ship_proximity_score}")
        #self.explanation_messages.append(f"Fitness: {asteroid_safe_time_score + mine_safe_time_score + asteroids_score + sequence_length_score + displacement_score}, Ast safe time score: {asteroid_safe_time_score} (safe time after maneuver is {safe_time_after_maneuver_s} s, mine safe time score: {mine_safe_time_score}, and current sim mode is {'stationary' if displacement < EPS else 'maneuver'}), asteroids score: {asteroids_score}, sequence length score: {sequence_length_score}, displacement score: {displacement_score}, other ship prox score: {other_ship_proximity_score}")
        debug_print(f"Fitness: {asteroid_safe_time_score + mine_safe_time_score + asteroids_score + sequence_length_score + displacement_score}, Ast safe time score: {asteroid_safe_time_score} (safe time after maneuver is {safe_time_after_maneuver_s} s, mine safe time score: {mine_safe_time_score}, and current sim mode is {'stationary' if displacement < EPS else 'maneuver'}), asteroids score: {asteroids_score}, sequence length score: {sequence_length_score}, displacement score: {displacement_score}, other ship prox score: {other_ship_proximity_score}")
        if asteroid_safe_time_score > 3:
            self.safety_messages.append("I'm dangerously close to being hit by asteroids. Trying my hardest to maneuver out of this situation.")
        elif asteroid_safe_time_score > 2:
            self.safety_messages.append("I'm close to being hit by asteroids.")
        elif asteroid_safe_time_score > 1:
            self.safety_messages.append("I'll eventually get hit by asteroids. Keeping my eye out for a dodge maneuver.")
        
        if mine_safe_time_score > 3:
            self.safety_messages.append("I'm dangerously close to being kablooied by a mine. Trying my hardest to maneuver out of this situation.")
        elif mine_safe_time_score > 2:
            self.safety_messages.append("I'm close to being boomed by a mine.")
        elif mine_safe_time_score > 1:
            self.safety_messages.append("I'm within the radius of a mine.")

        if other_ship_proximity_score > 3:
            self.safety_messages.append("I'm dangerously close to the other ship. Get away from me!")
        elif other_ship_proximity_score > 1:
            self.safety_messages.append("I'm close to the other ship. Being cautious.")
        
        if edge_proximity_score > 0.08:
            self.safety_messages.append("I'm really close to the edge. This is fine, but be careful!")

        overall_fitness = asteroid_safe_time_score + mine_safe_time_score + asteroids_score + sequence_length_score + displacement_score + other_ship_proximity_score + edge_proximity_score
        if overall_fitness < 1:
            self.safety_messages.append("I'm safe and chilling")
        return overall_fitness

    def find_extreme_shooting_angle_error(self, asteroid_list, threshold, mode='largest_below'):
        # Extract the shooting_angle_error_deg values
        shooting_angles = [d['shooting_angle_error_deg'] for d in asteroid_list]

        if mode == 'largest_below':
            # Find the index where threshold would be inserted
            idx = bisect.bisect_left(shooting_angles, threshold)

            # Adjust the index to get the largest value below the threshold
            if idx > 0:
                idx -= 1
            else:
                return None  # All values are greater than or equal to the threshold
        elif mode == 'smallest_above':
            # Find the index where threshold would be inserted
            idx = bisect.bisect_right(shooting_angles, threshold)

            # Check if all values are smaller than the threshold
            if idx >= len(shooting_angles):
                return None
        else:
            raise ValueError("Invalid mode. Choose 'largest_below' or 'smallest_above'.")

        # Return the corresponding dictionary
        return asteroid_list[idx]

    def target_selection(self) -> bool:
        global asteroid_size_shot_priority
        #print('\n\nGETTING INTO TARGET SELECTION')
        def simulate_shooting_at_target(target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps):
            # Just because we're lined up for a shot doesn't mean our shot will hit, unfortunately.
            # Bullets and asteroids travel in discrete timesteps, and it's possible for the bullet to miss the asteroid hitbox between timesteps, where the interception would have occurred on an intermediate timestep.
            # This is unavoidable, and we just have to choose targets that don't do this.
            # If the asteroids are moving slow enough, this should be rare, but especially if small asteroids are moving very quickly, this issue is common.
            # A simulation will easily show whether this will happen or not
            # TODO: CULLING
            debug_print(f"The last timestep fired is {self.last_timestep_fired}")
            aiming_move_sequence = self.get_rotate_heading_move_sequence(target_asteroid_shooting_angle_error_deg)
            if self.fire_first_timestep:
                timesteps_until_can_fire = max(0, 5 - len(aiming_move_sequence))
            else:
                timesteps_until_can_fire = max(0, 5 - (self.initial_timestep + self.future_timesteps + len(aiming_move_sequence) - self.last_timestep_fired))
            #if enable_assertions:
            #   assert timesteps_until_can_fire == 0
            debug_print(f'aiming move seq before append, and ts until can fire is {timesteps_until_can_fire}')
            debug_print(aiming_move_sequence)
            aiming_move_sequence.extend([{'thrust': 0, 'turn_rate': 0, 'fire': False} for _ in range(timesteps_until_can_fire)])
            debug_print('aiming move seq after append')
            debug_print(aiming_move_sequence)
            asteroid_advance_timesteps = len(aiming_move_sequence)
            debug_print(f"Asteroid advanced timesteps: {asteroid_advance_timesteps}")
            debug_print(f"Targetting turning timesteps: {target_asteroid_turning_timesteps}")
            if ENABLE_ASSERTIONS and target_asteroid_turning_timesteps != 0:
                assert asteroid_advance_timesteps <= target_asteroid_turning_timesteps
            if asteroid_advance_timesteps < target_asteroid_turning_timesteps:
                debug_print(f"asteroid_advance_timesteps {asteroid_advance_timesteps} < target_asteroid_turning_timesteps {target_asteroid_turning_timesteps}")
                #time.sleep(2)
                for _ in range(target_asteroid_turning_timesteps - asteroid_advance_timesteps):
                    debug_print('Waiting one more timestep!')
                    #print(f"APPENDING ANOTHER WAITING MOVE WHERE WE SHOULDNT FIRE")
                    aiming_move_sequence.append({'thrust': 0, 'turn_rate': 0, 'fire': False})
            target_asteroid = dict(target_asteroid)
            target_asteroid = time_travel_asteroid(target_asteroid, asteroid_advance_timesteps, self.game_state, True)
            #debug_print(f"We're targetting asteroid {ast_to_string(target_asteroid)}")
            #debug_print(f"Entering the bullet target sim, we're on timestep {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)}")
            #debug_print(self.game_state['asteroids'])
            #print('CURRENT SHIP STATE:')
            #print(self.get_ship_state())
            #print('APPLYING AIMING MOVE SEQ')
            #print(aiming_move_sequence)
            current_ship_state = self.get_ship_state()
            if not (abs(current_ship_state['velocity'][0]) < GRAIN and abs(current_ship_state['velocity'][1]) < GRAIN):
                debug_print(f"Current ship velocity is {current_ship_state['velocity']}")
            #if ENABLE_ASSERTIONS:
            #    if not abs(current_ship_state['velocity'][0]) < GRAIN and abs(current_ship_state['velocity'][1]) < GRAIN:
            #        print(f"Ship vel: {current_ship_state['velocity']}")
            #    assert abs(current_ship_state['velocity'][0]) < GRAIN and abs(current_ship_state['velocity'][1]) < GRAIN
            #ship_state_after_aiming_from_sim = simulate_ship_movement_with_inputs(self.game_state, current_ship_state, aiming_move_sequence)
            ship_state_after_aiming = current_ship_state
            #print(f"\nOld heading: {ship_state_after_aiming['heading']}, add error: {target_asteroid_shooting_angle_error_deg}")
            ship_state_after_aiming['heading'] = (ship_state_after_aiming['heading'] + target_asteroid_shooting_angle_error_deg)%360
            #print('sim:', ship_state_after_aiming_from_sim, 'shortcut:', ship_state_after_aiming)
            
            #if enable_assertions:
            #    assert math.isclose(ship_state_after_aiming_from_sim['heading']%360, ship_state_after_aiming['heading']%360, abs_tol=grain)
            #print('SHIP STTAT AFTER AIMING')
            #print(ship_state_after_aiming)
            # TODO: Maybe we can't ignore the variable of whether the ship was safe?
            actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, _ = self.bullet_target_sim(ship_state_after_aiming, self.fire_first_timestep, len(aiming_move_sequence))
            #if actual_asteroid_hit:
                #before_ship_state = self.get_ship_state()
                #print(f"During the bullet sim, the ship turned from heading {before_ship_state['heading']} to {ship_state_after_aiming['heading']}")
            return actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming

        aiming_underturn_allowance_deg = 10
        # First, find the most imminent asteroid
        #print('\nGOING INTO FUNCTION TO GET ALL FEASIBLE TARGETS FOR ASTEROIDS')
        target_asteroids_list = []
        dummy_ship_state = {'speed': 0, 'position': self.ship_state['position'], 'velocity': (0, 0), 'heading': self.ship_state['heading'], 'bullets_remaining': 0, 'lives_remaining': 1}
        if self.fire_first_timestep:
            timesteps_until_can_fire = 5
        else:
            timesteps_until_can_fire = max(0, 5 - (self.initial_timestep + self.future_timesteps - self.last_timestep_fired))
        debug_print(f"\nSimulation starting from timestep {self.initial_timestep + self.future_timesteps}, and we need to wait this many until we can fire: {timesteps_until_can_fire}")
        #debug_print('asteroids')
        #debug_print(self.game_state['asteroids'])
        #debug_print('Forecasetd splits')
        #debug_print(self.forecasted_asteroid_splits)
        #debug_print('bullets')
        #debug_print(self.game_state['bullets'])
        most_imminent_asteroid_exists = False
        asteroids_still_exist = False
        for asteroid in self.game_state['asteroids'] + self.forecasted_asteroid_splits:
            if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, asteroid, True):
                asteroids_still_exist = True
                #print(f"\nOn TS {self.initial_timestep + self.future_timesteps} We do not have a pending shot for the asteroid {ast_to_string(asteroid)}")
                unwrapped_asteroids = unwrap_asteroid(asteroid, self.game_state['map_size'][0], self.game_state['map_size'][1], UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON)
                # Iterate through all unwrapped asteroids to find which one of the unwraps is the best feasible target.
                # 99% of the time, only one of the unwraps will have a feasible target, but there's situations where we could either shoot the asteroid before it wraps, or wait for it to wrap and then shoot it.
                # In these cases, we need to pick whichever option is the fastest when factoring in turn time and waiting time.
                best_feasible_unwrapped_target = None
                for a in unwrapped_asteroids:
                    feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = solve_interception(a, dummy_ship_state, self.game_state, timesteps_until_can_fire)
                    
                    if feasible:
                        if best_feasible_unwrapped_target is None or aiming_timesteps_required < best_feasible_unwrapped_target[2]:
                            best_feasible_unwrapped_target = (feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception)
                    else:
                        pass
                        #debug_print('INFEASIBLE SHOT:')
                        #debug_print(feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception)
                if best_feasible_unwrapped_target is not None:
                    feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = best_feasible_unwrapped_target
                    imminent_collision_time_s = math.inf
                    for a in unwrapped_asteroids:
                        imminent_collision_time_s = min(imminent_collision_time_s, predict_next_imminent_collision_time_with_asteroid(self.ship_state['position'][0], self.ship_state['position'][1], self.ship_state['velocity'][0], self.ship_state['velocity'][1], SHIP_RADIUS, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'] + COLLISION_CHECK_PAD))
                    #print("APPENDING ASTERODI TO TARGETS LIST!!! SUCCESS!!")
                    target_asteroids_list.append({
                        'asteroid': dict(asteroid), # Record the canonical asteroid even if we're shooting at an unwrapped one
                        'feasible': feasible, # Will be True
                        'shooting_angle_error_deg': shooting_angle_error_deg,
                        'aiming_timesteps_required': aiming_timesteps_required,
                        'interception_time_s': interception_time_s,
                        'intercept_x': intercept_x,
                        'intercept_y': intercept_y,
                        'asteroid_dist_during_interception': asteroid_dist_during_interception,
                        'imminent_collision_time_s': imminent_collision_time_s,
                    })
                    if imminent_collision_time_s < math.inf:
                        debug_print(f"Imminent collision time is less than inf! {imminent_collision_time_s}")
                        most_imminent_asteroid_exists = True
        # Check whether we have enough time to aim at it and shoot it down
        # TODO: PROBLEM, what if the asteroid breaks into pieces and I need to shoot those down too? But I have plenty of time, and I still want the fitness function to be good in that case
        
        turn_angle_deg_until_can_fire = timesteps_until_can_fire*SHIP_MAX_TURN_RATE*DELTA_TIME # Can be up to 30 degrees
        #print(target_asteroids_list)
        # if there’s an imminent shot coming toward me, I will aim at the asteroid that gets me CLOSEST to the direction of the imminent shot.
        # So it only plans one shot at a time instead of a series of shots, and it’ll keep things simpler
        #debug_print(f"Least angular dist: {least_angular_distance_asteroid_shooting_angle_error_deg}")
        #debug_print('Target asts list: ', target_asteroids_list)
        actual_asteroid_hit = None
        aiming_move_sequence = []

        if most_imminent_asteroid_exists:
            # First try to shoot the most imminent asteroids, if they exist
            
            #debug_print(f"Shooting at most imminent asteroids. Most imminent collision time is {most_imminent_collision_time_s}s with turn angle error {most_imminent_asteroid_shooting_angle_error_deg}")
            #debug_print(most_imminent_asteroid)
            # Find the asteroid I can shoot at that gets me closest to the imminent shot, if I can't reach the imminent shot in time until I can shoot
            #print('\n\ntarget asteroids list')
            #print(target_asteroids_list)
            # Sort the targets such that we prioritize asteroids that are about to hit me
            
            sorted_imminent_targets = sorted(target_asteroids_list, key=lambda a: (
                min(10, a['imminent_collision_time_s']) +
                asteroid_size_shot_priority[a['asteroid']['size']]/4 +
                20*min(0.5, max(0, a['interception_time_s'] + a['aiming_timesteps_required']*DELTA_TIME - get_adversary_interception_time_lower_bound(a['asteroid'], self.other_ships, self.game_state)))
            ))
            #print('\nsorted imminent targets')
            #print(sorted_imminent_targets)
            # TODO: For each asteroid, give it a couple feasible times where we wait longer and longer. This way we can choose to wait a timestep to fire again if we'll get luckier with the bullet lining up
            for idx, candidate_target in enumerate(sorted_imminent_targets):
                if idx > 0:
                    debug_print(f"Checking candidate target iteration {idx+1}")
                if math.isinf(candidate_target['imminent_collision_time_s']):
                    debug_print("Ran through all imminent asteroids!")
                    break
                most_imminent_asteroid_aiming_timesteps = candidate_target['aiming_timesteps_required']
                most_imminent_asteroid = candidate_target['asteroid']
                most_imminent_asteroid_shooting_angle_error_deg = candidate_target['shooting_angle_error_deg']
                most_imminent_asteroid_interception_time_s = candidate_target['interception_time_s']
                debug_print(f"Shooting at asteroid that's going to hit me: {ast_to_string(most_imminent_asteroid)}")
                if most_imminent_asteroid_aiming_timesteps <= timesteps_until_can_fire:
                    if ENABLE_ASSERTIONS:
                        assert most_imminent_asteroid_aiming_timesteps == timesteps_until_can_fire
                    # I can reach the imminent shot without wasting a shot opportunity, so do it
                    actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming = simulate_shooting_at_target(most_imminent_asteroid, most_imminent_asteroid_shooting_angle_error_deg, most_imminent_asteroid_interception_time_s, most_imminent_asteroid_aiming_timesteps)
                    if actual_asteroid_hit:
                        actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid, self.game_state, True)
                        if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len(aiming_move_sequence), self.game_state, actual_asteroid_hit_when_firing, True):
                            break
                    #    print(f"DANG IT, the most imminent shot exists but we'll miss if we shoot now!")
                else:
                    # Between now and when I can shoot, I don't have enough time to aim at the imminent asteroid.
                    # Instead, find the closest asteroid along the way to shoot
                    # Sort by angular distance, with the unlikely tie broken by shot size
                    sorted_targets = sorted(target_asteroids_list, key=lambda a: (round(a['shooting_angle_error_deg']), asteroid_size_shot_priority[a['asteroid']['size']]))
                    #print('Sorted targets:')
                    #print(sorted_targets)
                    #debug_print(f"Turn angle deg until we can fire (max 30 degrees): {turn_angle_deg_until_can_fire}")
                    if most_imminent_asteroid_shooting_angle_error_deg > 0:
                        #debug_print("The imminent shot requires us to turn the ship to the left")
                        target = self.find_extreme_shooting_angle_error(sorted_targets, turn_angle_deg_until_can_fire, 'largest_below')
                        if target is None or target['shooting_angle_error_deg'] < 0 or target['shooting_angle_error_deg'] < turn_angle_deg_until_can_fire - aiming_underturn_allowance_deg:
                            # We're underturning too much, so instead find the next overturn
                            #debug_print("Underturning too much, so instead find the next overturn to the left")
                            target = self.find_extreme_shooting_angle_error(sorted_targets, turn_angle_deg_until_can_fire, 'smallest_above')
                    else:
                        #debug_print("The imminent shot requires us to turn the ship to the right")
                        target = self.find_extreme_shooting_angle_error(sorted_targets, -turn_angle_deg_until_can_fire, 'smallest_above')
                        #print("Found the next target to the right:")
                        #print(target)
                        if target is None or target['shooting_angle_error_deg'] > 0 or target['shooting_angle_error_deg'] > -turn_angle_deg_until_can_fire + aiming_underturn_allowance_deg:
                            # We're underturning too much, so instead find the next overturn
                            #debug_print("Underturning too much, so instead find the next overturn to the right")
                            target = self.find_extreme_shooting_angle_error(sorted_targets, -turn_angle_deg_until_can_fire, 'largest_below')
                            #print(target)
                    if target is not None:
                        #debug_print('As our target were choosing this one which will be on our way:')
                        #debug_print(target)
                        actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming = simulate_shooting_at_target(target['asteroid'], target['shooting_angle_error_deg'], target['interception_time_s'], target['aiming_timesteps_required'])
                        if actual_asteroid_hit:
                            actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid, self.game_state, True)
                            if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len(aiming_move_sequence), self.game_state, actual_asteroid_hit_when_firing, True):
                                break
                        #    print(f"DANG IT, we're shooting something on the way to the most imminent asteroid, but we'll miss this particular one!")
                    else:
                        # Just gonna have to waste shot opportunities and turn all the way
                        actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming = simulate_shooting_at_target(most_imminent_asteroid, most_imminent_asteroid_shooting_angle_error_deg, most_imminent_asteroid_interception_time_s, most_imminent_asteroid_aiming_timesteps)
                        if actual_asteroid_hit:
                            actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid, self.game_state, True)
                            if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len(aiming_move_sequence), self.game_state, actual_asteroid_hit_when_firing, True):
                                break
                        #    print(f"DANG IT, we're forced to turn to the most imminent shot, but we'll still miss if we do!")
            if actual_asteroid_hit:
                self.explanation_messages.append("Shooting at asteroid that is about to hit me.")
        if not actual_asteroid_hit:
            # Nothing has been hit from the imminent shots so far. Move down the list to trying for convenient shots.
            if target_asteroids_list:
                debug_print("Shooting at asteroid with least shot delay since there aren't imminent asteroids")
                self.explanation_messages.append("Shooting at asteroid with least shot delay since no asteroids are about to hit me.")
                # TODO: Can sort tiebreakers like key=lambda a: (a['aiming_timesteps_required'], a['size'], a['distance']))
                # Once I get more prioritization, I can make use of this choice to prioritize small asteroids, or ones that are far from me, or whatever!
                #print('BEFOER AND AFTER SORTING')
                #print(target_asteroids_list)
                # Sort by just convenience (and anything else I'd like)
                sorted_targets = sorted(target_asteroids_list, key=lambda a: (
                    a['aiming_timesteps_required'] +
                    asteroid_size_shot_priority[a['asteroid']['size']] +
                    20*min(0.5, max(0, a['interception_time_s'] + a['aiming_timesteps_required']*DELTA_TIME - get_adversary_interception_time_lower_bound(a['asteroid'], self.other_ships, self.game_state)))
                ))
                #print(sorted_targets)
                for target in sorted_targets:
                    least_shot_delay_asteroid = target['asteroid']
                    least_shot_delay_asteroid_shooting_angle_error_deg = target['shooting_angle_error_deg']
                    least_shot_delay_asteroid_interception_time_s = target['interception_time_s']
                    least_shot_delay_asteroid_aiming_timesteps = target['aiming_timesteps_required']
                    #print(sorted_targets)
                    #print(target)
                    actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming = simulate_shooting_at_target(least_shot_delay_asteroid, least_shot_delay_asteroid_shooting_angle_error_deg, least_shot_delay_asteroid_interception_time_s, least_shot_delay_asteroid_aiming_timesteps)
                    if actual_asteroid_hit is not None and check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, actual_asteroid_hit, True):
                        break
                #if not actual_asteroid_hit:
                #    print(f"DANG IT, we went through all asteroids with least shot delay and we'll miss them ALL!")
            else:
                # Ain't nothing to shoot at
                # TODO: IF THERE'S NOTHING TO SHOOT AT, TURN TOWARD THE CENTER! THIS WAY MAYBE there's a super fast one that I just keep not getting.
                debug_print('Nothing to shoot at!')
                self.explanation_messages.append("There's nothing I can feasibly shoot at!")
                # Pick a direction to turn the ship anyway, just to better line it up with more shots
                #for a in self.game_state['asteroids'] + self.forecasted_asteroid_splits:
                # TODO: THIS WORKS, BUT MAKE THIS SMARTER SO IT DOESN'T JUST ALWAYS TURN LEFT EVEN IF IT"S BETTER TO TURN RIGHRT
                if asteroids_still_exist:
                    if random.randint(1, 10) == 1:
                        self.explanation_messages.append("Asteroids exist but we can't hit them. Moving around a bit randomly.")
                        self.asteroids_shot -= 1
                    #turn_direction = random.random()
                    #idle_thrust = random.triangular(0, SHIP_MAX_THRUST, 0)
                    turn_direction = 0
                    idle_thrust = 0
                else:
                    #print('asteroids DO NOT exist')
                    self.explanation_messages.append("Asteroids no longer exist. We're all done!")
                    turn_direction = 0
                    idle_thrust = 0
                # We still simulate one iteration of this, because if we had a pending shot from before, this will do the shot!
                sim_complete_without_crash = self.update(idle_thrust, SHIP_MAX_TURN_RATE*turn_direction, False)
                #print("NOT SHOOTING AT ANYTHING, THE MOVE SEQUENCE IS BELOW AND HOPEFULLY WE DONT SHOOT ON THIS TIMESTEP")
                #print(self.ship_move_sequence)
                #print(f"Sim id {self.sim_id} is returning from target sim with success value {sim_complete_without_crash}")
                return sim_complete_without_crash

        #debug_print('Closest ang asteroid:')
        #debug_print(target_asteroids_list[least_angular_distance_asteroid_index])
        #debug_print('Second closest ang asteroid:')
        #debug_print(target_asteroids_list[second_least_angular_distance_asteroid_index])

        #print(f"Bullet should have been fired on simulated timestep {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)}")
        
        # The following code is confusing, because we're working with different times.
        # We need to make sure that when we talk about an asteroid, we talk about its position at a specific time. If the time we talk about is not synced up with the position, everything's wrong.
        # So if an asteroid is on position 1 at time 1, position 2 at time 2, position 3 at time 3, etc, then we must associate the asteroid with a timestep.
        # A smarter, less buggy way to store asteroids is to include not only their position, but their timestep as well. But too late.
        # Timestep [self.initial_timestep + self.future_timesteps] is the current timestep, or the timestep the sim was started on. Future timesteps should be 0 so far since we haven't moved.
        # Timestep [self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)] gives us the timestep after we do our aiming, and when we shoot our bullet
        # Timestep [self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid] gives us the timestep the asteroid got hit

        if actual_asteroid_hit:
            actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid, self.game_state, True)
        if actual_asteroid_hit is None or not check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len(aiming_move_sequence), self.game_state, actual_asteroid_hit_when_firing, True):
            self.fire_next_timestep_flag = False
            #print("We can't hit anything this timestep.")
            if asteroids_still_exist:
                #print('asteroids still exist')
                if random.randint(1, 10) == 1:
                    self.explanation_messages.append("Asteroids exist but we can't hit them. Moving around a bit randomly.")
                    self.asteroids_shot -= 1
                #turn_direction = random.random()
                #idle_thrust = random.triangular(0, SHIP_MAX_THRUST, 0)
                turn_direction = 0
                idle_thrust = 0
            else:
                #print('asteroids DO NOT exist')
                turn_direction = 0
                idle_thrust = 0
            sim_complete_without_crash = self.update(idle_thrust, SHIP_MAX_TURN_RATE*turn_direction, False)
            #print(f"Sim id {self.sim_id} is returning from target sim with success value {sim_complete_without_crash}")
            return sim_complete_without_crash
        else:
            #print(f"Asserting that we don't have a pending shot for asteroid {ast_to_string(actual_asteroid_hit)} on timestep {self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid}")
            if ENABLE_ASSERTIONS:
                assert check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid, self.game_state, actual_asteroid_hit, True)
            #print(f"Current timestep: {self.initial_timestep + self.future_timesteps}, and the aiming maneuver is {len(aiming_move_sequence)}")
            #print(self.asteroids_pending_death)
            #actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid, self.game_state, True)
            #print(f"Asserting that we don't have a pending shot for asteroid {ast_to_string(actual_asteroid_hit_when_firing)} on timestep {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)}")
            if ENABLE_ASSERTIONS:
                assert check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len(aiming_move_sequence), self.game_state, actual_asteroid_hit_when_firing, True)
            
            #actual_asteroid_hit_UNEXTRAPOLATED = dict(actual_asteroid_hit)
            # This back-extrapolates the asteroid to when we're firing our bullet
            
            debug_print(f"We used a sim to forecast the asteroid that we'll hit being the one at pos ({actual_asteroid_hit['position'][0]}, {actual_asteroid_hit['position'][1]}) with vel ({actual_asteroid_hit['velocity'][0]}, {actual_asteroid_hit['velocity'][1]}) in {timesteps_until_bullet_hit_asteroid - len(aiming_move_sequence)} timesteps")
            debug_print(f"The primitive forecast would have said we'd hit asteroid at pos ({target_asteroid['position'][0]}, {target_asteroid['position'][1]}) with vel ({target_asteroid['velocity'][0]}, {target_asteroid['velocity'][1]}) in {calculate_timesteps_until_bullet_hits_asteroid(target_asteroid_interception_time_s, target_asteroid['radius'])} timesteps")
            #print(self.asteroids_pending_death)
            #print(f"\nTracking that we just shot at the asteroid {ast_to_string(actual_asteroid_hit)}, our intended target was {target_asteroid}")
            #actual_asteroid_hit_UNEXTRAPOLATED = extrapolate_asteroid_forward(actual_asteroid_hit, -(len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid), self.game_state, True)
            actual_asteroid_hit_at_present_time = time_travel_asteroid(actual_asteroid_hit, -timesteps_until_bullet_hit_asteroid, self.game_state, True)
            actual_asteroid_hit_at_present_time_for_plotting = time_travel_asteroid(actual_asteroid_hit, -timesteps_until_bullet_hit_asteroid - 1, self.game_state, True)
            global gamestate_plotting
            if gamestate_plotting and next_target_plotting and (start_gamestate_plotting_at_second is None or start_gamestate_plotting_at_second/DELTA_TIME <= self.initial_timestep + self.future_timesteps):
                self.game_state_plotter.update_plot([], [], [], [], [actual_asteroid_hit_at_present_time_for_plotting], [], [], [], False, new_target_plot_pause_time_s, 'FEASIBLE TARGETS') #[dict(a['asteroid']) for a in sorted_targets]
            #actual_asteroid_hit_tracking_purposes_super_early = extrapolate_asteroid_forward(actual_asteroid_hit, )
            #print(f"Asserting that we don't have a pending shot for asteroid {ast_to_string(actual_asteroid_hit_at_present_time)} on timestep {self.initial_timestep + self.future_timesteps}")
            #assert check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + 0*len(aiming_move_sequence), self.game_state, actual_asteroid_hit_at_present_time, True)
            #print(f'\n\nTracking that we shot at the asteroid {ast_to_string(actual_asteroid_hit_at_present_time)} at the timestep after turning {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence) - 1}')
            #print(self.asteroids_pending_death)
            self.asteroids_pending_death = track_asteroid_we_shot_at(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len(aiming_move_sequence), self.game_state, timesteps_until_bullet_hit_asteroid - len(aiming_move_sequence), actual_asteroid_hit_when_firing)
            #print(self.asteroids_pending_death)
            # Forecasted splits get progressed while doing the move sequence which includes rotation, so we need to start the forecast before the rotation even starts
            self.forecasted_asteroid_splits.extend(forecast_asteroid_bullet_splits(actual_asteroid_hit_at_present_time, timesteps_until_bullet_hit_asteroid, bullet_heading_deg=ship_state_after_aiming['heading'], game_state=self.game_state, wrap=True))
            #forecast_backup = self.forecasted_asteroid_splits.copy()
            #print('extending  list with')
            #print(forecast_asteroid_bullet_splits(actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, self.ship_state['heading'], self.game_state, True))
            
            #if enable_assertions:
            #   assert(is_asteroid_in_list(self.game_state['asteroids'], actual_asteroid_hit))
            #debug_print(f"Length of asteroids list before removing: {len(self.game_state['asteroids'])}")
            #if math.isclose(actual_asteroid_hit['velocity'][1], -150, abs_tol=0.01):
            #    debug_print("WE GOTTTTTTEEYYYYYYMMMMMM")
            #    debug_print(self.game_state['asteroids'])
            # To make it so that checking for the next imminent collision no longer considers the asteroid we just shot at, we'll remove it from the list of asteroids as if the ship instantly shot it down
            #self.game_state['asteroids'] = [a for a in self.game_state['asteroids'] if not is_asteroid_in_list([actual_asteroid_hit_at_present_time, actual_asteroid_hit_at_present_time_for_plotting, actual_asteroid_hit, actual_asteroid_hit_when_firing], a)]
            #self.game_state['asteroids'] = [a for a in self.game_state['asteroids'] if not math.isclose(a['velocity'][1], -150, abs_tol=0.01)]
            #debug_print(f"Length of asteroids list after removing: {len(self.game_state['asteroids'])}")
            #debug_print(self.game_state['asteroids'])
            #if locked_in:
            debug_print('Were gonna fire the next timestep! Printing out the move sequence:', aiming_move_sequence)
            # TODO: Currently Neo can't fire on timestep 1! Fix this! Although it's only a minor issue.
            self.asteroids_shot += 1
            #self.ship_move_sequence.append(aiming_move_sequence) # Bit of a hack...
            #print('AIMING MOVE SEQUENCE')
            #print(aiming_move_sequence)
            sim_complete_without_crash = self.apply_move_sequence(aiming_move_sequence)
            #self.forecasted_asteroid_splits = forecast_backup
            self.fire_next_timestep_flag = True
            #print('THE ACTUAL MOVE SEQUENCE WE GET BACK FROM THE SIM:')
            #print(self.ship_move_sequence)
            #print(f"Sim id {self.sim_id} is returning from target sim with success value {sim_complete_without_crash}")
            return sim_complete_without_crash

    #@line_profiler.profile
    def bullet_target_sim(self, ship_state: dict=None, fire_first_timestep: bool=False, fire_after_timesteps: int=0, skip_half_of_first_cycle: bool=False, current_move_index: int=None, whole_move_sequence: list=None, timestep_limit: int=math.inf):
        # Assume we shoot on the next timestep, so we'll create a bullet and then track it and simulate it to see what it hits, if anything
        # This sim doesn't modify the state of the simulation class. Everything here is discarded after the sim is over, and this is just to see what my bullet hits, if anything.
        asteroids = [dict(a) for a in self.game_state['asteroids']] # TODO: CULL ASTEROIDS? But then we won't be considering other bullets so maybe this is a bad idea. Maybe cull for all bullets and mines combined? Remember to unwrap the asteroids!
        #if ship_state and math.isclose(ship_state['heading'], 76.54927196048133, abs_tol=0.000000001):
        #    print('messed up asties', asteroids)
        mines = [dict(m) for m in self.game_state['mines']]
        bullets = [dict(b) for b in self.game_state['bullets']]
        initial_ship_state = self.get_ship_state()
        if ship_state and ENABLE_ASSERTIONS:
            assert check_coordinate_bounds(self.game_state, ship_state['position'][0], ship_state['position'][1])
        if whole_move_sequence:
            bullet_sim_ship_state = self.get_ship_state()
        #debug_print(f"Beginning sim, and here's the midair bullets we have before the sim started:")
        #debug_print(bullets)
        bullet_created = False
        my_bullet = None
        ship_not_collided_with_asteroid = True
        if skip_half_of_first_cycle:
            timesteps_until_bullet_hit_asteroid = 0
        else:
            timesteps_until_bullet_hit_asteroid = 0
        # Keep iterating until our bullet flies off the edge of the screen, or it hits an asteroid
        global gamestate_plotting
        while True:
            # Simplified simulation loop
            timesteps_until_bullet_hit_asteroid += 1
            if timesteps_until_bullet_hit_asteroid > timestep_limit:
                return None, None, ship_not_collided_with_asteroid
            #print(f"BULLET SIM STATE DUMP FOR FUTURE TIMESTEPS {timesteps_until_bullet_hit_asteroid}")
            #print("BULLETS")
            #print(bullets + (bullet_created)*[my_bullet])
            #print("ASTEOIRDS")
            #print(asteroids)
            #if gamestate_plotting:
               #flattened_asteroids_pending_death = [ast for ast_list in self.asteroids_pending_death.values() for ast in ast_list]
            if gamestate_plotting and bullet_sim_plotting and (start_gamestate_plotting_at_second is None or start_gamestate_plotting_at_second/DELTA_TIME <= self.initial_timestep + timesteps_until_bullet_hit_asteroid):# and random.random() > 0.0:#or math.isclose(self.ship_state['heading'], 359.8992958775248, abs_tol=0.000000001):
                flattened_asteroids_pending_death = [ast for ast_list in self.asteroids_pending_death.values() for ast in ast_list]
                if whole_move_sequence:
                    ship_plot_state = bullet_sim_ship_state
                else:
                    ship_plot_state = None
                self.game_state_plotter.update_plot(asteroids, ship_plot_state, bullets, [my_bullet], [], flattened_asteroids_pending_death, [], mines, True, EPS, f'BULLET SIMULATION TIMESTEP {self.initial_timestep + timesteps_until_bullet_hit_asteroid}')
                #print('INSIDE PLOTTING MY SPECIAL BULLET')
            # Simulate bullets
            if skip_half_of_first_cycle:
                skip_half_of_first_cycle = False
            else:
                bullet_remove_idxs = []
                for b_ind, b in enumerate(bullets):
                    new_bullet_pos = (b['position'][0] + b['velocity'][0]*DELTA_TIME, b['position'][1] + b['velocity'][1]*DELTA_TIME)
                    if check_coordinate_bounds(self.game_state, new_bullet_pos[0], new_bullet_pos[1]):
                        b['position'] = new_bullet_pos
                    else:
                        bullet_remove_idxs.append(b_ind)
                if bullet_remove_idxs:
                    bullets = [bullet for idx, bullet in enumerate(bullets) if idx not in bullet_remove_idxs]
                if bullet_created:
                    my_new_bullet_pos = (my_bullet['position'][0] + my_bullet['velocity'][0]*DELTA_TIME, my_bullet['position'][1] + my_bullet['velocity'][1]*DELTA_TIME)
                    if check_coordinate_bounds(self.game_state, my_new_bullet_pos[0], my_new_bullet_pos[1]):
                        my_bullet['position'] = my_new_bullet_pos
                    else:
                        return None, None, ship_not_collided_with_asteroid # The bullet got shot into the void without hitting anything :(
                
                for m in mines:
                    m['remaining_time'] -= DELTA_TIME
                
                for a in asteroids:
                    # Use an inline approximate wrap instead of the completely accurate but slower Kessler wrap. This is good enough 99.99% of the time, except for contrived edge cases. But for random asteroids, the chance of this going wrong is basically 0.
                    #a['position'] = ((a['position'][0] + a['velocity'][0]*DELTA_TIME)%self.game_state['map_size'][0], (a['position'][1] + a['velocity'][1]*DELTA_TIME)%self.game_state['map_size'][1])
                    a['position'] = wrap_position((a['position'][0] + a['velocity'][0]*DELTA_TIME, a['position'][1] + a['velocity'][1]*DELTA_TIME), self.game_state['map_size'])
            
            #debug_print(f"TS ahead of sim end: {timesteps_until_bullet_hit_asteroid}")
            #debug_print(asteroids)
            # Create the initial bullet we fire, if we're locked in
            if fire_first_timestep and timesteps_until_bullet_hit_asteroid == 1:
                rad_heading = radians(initial_ship_state['heading'])
                cos_heading = cos(rad_heading)
                sin_heading = sin(rad_heading)
                bullet_x = initial_ship_state['position'][0] + SHIP_RADIUS*cos_heading
                bullet_y = initial_ship_state['position'][1] + SHIP_RADIUS*sin_heading
                # Make sure the bullet isn't being fired out into the void
                if check_coordinate_bounds(self.game_state, bullet_x, bullet_y):
                    vx = BULLET_SPEED*cos_heading
                    vy = BULLET_SPEED*sin_heading
                    initial_timestep_fire_bullet = {
                        'position': (bullet_x, bullet_y),
                        'velocity': (vx, vy),
                        'heading': initial_ship_state['heading'],
                        'mass': BULLET_MASS,
                        'tail_delta': (-BULLET_LENGTH*cos(radians(initial_ship_state['heading'])), -BULLET_LENGTH*sin(radians(initial_ship_state['heading']))),
                    }
                    bullets.append(initial_timestep_fire_bullet)
            # The new bullet we create will end up at the end of the list of bullets
            if not bullet_created and timesteps_until_bullet_hit_asteroid == fire_after_timesteps + 1:
                if ship_state is not None:
                    bullet_fired_from_ship_heading = ship_state['heading']
                    bullet_fired_from_ship_position = ship_state['position']
                else:
                    bullet_fired_from_ship_heading = self.ship_state['heading']
                    bullet_fired_from_ship_position = self.ship_state['position']
                # TODO: LRU CACHE THE SIN AND COS FUNCTIONS OR JUST MODIFY THE BULLET DICT TO INCLUDE VELS AND PROBABLY ALSO THE TAIL
                rad_heading = radians(bullet_fired_from_ship_heading)
                cos_heading = cos(rad_heading)
                sin_heading = sin(rad_heading)
                bullet_x = bullet_fired_from_ship_position[0] + SHIP_RADIUS*cos_heading
                bullet_y = bullet_fired_from_ship_position[1] + SHIP_RADIUS*sin_heading
                # Make sure the bullet isn't being fired out into the void
                if not check_coordinate_bounds(self.game_state, bullet_x, bullet_y):
                    return None, None, ship_not_collided_with_asteroid # The bullet got shot into the void without hitting anything :(
                vx = BULLET_SPEED*cos_heading
                vy = BULLET_SPEED*sin_heading
                my_bullet = {
                    'position': (bullet_x, bullet_y),
                    'velocity': (vx, vy),
                    'heading': bullet_fired_from_ship_heading,
                    'mass': BULLET_MASS,
                    'tail_delta': (-BULLET_LENGTH*cos_heading, -BULLET_LENGTH*sin_heading),
                }
                #if math.isclose(my_bullet['heading'], 76.54927196048133, abs_tol=0.000000001):
                #    print(f'my bullet created after {timesteps_until_bullet_hit_asteroid} timesteps:', my_bullet)
                bullet_created = True
                #print(f"We created our own simulated bullet, we're on timestep actually idk and idc, and the bullet is:")
                #print(my_bullet)
                #print(f"Also on this timestep, the asteroids are")
                #print(asteroids)
            #debug_print(f"My sim bullet is at on timestep {timesteps_until_bullet_hit_asteroid}:")
            #debug_print(my_bullet)
            
            if whole_move_sequence and current_move_index + timesteps_until_bullet_hit_asteroid - 1 < len(whole_move_sequence):
                # Simulate ship dynamics, if we have the full future list of moves to go off of
                thrust = whole_move_sequence[current_move_index + timesteps_until_bullet_hit_asteroid - 1]['thrust']
                turn_rate = whole_move_sequence[current_move_index + timesteps_until_bullet_hit_asteroid - 1]['turn_rate']
                #if math.isclose(self.ship_state['heading'], 79.12893456534394, abs_tol=0.000000001):
                #    print(f"We're on ship bullet move ind {timesteps_until_bullet_hit_asteroid - 1} and we're doing thrust: {thrust}, turn_rate: {turn_rate}, and the bullet sim ship state is:")
                #    print(bullet_sim_ship_state)
                drag_amount = SHIP_DRAG*DELTA_TIME
                if drag_amount > abs(bullet_sim_ship_state['speed']):
                    bullet_sim_ship_state['speed'] = 0
                else:
                    bullet_sim_ship_state['speed'] -= drag_amount*np.sign(bullet_sim_ship_state['speed'])
                if ENABLE_ASSERTIONS:
                    assert -SHIP_MAX_THRUST <= thrust <= SHIP_MAX_THRUST
                #thrust = min(max(-SHIP_MAX_THRUST, thrust), SHIP_MAX_THRUST)
                # Apply thrust
                bullet_sim_ship_state['speed'] += thrust*DELTA_TIME
                if ENABLE_ASSERTIONS:
                    assert -SHIP_MAX_SPEED <= bullet_sim_ship_state['speed'] <= SHIP_MAX_SPEED
                #bullet_sim_ship_state['speed'] = min(max(-SHIP_MAX_SPEED, bullet_sim_ship_state['speed']), SHIP_MAX_SPEED)
                if ENABLE_ASSERTIONS:
                    assert -SHIP_MAX_TURN_RATE <= turn_rate <= SHIP_MAX_TURN_RATE
                #turn_rate = min(max(-SHIP_MAX_TURN_RATE, turn_rate), SHIP_MAX_TURN_RATE)
                # Update the angle based on turning rate
                bullet_sim_ship_state['heading'] += turn_rate*DELTA_TIME
                # Keep the angle within (0, 360)
                bullet_sim_ship_state['heading'] %= 360
                # Use speed magnitude to get velocity vector
                bullet_sim_ship_state['velocity'] = (cos(radians(bullet_sim_ship_state['heading']))*bullet_sim_ship_state['speed'], sin(radians(bullet_sim_ship_state['heading']))*bullet_sim_ship_state['speed'])
                # Update the position based off the velocities
                # Do the wrap in the same operation
                bullet_sim_ship_state['position'] = wrap_position((bullet_sim_ship_state['position'][0] + bullet_sim_ship_state['velocity'][0]*DELTA_TIME, bullet_sim_ship_state['position'][1] + bullet_sim_ship_state['velocity'][1]*DELTA_TIME), self.game_state['map_size'])

            # Check bullet/asteroid collisions
            bullet_remove_idxs = []
            asteroid_remove_idxs = []
            for b_idx, b in enumerate(bullets + (bullet_created)*[my_bullet]):
                b_tail = (b['position'][0] + b['tail_delta'][0], b['position'][1] + b['tail_delta'][1])
                for a_idx, a in enumerate(asteroids):
                    # If collision occurs
                    if asteroid_bullet_collision(b['position'], b_tail, a['position'], a['radius']):
                        if b_idx == len(bullets):
                            # This bullet is my bullet!
                            #print('MY BULLET HIT THE ASTEROID!')
                            #if math.isclose(my_bullet['heading'], 76.54927196048133, abs_tol=0.000000001):
                            #    print(f'MY BULLET HIT THE ASTEROID! After {timesteps_until_bullet_hit_asteroid} timesteps')
                            return a, timesteps_until_bullet_hit_asteroid, ship_not_collided_with_asteroid
                        else:
                            # Mark bullet for removal
                            bullet_remove_idxs.append(b_idx)
                            # Create asteroid splits and mark it for removal
                            new_asteroids_from_collision = forecast_asteroid_bullet_splits(a, 0, bullet_velocity=b['velocity'])
                            asteroids.extend(new_asteroids_from_collision)
                            asteroid_remove_idxs.append(a_idx)
                            # Stop checking this bullet
                            break
            # Remove bullets and asteroids
            if bullet_remove_idxs:
                bullets = [bullet for idx, bullet in enumerate(bullets) if idx not in bullet_remove_idxs]
            if asteroid_remove_idxs:
                asteroids = [asteroid for idx, asteroid in enumerate(asteroids) if idx not in asteroid_remove_idxs]

            # Check mine/asteroid collisions
            mine_remove_idxs = []
            asteroid_remove_idxs = set() # Use a set, since this is the only case where we may have many asteroids removed at once
            new_asteroids = []
            for m_idx, mine in enumerate(mines):
                if mine['remaining_time'] < EPS:
                    # Mine is detonating
                    mine_remove_idxs.append(m_idx)
                    for a_idx, asteroid in enumerate(asteroids):
                        if check_collision(asteroid['position'][0], asteroid['position'][1], asteroid['radius'], mine['position'][0], mine['position'][1], MINE_BLAST_RADIUS):
                            new_asteroids.extend(forecast_asteroid_mine_splits(a, 0, mine, self.game_state, True))
                            asteroid_remove_idxs.add(a_idx)
            if mine_remove_idxs:
                mines = [mine for idx, mine in enumerate(mines) if idx not in mine_remove_idxs]
            if asteroid_remove_idxs:
                asteroids = [asteroid for idx, asteroid in enumerate(asteroids) if idx not in asteroid_remove_idxs]
            asteroids.extend(new_asteroids)

            # Check ship/asteroid collisions
            if ship_not_collided_with_asteroid:
                if whole_move_sequence is not None:
                    ship_position = bullet_sim_ship_state['position']
                elif ship_state is not None:
                    ship_position = ship_state['position']
                else:
                    ship_position = self.ship_state['position']
                asteroid_remove_idxs = []
                for a_idx, asteroid in enumerate(asteroids):
                    if check_collision(ship_position[0], ship_position[1], SHIP_RADIUS, asteroid['position'][0], asteroid['position'][1], asteroid['radius']):
                        # TODO: ADD IN SHIP VELOCITY INSTEAD OF ASSUMING 0
                        new_asteroids_from_collision = forecast_asteroid_ship_splits(asteroid, 0, (0, 0), self.game_state, True)
                        #if ship_state and math.isclose(ship_state['heading'], 76.54927196048133, abs_tol=0.000000001):
                        #    print(f"We just extended the asties list with the follwing from a collision between ship and {ast_to_string(a)}:", new_asteroids_from_collision)
                        #if math.isclose(self.ship_state['heading'], 79.12893456534394, abs_tol=0.000000001):
                            #print(f"THE SHIP COLLIDED WITH THE ASTEROID {ast_to_string(asteroid)}! Ship position is at: {ship_position}, which should be the same as {bullet_sim_ship_state['position']}")
                            #print(asteroids)
                        asteroids.extend(new_asteroids_from_collision)
                        asteroid_remove_idxs.append(a_idx)
                        # Stop checking this ship's collisions. And also return saying the ship took damage!
                        ship_not_collided_with_asteroid = False
                        break
                # Cull asteroids marked for removal
                if asteroid_remove_idxs:
                    asteroids = [asteroid for idx, asteroid in enumerate(asteroids) if idx not in asteroid_remove_idxs]

    def apply_move_sequence(self, move_sequence: list=None):
        if move_sequence is None:
            move_sequence = []
        sim_was_safe = True
        for move in move_sequence:
            thrust = 0
            turn_rate = 0
            fire = None
            if 'thrust' in move:
                thrust = move['thrust']
            if 'turn_rate' in move:
                turn_rate = move['turn_rate']
            if 'fire' in move:
                fire = move['fire']
            if 'drop_mine' in move:
                drop_mine = move['drop_mine'] # TODO: Implement
            if not self.update(thrust, turn_rate, fire):
                sim_was_safe = False
        return sim_was_safe
    
    def simulate_maneuver(self, move_sequence: list, allow_firing: bool):
        for move in move_sequence:
            thrust = 0
            turn_rate = 0
            if 'thrust' in move:
                thrust = move['thrust']
            if 'turn_rate' in move:
                turn_rate = move['turn_rate']
            if not self.update(thrust, turn_rate, None if allow_firing else False, move_sequence):
                return False
        return True

    def update(self, thrust=0, turn_rate=0, fire=None, whole_move_sequence: list=None) -> bool:
        # This is a highly optimized simulation of what kessler_game.py does, and should match exactly its behavior
        # Being even one timestep off is the difference between life and death!!!
        return_value = None
        if not PRUNE_SIM_STATE_SEQUENCE or self.future_timesteps == 0:
            self.state_sequence.append({'timestep': self.initial_timestep + self.future_timesteps, 'ship_state': dict(self.ship_state), 'game_state': self.get_game_state(), 'asteroids_pending_death': dict(self.asteroids_pending_death), 'forecasted_asteroid_splits': [dict(a) for a in self.forecasted_asteroid_splits]})
        else:
            self.state_sequence.append({'timestep': self.initial_timestep + self.future_timesteps})
        #print(f"Current SIM STATE ON TIMESTEP {self.initial_timestep + self.future_timesteps}, sim id {self.sim_id}")
        #print(f"Ship is respawning: {self.ship_state['is_respawning']}, respawn timer: {self.respawn_timer}")
        #print(self.state_sequence[-1])
        # The simulation starts by evaluating actions and dynamics of the current present timestep, and then steps into the future
        # The game state we're given is actually what we had at the end of the previous timestep
        # The game will take the previous state, and apply current actions and then update to get the result of this timestep

        # Simulation order:
        # Ships are given the game state from after the previous timestep. Ships then decide the inputs.
        # Update bullets/mines/asteroids.
        # Ship has inputs applied and updated. Any new bullets and mines that the ship creates is added to the list.
        # Bullets past the map edge get culled
        # Ships and asteroids beyond the map edge get wrapped
        # Check for bullet/asteroid collisions. Checked for each bullet in list order, check for each asteroid in list order. Bullets and asteroids are removed here, and any new asteroids created are added to the list now.
        # Check mine collisions with asteroids/ships. For each mine in list order, check whether it is detonating and if it collides with first asteroids in list order (and add new asteroids to list), and then ships.
        # Check for asteroid/ship collisions. For each ship in list order, check collisions with each asteroid in list order. New asteroids are added now. Ships and asteroids get culled if they collide.
        # Check ship/ship collisions and cull them.

        # Simulate dynamics of bullets
        # Kessler will move bullets and cull them in different steps, but we combine them in one operation here
        # So we need to detect when the bullets are crossing the boundary, and delete them if they try to
        # Enumerate and track indices to delete
        bullet_remove_idxs = []
        for b_ind, b in enumerate(self.game_state['bullets']):
            new_bullet_pos = (b['position'][0] + b['velocity'][0]*DELTA_TIME, b['position'][1] + b['velocity'][1]*DELTA_TIME)
            if check_coordinate_bounds(self.game_state, new_bullet_pos[0], new_bullet_pos[1]):
                b['position'] = new_bullet_pos
            else:
                bullet_remove_idxs.append(b_ind)
        if bullet_remove_idxs:
            self.game_state['bullets'] = [bullet for idx, bullet in enumerate(self.game_state['bullets']) if idx not in bullet_remove_idxs]

        # Update mines
        #if len(self.game_state['mines']) > 0:
            #print('BEFORE UPDATING MINES', self.game_state['mines'])
        for m in self.game_state['mines']:
            if ENABLE_ASSERTIONS:
                assert m['remaining_time'] > EPS - DELTA_TIME
            if not m['remaining_time'] > EPS - DELTA_TIME:
                print(f"WARNING, mine time remaining is negative, it's {m['remaining_time']} s")
            m['remaining_time'] -= DELTA_TIME
            # If the timer is below eps, it'll detonate this timestep
            #print('AFTER UPDATING MINES', self.game_state['mines'])

        # Simulate dynamics of asteroids
        # Wrap the asteroid positions in the same operation
        #if self.future_timesteps >= self.timesteps_to_not_check_collision_for:
        for a in self.game_state['asteroids']:
            a['position'] = wrap_position((a['position'][0] + a['velocity'][0]*DELTA_TIME, a['position'][1] + a['velocity'][1]*DELTA_TIME), self.game_state['map_size'])

        # Simulate the ship!
        # Bullet firing happens before we turn the ship
        # Check whether we want to shoot a simulated bullet
        #print(f"SIMULATION TS {self.future_timesteps}")
        if self.fire_first_timestep and self.future_timesteps == 0:
            #print('FIRE FIRST TIMETSEP IS TRUE SO WERE FIRING')
            fire_this_timestep = True
        elif fire is None:
            timesteps_until_can_fire = max(0, 5 - (self.initial_timestep + self.future_timesteps - self.last_timestep_fired))
            fire_this_timestep = False
            if timesteps_until_can_fire == 0 and self.ship_state['bullets_remaining'] != 0 and not self.halt_shooting: #self.future_timesteps >= self.timesteps_to_not_check_collision_for:
                for asteroid in self.game_state['asteroids']:
                    if fire_this_timestep:
                        break
                    if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, asteroid, True):
                        unwrapped_asteroids = unwrap_asteroid(asteroid, self.game_state['map_size'][0], self.game_state['map_size'][1], UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON)
                        for a in unwrapped_asteroids:
                            if fire_this_timestep:
                                break
                            sols_list = calculate_interception(self.ship_state['position'][0], self.ship_state['position'][1], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'], self.ship_state['heading'], self.game_state)
                            for sol in sols_list:
                                if fire_this_timestep:
                                    break
                                feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist_during_interception = sol
                                if feasible and abs(shot_heading_error_rad) <= shot_heading_tolerance_rad:
                                    # Use the bullet sim to confirm that this will hit
                                    bullet_sim_timestep_limit = ceil(interception_time/DELTA_TIME) + 2
                                    actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, ship_was_safe = self.bullet_target_sim(None, False, 0, True, self.future_timesteps, whole_move_sequence, bullet_sim_timestep_limit)
                                    # TODO: Check to make sure this following statement is fine
                                    if actual_asteroid_hit is not None and ship_was_safe:
                                        actual_asteroid_hit_at_fire_time = time_travel_asteroid(actual_asteroid_hit, -timesteps_until_bullet_hit_asteroid, self.game_state, True)
                                        if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, actual_asteroid_hit_at_fire_time, True):
                                            fire_this_timestep = True
                                            self.asteroids_shot += 1
                                            #print(f"Tracking that we shot at the asteroid {ast_to_string(asteroid)}")
                                            self.forecasted_asteroid_splits.extend(forecast_asteroid_bullet_splits(actual_asteroid_hit_at_fire_time, timesteps_until_bullet_hit_asteroid, bullet_heading_deg=self.ship_state['heading'], game_state=self.game_state, wrap=True))
                                            #print(self.asteroids_pending_death)
                                            self.asteroids_pending_death = track_asteroid_we_shot_at(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, timesteps_until_bullet_hit_asteroid, actual_asteroid_hit_at_fire_time)
                                            break
        else:
            fire_this_timestep = fire
        
        if fire_this_timestep:
            self.last_timestep_fired = self.initial_timestep + self.future_timesteps
            # Remove respawn cooldown if we were in it
            self.ship_state['is_respawning'] = False
            self.respawn_timer = 0
            # Create new bullets/mines
            if self.ship_state['bullets_remaining'] != -1:
                self.ship_state['bullets_remaining'] -= 1
            #rad_heading = radians(self.ship_state['heading'])
            rad_heading = pi*self.ship_state['heading']/180
            cos_heading = cos(rad_heading)
            sin_heading = sin(rad_heading)
            #bullet_x = self.ship_state['position'][0] + SHIP_RADIUS*cos_heading#SHIP_RADIUS*cos(radians(self.ship_state['heading'])) # SHIP_RADIUS*cos_heading
            #bullet_y = self.ship_state['position'][1] + SHIP_RADIUS*sin_heading#SHIP_RADIUS*sin(radians(self.ship_state['heading'])) # SHIP_RADIUS*sin_heading
            bullet_x = self.ship_state['position'][0] + SHIP_RADIUS*cos(radians(self.ship_state['heading'])) # Exact
            bullet_y = self.ship_state['position'][1] + SHIP_RADIUS*sin(radians(self.ship_state['heading']))
            # Make sure the bullet isn't being fired out into the void
            if check_coordinate_bounds(self.game_state, bullet_x, bullet_y):
                vx = BULLET_SPEED*cos_heading
                vy = BULLET_SPEED*sin_heading
                new_bullet = {
                    'position': (bullet_x, bullet_y),
                    'velocity': (vx, vy),
                    'heading': self.ship_state['heading'],
                    'mass': BULLET_MASS,
                    'tail_delta': (-BULLET_LENGTH*cos_heading, -BULLET_LENGTH*sin_heading),
                }
                self.game_state['bullets'].append(new_bullet)

        # Only drop mines at the beginning of the sim
        # TODO: check the last timestep mined, and the -31 for off by one error. I think mines can be dropped every 31 frames but I'm not sure!
        if self.future_timesteps == 0 and self.last_timestep_mined <= self.initial_timestep + self.future_timesteps - 31 and not self.halt_shooting:
            drop_mine_this_timestep = check_mine_opportunity(self.ship_state, self.game_state) # Read only access of the ship and game states
        else:
            drop_mine_this_timestep = False
        if drop_mine_this_timestep:
            self.last_timestep_mined = self.initial_timestep + self.future_timesteps
            # This doesn't check whether it's valid to place a mine! It just does it!
            self.explanation_messages.append("This is a good chance to drop a mine. Bombs away!")
            # Remove respawn cooldown if we were in it
            self.ship_state['is_respawning'] = False
            self.respawn_timer = 0
            debug_print(f'BOMBS AWAY! Sim ID {self.sim_id}, future timesteps {self.future_timesteps}')
            new_mine = {
                'position': self.ship_state['position'],
                'mass': MINE_MASS,
                'fuse_time': 3,
                'remaining_time': 3,
            }
            self.game_state['mines'].append(new_mine)
            self.ship_state['mines_remaining'] -= 1
            if ENABLE_ASSERTIONS:
                assert self.ship_state['mines_remaining'] >= 0
        
        # Update respawn timer
        if self.respawn_timer <= 0:
            self.respawn_timer = 0
        else:
            self.respawn_timer -= DELTA_TIME
        if not self.respawn_timer:
            self.ship_state['is_respawning'] = False
            self.respawn_timer = 0
        # Simulate ship dynamics
        drag_amount = SHIP_DRAG*DELTA_TIME
        if drag_amount > abs(self.ship_state['speed']):
            self.ship_state['speed'] = 0
        else:
            self.ship_state['speed'] -= drag_amount*np.sign(self.ship_state['speed'])
        # Bounds check the thrust
        # TODO: REMOVE BOUNDS CHECKS
        if ENABLE_ASSERTIONS:
            assert -SHIP_MAX_THRUST <= thrust <= SHIP_MAX_THRUST
        #thrust = min(max(-SHIP_MAX_THRUST, thrust), SHIP_MAX_THRUST)
        # Apply thrust
        self.ship_state['speed'] += thrust*DELTA_TIME
        if self.ship_state['speed'] > SHIP_MAX_SPEED:
            self.ship_state['speed'] = SHIP_MAX_SPEED
        elif self.ship_state['speed'] < -SHIP_MAX_SPEED:
            self.ship_state['speed'] = -SHIP_MAX_SPEED
        if ENABLE_ASSERTIONS:
            if not (-SHIP_MAX_SPEED <= self.ship_state['speed'] <= SHIP_MAX_SPEED):
                print(self.ship_state['speed'])
            assert -SHIP_MAX_SPEED <= self.ship_state['speed'] <= SHIP_MAX_SPEED
        #self.ship_state['speed'] = min(max(-SHIP_MAX_SPEED, self.ship_state['speed']), SHIP_MAX_SPEED)
        if ENABLE_ASSERTIONS:
            if not (-SHIP_MAX_TURN_RATE <= turn_rate <= SHIP_MAX_TURN_RATE):
                print(turn_rate)
            assert -SHIP_MAX_TURN_RATE <= turn_rate <= SHIP_MAX_TURN_RATE
        #turn_rate = min(max(-SHIP_MAX_TURN_RATE, turn_rate), SHIP_MAX_TURN_RATE)
        # Update the angle based on turning rate
        self.ship_state['heading'] += turn_rate*DELTA_TIME
        # Keep the angle within (0, 360)
        if self.ship_state['heading'] > 360:
            self.ship_state['heading'] -= 360.0
        elif self.ship_state['heading'] < 0:
            self.ship_state['heading'] += 360.0
        # Use speed magnitude to get velocity vector
        self.ship_state['velocity'] = (cos(radians(self.ship_state['heading']))*self.ship_state['speed'], sin(radians(self.ship_state['heading']))*self.ship_state['speed'])
        # Update the position based off the velocities
        # Do the wrap in the same operation
        self.ship_state['position'] = wrap_position((self.ship_state['position'][0] + self.ship_state['velocity'][0]*DELTA_TIME, self.ship_state['position'][1] + self.ship_state['velocity'][1]*DELTA_TIME), self.game_state['map_size'])
        
        # Check bullet/asteroid collisions
        bullet_remove_idxs = []
        asteroid_remove_idxs = []
        for b_idx, b in enumerate(self.game_state['bullets']):
            b_tail = (b['position'][0] + b['tail_delta'][0], b['position'][1] + b['tail_delta'][1])
            for a_idx, a in enumerate(self.game_state['asteroids']):
                # If collision occurs
                #print(f"BULLET COLLIDED WITH AST:", b, ast_to_string(a))
                if asteroid_bullet_collision(b['position'], b_tail, a['position'], a['radius']):
                    # Mark bullet for removal
                    bullet_remove_idxs.append(b_idx)
                    # Create asteroid splits and mark it for removal
                    self.game_state['asteroids'].extend(forecast_asteroid_bullet_splits(a, 0, bullet_velocity=b['velocity']))
                    asteroid_remove_idxs.append(a_idx)
                    # Stop checking this bullet
                    break
        # Cull bullets and asteroids that are marked for removal
        if bullet_remove_idxs:
            self.game_state['bullets'] = [bullet for idx, bullet in enumerate(self.game_state['bullets']) if idx not in bullet_remove_idxs]
        if asteroid_remove_idxs:
            self.game_state['asteroids'] = [asteroid for idx, asteroid in enumerate(self.game_state['asteroids']) if idx not in asteroid_remove_idxs]

        self.ship_move_sequence.append({'timestep': self.initial_timestep + self.future_timesteps, 'thrust': thrust, 'turn_rate': turn_rate, 'fire': fire_this_timestep, 'drop_mine': drop_mine_this_timestep})

        # Check mine/asteroid and mine/ship collisions
        mine_remove_idxs = []
        asteroid_remove_idxs = set() # Use a set, since this is the only case where we may have many asteroids removed at once
        new_asteroids = []
        #if self.game_state['mines']:
        #    debug_print('mines:', self.game_state['mines'])
        for idx_mine, mine in enumerate(self.game_state['mines']):
            if return_value == False:
                break
            if mine['remaining_time'] < EPS:
                # Mine is detonating
                debug_print(f"\nSIM ID {self.sim_id}, MINE NUMBER {idx_mine} IS DETONATING IN THE SIM ON TIMESTEP: {self.initial_timestep + self.future_timesteps}, the ship is at {self.ship_state['position']}")
                mine_remove_idxs.append(idx_mine)
                for a_idx, asteroid in enumerate(self.game_state['asteroids']):
                    if check_collision(asteroid['position'][0], asteroid['position'][1], asteroid['radius'], mine['position'][0], mine['position'][1], MINE_BLAST_RADIUS):
                        #print(asteroid)
                        new_asteroids.extend(forecast_asteroid_mine_splits(asteroid, 0, mine, self.game_state, True))
                        asteroid_remove_idxs.add(a_idx)
                if check_collision(self.ship_state['position'][0], self.ship_state['position'][1], SHIP_RADIUS, mine['position'][0], mine['position'][1], MINE_BLAST_RADIUS):
                    # Ship got hit by mine, RIP
                    #print(f"Ship got hit by mine in sim ID {self.sim_id}, RIP")
                    # Even if the player is invincible, we still need to check for these collisions since invincibility is only for asteroids and ship-ship collisions
                    return_value = False
                    self.ship_state['lives_remaining'] -= 1
                    self.ship_state['is_respawning'] = True
                    self.ship_state['speed'] = 0
                    self.respawn_timer = 3
                ## As a bandaid, we'll just clear the list of predicted asteroids since the mine blowing up will completely affect this!
                #self.forecasted_asteroid_splits.clear()
                #assert not self.forecasted_asteroid_splits
        if mine_remove_idxs:
            self.game_state['mines'] = [mine for idx, mine in enumerate(self.game_state['mines']) if idx not in mine_remove_idxs]
        if asteroid_remove_idxs:
            self.game_state['asteroids'] = [asteroid for idx, asteroid in enumerate(self.game_state['asteroids']) if idx not in asteroid_remove_idxs]
        self.game_state['asteroids'].extend(new_asteroids)

        # TODO: Might be edge case where ship collides with multiple things at the same frame. Make sure to check this! I don't handle this currently, but Kessler might do it differently, and cause desyncs.

        # Check ship/asteroid collisions
        if return_value is None and not self.ship_state['is_respawning']:
            asteroid_remove_idxs = []
            for a_idx, asteroid in enumerate(self.game_state['asteroids']):
                if check_collision(self.ship_state['position'][0], self.ship_state['position'][1], SHIP_RADIUS, asteroid['position'][0], asteroid['position'][1], asteroid['radius']):
                    #if not (self.ship_state['velocity'][0] == 0 and self.ship_state['velocity'][1] == 0):
                    #    print(self.ship_state['velocity'])
                    #assert self.ship_state['velocity'][0] == 0 and self.ship_state['velocity'][1] == 0
                    new_asteroids_from_collision = forecast_asteroid_ship_splits(asteroid, 0, self.ship_state['velocity'], self.game_state, True)
                    self.game_state['asteroids'].extend(new_asteroids_from_collision)
                    asteroid_remove_idxs.append(a_idx)
                    # Stop checking this ship's collisions. And also return saying the ship took damage!
                    return_value = False
                    self.ship_state['lives_remaining'] -= 1
                    self.ship_state['is_respawning'] = True
                    self.ship_state['speed'] = 0
                    self.respawn_timer = 3
                    break
            # Cull asteroids marked for removal
            if asteroid_remove_idxs:
                self.game_state['asteroids'] = [asteroid for idx, asteroid in enumerate(self.game_state['asteroids']) if idx not in asteroid_remove_idxs]

        # Check ship/ship collisions
        if return_value is None and not self.ship_state['is_respawning']:
            if self.get_instantaneous_ship_collision():
                return_value = False
                self.ship_state['lives_remaining'] -= 1
                self.ship_state['is_respawning'] = True
                self.ship_state['speed'] = 0
                self.respawn_timer = 3

        self.forecasted_asteroid_splits = maintain_forecasted_asteroids(self.forecasted_asteroid_splits, self.game_state, True)

        self.future_timesteps += 1
        if return_value is None:
            #print(f"Update returning TRUE")
            return True
        else:
            #print(f"Update returning {return_value} for sim id {self.sim_id}")
            return return_value

    def rotate_heading(self, heading_difference_deg, shoot_on_first_timestep=False):
        target_heading = (self.ship_state['heading'] + heading_difference_deg)%360
        still_need_to_turn = heading_difference_deg
        while abs(still_need_to_turn) > SHIP_MAX_TURN_RATE*DELTA_TIME + EPS:
            assert -SHIP_MAX_TURN_RATE <= SHIP_MAX_TURN_RATE*np.sign(heading_difference_deg) <= SHIP_MAX_TURN_RATE
            if not self.update(0, SHIP_MAX_TURN_RATE*np.sign(heading_difference_deg), shoot_on_first_timestep):
                return False
            shoot_on_first_timestep = False
            still_need_to_turn -= SHIP_MAX_TURN_RATE*np.sign(heading_difference_deg)*DELTA_TIME
        assert -SHIP_MAX_TURN_RATE <= still_need_to_turn/DELTA_TIME <= SHIP_MAX_TURN_RATE
        if not self.update(0, still_need_to_turn/DELTA_TIME, shoot_on_first_timestep):
            return False
        if ENABLE_ASSERTIONS:
            assert abs(target_heading - self.ship_state['heading']) <= 2*EPS
        return True

    def get_rotate_heading_move_sequence(self, heading_difference_deg, shoot_on_first_timestep=False):
        move_sequence = []
        if abs(heading_difference_deg)*10 < GRAIN:
            move_sequence.append({'thrust': 0, 'turn_rate': 0, 'fire': shoot_on_first_timestep})
            return move_sequence
        still_need_to_turn = heading_difference_deg
        while abs(still_need_to_turn) > SHIP_MAX_TURN_RATE*DELTA_TIME:
            assert -SHIP_MAX_TURN_RATE <= SHIP_MAX_TURN_RATE*np.sign(heading_difference_deg) <= SHIP_MAX_TURN_RATE
            move_sequence.append({'thrust': 0, 'turn_rate': SHIP_MAX_TURN_RATE*np.sign(heading_difference_deg), 'fire': shoot_on_first_timestep})
            shoot_on_first_timestep = False
            still_need_to_turn -= SHIP_MAX_TURN_RATE*np.sign(heading_difference_deg)*DELTA_TIME
        if abs(still_need_to_turn) > EPS:
            assert -SHIP_MAX_TURN_RATE <= still_need_to_turn/DELTA_TIME <= SHIP_MAX_TURN_RATE
            move_sequence.append({'thrust': 0, 'turn_rate': still_need_to_turn/DELTA_TIME, 'fire': shoot_on_first_timestep})
        return move_sequence

    def accelerate(self, target_speed, turn_rate=0):
        # Keep in mind speed can be negative
        # Drag will always slow down the ship
        while abs(self.ship_state['speed'] - target_speed) > EPS:
            drag = -SHIP_DRAG*np.sign(self.ship_state['speed'])
            drag_amount = SHIP_DRAG*DELTA_TIME
            if drag_amount > abs(self.ship_state['speed']):
                # The drag amount is reduced if it would make the ship cross 0 speed on its own
                adjust_drag_by = abs((drag_amount - abs(self.ship_state['speed']))/DELTA_TIME)
                drag -= adjust_drag_by*np.sign(drag)
            delta_speed_to_target = target_speed - self.ship_state['speed']
            thrust_amount = delta_speed_to_target/DELTA_TIME - drag
            #print(thrust_amount, self.ship_state['speed'], target_speed, delta_speed_to_target)
            thrust_amount = min(max(-SHIP_MAX_THRUST, thrust_amount), SHIP_MAX_THRUST)
            if not self.update(thrust_amount, turn_rate):
                return False
        return True

    def cruise(self, cruise_time, cruise_turn_rate=0):
        # Maintain current speed
        for _ in range(cruise_time):
            if not self.update(np.sign(self.ship_state['speed'])*SHIP_DRAG, cruise_turn_rate):
                return False
        return True

    def get_move_sequence(self):
        return self.ship_move_sequence

    def get_state_sequence(self):
        if self.state_sequence and self.state_sequence[-1]['timestep'] != self.initial_timestep + self.future_timesteps:
            #self.state_sequence.append({'timestep': self.initial_timestep + self.future_timesteps, 'position': self.ship_state['position'], 'velocity': self.ship_state['velocity'], 'speed': self.ship_state['speed'], 'heading': self.ship_state['heading'], 'ship_state': self.get_ship_state(), 'asteroids': [dict(a) for a in self.game_state['asteroids']], 'bullets': [dict(b) for b in self.game_state['bullets']], 'asteroids_pending_death': dict(self.asteroids_pending_death), 'forecasted_asteroid_splits': [dict(a) for a in self.forecasted_asteroid_splits]})
            self.state_sequence.append({'timestep': self.initial_timestep + self.future_timesteps, 'ship_state': dict(self.ship_state), 'game_state': self.get_game_state(), 'asteroids_pending_death': dict(self.asteroids_pending_death), 'forecasted_asteroid_splits': [dict(a) for a in self.forecasted_asteroid_splits]})
        return self.state_sequence

    def get_sequence_length(self):
        #debug_print(f"Length of move seq: {len(self.ship_move_sequence)}, length of state seq: {len(self.state_sequence)}")
        if ENABLE_ASSERTIONS:
            if not (len(self.ship_move_sequence) + 1 == len(self.state_sequence) or len(self.ship_move_sequence) == len(self.state_sequence)):
                print(f"len(self.ship_move_sequence): {len(self.ship_move_sequence)}, len(self.state_sequence): {len(self.state_sequence)}")
            assert len(self.ship_move_sequence) + 1 == len(self.state_sequence) or len(self.ship_move_sequence) == len(self.state_sequence)
        return len(self.state_sequence)

    def get_future_timesteps(self):
        return self.future_timesteps

    def get_position(self):
        return self.ship_state['position']

    def get_last_timestep_fired(self):
        return self.last_timestep_fired

    def get_last_timestep_mined(self):
        return self.last_timestep_mined

    def get_velocity(self):
        return self.ship_state['velocity']

    def get_heading(self):
        return self.ship_state['heading']

class Neo(KesslerController):
    @property
    def name(self) -> str:
        return "Neo"

    def __init__(self):
        debug_print('INITIALIZING NEO')
        self.init_done = False
        self.ship_id = None
        self.current_timestep = -1
        #self.last_timestep_fired = -math.inf
        self.action_queue = []  # This will become our heap
        self.asteroids_pending_death = {} # Keys are timesteps, and the values are the asteroids that still have midair bullets travelling toward them, so we don't want to shoot at them again
        self.forecasted_asteroid_splits = [] # List of asteroids that are forecasted to appear
        self.previous_asteroids_list = []
        self.last_respawn_maneuver_timestep_range = (-math.inf, 0)
        self.last_respawn_invincible_timestep_range = (-math.inf, 0)
        #self.fire_next_timestep_flag = False
        self.previous_bullets = []
        self.game_state_plotter = None
        self.actioned_timesteps = set()
        self.sims_this_planning_period = [] # The first sim in the list is stationary targetting, and the rest is maneuvers
        self.best_fitness_this_planning_period = math.inf
        self.best_fitness_this_planning_period_index = None
        self.second_best_fitness_this_planning_period = math.inf
        self.second_best_fitness_this_planning_period_index = None
        self.stationary_targetting_sim_index = None
        self.game_state_to_base_planning = None
        self.set_of_base_gamestate_timesteps = set()
        self.base_gamestates = {} # Key is timestep, value is the state
        self.other_ships_exist = None
        self.reality_move_sequence = []
        self.simulated_gamestate_history = {}
        self.lives_remaining_that_we_did_respawn_maneuver_for = set()

        global explanation_messages_with_timestamps
        if explanation_messages_with_timestamps:
            debug_print('This is not a fresh run of this controller!')
            explanation_messages_with_timestamps.clear()
        else:
            debug_print('Freshly running controller')

    def finish_init(self, game_state, ship_state):
        # If we need the game state or ship state to finish init, we can use this function to do that
        if self.ship_id is None:
            self.ship_id = ship_state['id']
        if gamestate_plotting:
            self.game_state_plotter = GameStatePlotter(game_state)
        if len(self.get_other_ships(game_state)) > 0:
            self.other_ships_exist = True
            print_explanation("I've got another ship friend here with me. I'll try coexisting with them, but be careful to avoid them.", self.current_timestep)
        else:
            self.other_ships_exist = False
            print_explanation("I'm alone. We can see into the future perfectly!", self.current_timestep)
        asteroid_density = ctrl.Antecedent(np.arange(0, 11, 1), 'asteroid_density')
        asteroids_entropy = ctrl.Antecedent(np.arange(0, 11, 1), 'asteroids_entropy')
        other_ship_lives = ctrl.Antecedent(np.arange(0, 4, 1), 'other_ship_lives')
        
        aggression = ctrl.Consequent(np.arange(0, 1, 1), 'asteroid_growth_factor')


    def get_other_ships(self, game_state):
        other_ships = []
        for ship in game_state['ships']:
            if ship['id'] != self.ship_id:
                other_ships.append(ship)
        return other_ships

    def enqueue_action(self, timestep, thrust=None, turn_rate=None, fire=None, drop_mine=None):
        if thrust is not None:
            first_nonnone_index = 0
        elif turn_rate is not None:
            first_nonnone_index = 1
        elif fire is not None:
            first_nonnone_index = 2
        elif drop_mine is not None:
            first_nonnone_index = 3
        else:
            first_nonnone_index = -1
        # first_nonnone_index is just a unique dummy variable since I don't want my none values being compared and crashing my script ughh
        heapq.heappush(self.action_queue, (timestep, first_nonnone_index, thrust, turn_rate, fire, drop_mine))

    def decide_next_action(self, game_state: dict=None, ship_state: dict=None):
        assert self.stationary_targetting_sim_index is not None or self.game_state_to_base_planning['respawning']
        debug_print(f"Deciding next action! Respawn maneuver status is: {self.game_state_to_base_planning['respawning']}")
        # Go through the list of planned maneuvers and pick the one with the best fitness function score
        # Update the state to base planning off of, so Neo can get to work on planning the next set of moves while this current set of moves executes
        #print('Going through sorted sims list to pick the best action')
        if self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['state_type'] == 'predicted':
            if ENABLE_ASSERTIONS:
                assert game_state is not None and ship_state is not None
            # Since the game is non-deterministic, we need to apply our simulated moves onto the actual corrected state, so errors don't build up
            best_action_sim_predicted: Simulation = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['sim']
            debug_print(f"\nPredicted best action sim first state:", best_action_sim_predicted.get_state_sequence()[0])
            best_action_fitness_predicted = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness']
            if ENABLE_ASSERTIONS:
                assert self.current_timestep == self.game_state_to_base_planning['timestep']
                assert game_state is not None
            best_predicted_sim_fire_next_timestep_flag = best_action_sim_predicted.get_fire_next_timestep_flag()
            debug_print(f"best_predicted_sim_fire_next_timestep_flag: {best_predicted_sim_fire_next_timestep_flag}, self.game_state_to_base_planning['fire_next_timestep_flag']: {self.game_state_to_base_planning['fire_next_timestep_flag']}")
            # self.game_state_to_base_planning['fire_next_timestep_flag'] is whether we fire at the BEGINNING of the period, while best_action_sim_predicted.get_fire_next_timestep_flag() is whether we fire AFTER this period
            debug_print('DECIDE NEXT ACTION REDO sim ast pending deah:')
            debug_print(self.game_state_to_base_planning['asteroids_pending_death'])
            best_action_sim = Simulation(game_state, ship_state, self.current_timestep, self.game_state_to_base_planning['ship_respawn_timer'], self.game_state_to_base_planning['asteroids_pending_death'], self.game_state_to_base_planning['forecasted_asteroid_splits'], self.game_state_to_base_planning['last_timestep_fired'], self.game_state_to_base_planning['last_timestep_mined'], self.game_state_to_base_planning['respawning'], self.game_state_to_base_planning['fire_next_timestep_flag'], self.game_state_plotter)
            best_action_sim_predicted_move_sequence = best_action_sim_predicted.get_move_sequence()
            debug_print(best_action_sim_predicted_move_sequence)
            best_action_sim.apply_move_sequence(best_action_sim_predicted_move_sequence)
            best_action_sim.set_fire_next_timestep_flag(best_predicted_sim_fire_next_timestep_flag)
            best_action_fitness = best_action_sim.get_fitness()
            debug_print(f"\nActual best action first state:", best_action_sim.get_state_sequence()[0])
            debug_print(f"\nUpdated simmed state. Old predicted fitness: {best_action_fitness_predicted}, new predicted fitness: {best_action_fitness}")
            if best_action_fitness > best_action_fitness_predicted + 0.1:
                print(f"\n\n\n\nDANGERRRRR!!!!! Updated simmed state. Old predicted fitness: {best_action_fitness_predicted}, new predicted fitness IS MUCH WORSE!!!!!!!: {best_action_fitness}")
                if self.second_best_fitness_this_planning_period_index is not None:
                    # The best action sim's reality is worse than expected. Try our second best as a backup and hopefully this will be better, and go according to plan!
                    second_best_action_sim_predicted: Simulation = self.sims_this_planning_period[self.second_best_fitness_this_planning_period_index]['sim']
                    second_best_action_fitness_predicted = self.sims_this_planning_period[self.second_best_fitness_this_planning_period_index]['fitness']
                    if ENABLE_ASSERTIONS:
                        assert second_best_action_fitness_predicted == self.second_best_fitness_this_planning_period
                    second_best_predicted_sim_fire_next_timestep_flag = second_best_action_sim_predicted.get_fire_next_timestep_flag()
                    second_best_action_sim = Simulation(game_state, ship_state, self.current_timestep, self.game_state_to_base_planning['ship_respawn_timer'], self.game_state_to_base_planning['asteroids_pending_death'], self.game_state_to_base_planning['forecasted_asteroid_splits'], self.game_state_to_base_planning['last_timestep_fired'], self.game_state_to_base_planning['last_timestep_mined'], self.game_state_to_base_planning['respawning'], self.game_state_to_base_planning['fire_next_timestep_flag'], self.game_state_plotter)
                    second_best_action_sim_predicted_move_sequence = second_best_action_sim_predicted.get_move_sequence()
                    second_best_action_sim.apply_move_sequence(second_best_action_sim_predicted_move_sequence)
                    second_best_action_sim.set_fire_next_timestep_flag(second_best_predicted_sim_fire_next_timestep_flag)
                    second_best_action_fitness = second_best_action_sim.get_fitness()
                    if second_best_action_fitness < best_action_fitness:
                        print(f"HOORAY, the second best action's real fitness of {second_best_action_fitness} and predicted fitness of {second_best_action_fitness_predicted} is better than the best!")
                        best_action_fitness = second_best_action_fitness
                        best_action_sim = second_best_action_sim
                    else:
                        print(f"CRAP, even the second best action's real fitness of {second_best_action_fitness} and predicted fitness of {second_best_action_fitness_predicted} isn't better than the first, so we'll just have to go with what we have and maybe get screwed.")

            if self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['type'] == 'targetting':
                # The targetting sim was done with the true state, so this should be the exact same and redundant
                raise Exception("WHY THE HECK IS IT IN HERE")
                assert best_action_fitness_predicted == best_action_fitness
        else:
            assert self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['state_type'] == 'exact'
            best_action_sim: Simulation = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['sim']
            best_action_fitness = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness']
        if best_action_fitness >= 10:
            # We're gonna die. Force select the one where I stay put and accept my fate, and don't even begin a maneuver.
            debug_print("RIP, I'm gonna die. Force select the one where I stay put and accept my fate, and don't even begin a maneuver.")
            print_explanation("RIP, I'm gonna die.", self.current_timestep)
            self.best_fitness_this_planning_period_index = self.stationary_targetting_sim_index
            best_action_sim: Simulation = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['sim']
            best_action_fitness = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness']
        stationary_safety_messages = self.sims_this_planning_period[0]['sim'].get_safety_messages()
        for message in stationary_safety_messages:
            print_explanation(message, self.current_timestep)
        debug_print(f"Best sim ID: {best_action_sim.get_sim_id()}, with index {self.best_fitness_this_planning_period_index} and fitness {best_action_fitness}")
        best_move_sequence = best_action_sim.get_move_sequence()
        best_action_sim_state_sequence = best_action_sim.get_state_sequence()
        debug_print(f"The action we're taking is from timestep {best_action_sim_state_sequence[0]['timestep']} to {best_action_sim_state_sequence[-1]['timestep']}")
        if VALIDATE_ALL_SIMULATED_STATES and not PRUNE_SIM_STATE_SEQUENCE:
            for state in best_action_sim_state_sequence:
                self.simulated_gamestate_history[state['timestep']] = state
        explanation_messages = best_action_sim.get_explanations()
        for explanation in explanation_messages:
            print_explanation(explanation, self.current_timestep)
        #end_state = sim_ship.get_state_sequence()[-1]
        #debug_print(f"Maneuver fitness: {best_maneuver_fitness}, stationary fitness: {best_stationary_targetting_fitness}")
        #print('state seq:', best_action_sim_state_sequence)
        debug_print('Best move seq:', best_move_sequence)
        debug_print(f"Best sim index: {self.best_fitness_this_planning_period_index}")
        debug_print(f"Choosing action: {self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['type']} with fitness {best_action_fitness}")
        #print('all sims this planning period:')
        #print(self.sims_this_planning_period)
        if not best_action_sim_state_sequence:
            #self.enqueue_action()
            debug_print(best_move_sequence)
            raise Exception("Why in the world is this state sequence empty?")
        best_action_sim_last_state = best_action_sim_state_sequence[-1]
        # Prune out the list of asteroids we shot at if the timestep (key) is in the past
        asteroids_pending_death = best_action_sim.get_asteroids_pending_death()
        asteroids_pending_death = {timestep: asteroids for timestep, asteroids in asteroids_pending_death.items() if timestep >= best_action_sim_last_state['timestep']}
        forecasted_asteroid_splits = best_action_sim.get_forecasted_asteroid_splits()
        game_state = best_action_sim.get_game_state()
        forecasted_asteroid_splits = maintain_forecasted_asteroids(forecasted_asteroid_splits, game_state, True)
        self.set_of_base_gamestate_timesteps.add(best_action_sim_last_state['timestep'])
        new_ship_state = best_action_sim.get_ship_state()
        new_fire_next_timestep_flag = best_action_sim.get_fire_next_timestep_flag()
        if new_ship_state['is_respawning'] and new_fire_next_timestep_flag and new_ship_state['lives_remaining'] not in self.lives_remaining_that_we_did_respawn_maneuver_for:
            debug_print(f"Forcing off the fire next timestep, because we just took damage")
            new_fire_next_timestep_flag = False
        if ENABLE_ASSERTIONS and new_ship_state['lives_remaining'] not in self.lives_remaining_that_we_did_respawn_maneuver_for and new_ship_state['is_respawning']:
            # If our ship is hurt in our next next action and I haven't done a respawn maneuver yet (in this situation, the next next action is a respawn maneuver)
            
            # Then I assert that our next action is not a respawning action, AND we're not firing at the start of the next next action
            if (self.game_state_to_base_planning['respawning'] or new_fire_next_timestep_flag):
                print(f"We haven't done a respawn maneuver for having {new_ship_state['lives_remaining']} lives left")
                print(f"self.game_state_to_base_planning['respawning']: {self.game_state_to_base_planning['respawning']}, new_fire_next_timestep_flag: {new_fire_next_timestep_flag}")
            #assert not (self.game_state_to_base_planning['respawning'] or new_fire_next_timestep_flag)
        self.game_state_to_base_planning = {
            'timestep': best_action_sim_last_state['timestep'],
            'respawning': new_ship_state['lives_remaining'] not in self.lives_remaining_that_we_did_respawn_maneuver_for and new_ship_state['is_respawning'],
            'ship_state': new_ship_state,
            'game_state': game_state,
            'ship_respawn_timer': best_action_sim.get_respawn_timer(),
            'asteroids_pending_death': asteroids_pending_death,
            'forecasted_asteroid_splits': forecasted_asteroid_splits,
            'last_timestep_fired': best_action_sim.get_last_timestep_fired(),
            'last_timestep_mined': best_action_sim.get_last_timestep_mined(),
            'fire_next_timestep_flag': new_fire_next_timestep_flag,
        }
        if ENABLE_ASSERTIONS:
            pass
            # TODO: FALSE BECUZ SEED 889195328 2024-01-23 12:33 AM
            if not (bool(self.game_state_to_base_planning['ship_respawn_timer']) == self.game_state_to_base_planning['ship_state']['is_respawning']):
                print(f"self.game_state_to_base_planning['ship_respawn_timer']: {self.game_state_to_base_planning['ship_respawn_timer']}, self.game_state_to_base_planning['ship_state']['is_respawning']: {self.game_state_to_base_planning['ship_state']['is_respawning']}")
            assert bool(self.game_state_to_base_planning['ship_respawn_timer']) == self.game_state_to_base_planning['ship_state']['is_respawning']
        if self.game_state_to_base_planning['respawning']:
            self.lives_remaining_that_we_did_respawn_maneuver_for.add(new_ship_state['lives_remaining'])
        debug_print(f"The next base state's respawning state is {self.game_state_to_base_planning['respawning']}")
        debug_print("The new ship state is", new_ship_state)
        debug_print(f"The fire next timestep flag is: {new_fire_next_timestep_flag}")
        debug_print(f"\nNext base state ts: {self.game_state_to_base_planning['timestep']}, respawn maneuver: {self.game_state_to_base_planning['respawning']}, respawn timer: {self.game_state_to_base_planning['ship_respawn_timer']}, ship state: {new_ship_state}")
        self.base_gamestates[best_action_sim_last_state['timestep']] = self.game_state_to_base_planning
        state_dump_dict = {
            'timestep': self.game_state_to_base_planning['timestep'],
            'ship_state': self.game_state_to_base_planning['ship_state'],
            'asteroids': self.game_state_to_base_planning['game_state']['asteroids'],
            'bullets': self.game_state_to_base_planning['game_state']['bullets'],
        }
        if KEY_STATE_DUMP:
            append_dict_to_file(state_dump_dict, 'Key Simulation State Dump.txt')
        
        if SIMULATION_STATE_DUMP and best_action_sim_last_state:
            for sim_state in best_action_sim_state_sequence:
                append_dict_to_file(sim_state, 'Simulation State Dump.txt')
        if gamestate_plotting and maneuver_sim_plotting and (start_gamestate_plotting_at_second is None or start_gamestate_plotting_at_second/DELTA_TIME <= self.current_timestep):
            for sim_state in best_action_sim_state_sequence:
                flattened_asteroids_pending_death = [ast for ast_list in sim_state['asteroids_pending_death'].values() for ast in ast_list]
                self.game_state_plotter.update_plot(sim_state['asteroids'], sim_state['ship_state'], sim_state['bullets'], [], [], flattened_asteroids_pending_death, sim_state['forecasted_asteroid_splits'], sim_state['mines'], True, 0.1, f"MANEUVER SIMULATION PREVIEW TIMESTEP {self.current_timestep}")
        #print(f"Best move sequence:", best_move_sequence)
        for move in best_move_sequence:
            if ENABLE_ASSERTIONS:
                if not move['timestep'] not in self.actioned_timesteps:
                    print("DUPLICATE TIMESTEPS")
                    print('actioned timesteps:', self.actioned_timesteps)
                    print('best move sequence:', best_move_sequence)
                    pass
                assert move['timestep'] not in self.actioned_timesteps, "DUPLICATE TIMESTEPS IN ENQUEUED MOVES"
                self.actioned_timesteps.add(move['timestep'])
            self.enqueue_action(move['timestep'], move['thrust'], move['turn_rate'], move['fire'], move['drop_mine'])
        self.sims_this_planning_period.clear()
        self.best_fitness_this_planning_period = math.inf
        self.best_fitness_this_planning_period_index = None
        self.second_best_fitness_this_planning_period = math.inf
        self.second_best_fitness_this_planning_period_index = None
        self.stationary_targetting_sim_index = None

    def plan_action(self, other_ships_exist: bool, base_state_is_exact: bool, iterations_boost: bool=False, plan_stationary: bool=False):
        # Simulate and look for a good move
        # We have two options. Stay put and focus on targetting asteroids, or we can come up with an avoidance maneuver and target asteroids along the way if convenient
        # We simulate both options, and take the one with the higher fitness score
        # If we stay still, we can potentially keep shooting asteroids that are on collision course with us without having to move
        # But if we're overwhelmed, it may be a lot better to move to a safer spot
        # The third scenario is that even if we're safe where we are, we may be able to be on the offensive and seek out asteroids to lay mines, so that can also increase the fitness function of moving, making it better than staying still
        # Our number one priority is to stay alive. Second priority is to shoot as much as possible. And if we can, lay mines without putting ourselves in danger.
        if self.game_state_to_base_planning['respawning']:
            debug_print("Planning respawn maneuver")
            # Simulate and look for a good move
            #print(f"Checking for imminent danger. We're currently at position {ship_state['position'][0]} {ship_state['position'][1]}")
            #print(f"Current ship location: {ship_state['position'][0]}, {ship_state['position'][1]}, ship heading: {ship_state['heading']}")

            # Check for danger
            #max_search_iterations = 60
            #min_search_iterations = 6
            search_iterations = 10
            max_cruise_seconds = 1 + 26*DELTA_TIME
            #ship_random_range, ship_random_max_maneuver_length = get_simulated_ship_max_range(max_cruise_seconds)
            #print(f"Respawn maneuver max length: {ship_random_max_maneuver_length}s")

            debug_print("Look for a respawn maneuver")
            # Run a simulation and find a course of action to put me to safety
            search_iterations_count = 0
            
            #while search_iterations_count < min_search_iterations or (not safe_maneuver_found and search_iterations_count < max_search_iterations):
            for _ in range(search_iterations):
                search_iterations_count += 1
                if search_iterations_count%5 == 0:
                    debug_print(f"Respawn search iteration {search_iterations_count}")
                    pass
                if not self.sims_this_planning_period:
                    # On the first iteration, try the null action. For ring scenarios, it may be best to stay at the center of the ring.
                    random_ship_heading_angle = 0
                    random_ship_accel_turn_rate = 0
                    random_ship_cruise_speed = 0
                    random_ship_cruise_turn_rate = 0
                    random_ship_cruise_timesteps = 1 # TODO: SET TO 0 LATER
                else:
                    random_ship_heading_angle = random.uniform(-20.0, 20.0)
                    random_ship_accel_turn_rate = random.uniform(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)
                    random_ship_cruise_speed = SHIP_MAX_SPEED*random.choice([-1, 1])
                    random_ship_cruise_turn_rate = 0
                    random_ship_cruise_timesteps = random.randint(0, round(max_cruise_seconds/DELTA_TIME))

                maneuver_sim = Simulation(self.game_state_to_base_planning['game_state'], self.game_state_to_base_planning['ship_state'], self.game_state_to_base_planning['timestep'], self.game_state_to_base_planning['ship_respawn_timer'], self.game_state_to_base_planning['asteroids_pending_death'], self.game_state_to_base_planning['forecasted_asteroid_splits'], self.game_state_to_base_planning['last_timestep_fired'], self.game_state_to_base_planning['last_timestep_mined'], True, False and self.game_state_to_base_planning['fire_next_timestep_flag'], self.game_state_plotter)
                if maneuver_sim.rotate_heading(random_ship_heading_angle) and maneuver_sim.accelerate(random_ship_cruise_speed, random_ship_accel_turn_rate) and maneuver_sim.cruise(random_ship_cruise_timesteps, random_ship_cruise_turn_rate) and maneuver_sim.accelerate(0):
                    # The ship went through all the steps without colliding
                    debug_print("The ship went through all the steps without colliding")
                    maneuver_complete_without_crash = True
                else:
                    # The ship crashed somewhere before reaching the final resting spot
                    debug_print("The ship crashed somewhere before reaching the final resting spot")
                    maneuver_complete_without_crash = False
                #if ENABLE_ASSERTIONS:
                    #assert maneuver_complete_without_crash
                    # NVM This isn't true in extreme cases! It means we're in real trouble
                #move_sequence = maneuver_sim.get_move_sequence()
                #state_sequence = maneuver_sim.get_state_sequence()
                #debug_print(f"\nGetting maneuver fitness:")
                #maneuver_sim.advance_asteroids_to_future_timestep()
                maneuver_fitness = maneuver_sim.get_fitness()
                maneuver_length = maneuver_sim.get_sequence_length() - 1
                debug_print(f"Respawn maneuver fitness: {maneuver_fitness}")
                #if maneuver_complete_without_crash:
                #    next_imminent_collision_time = maneuver_sim.get_next_extrapolated_collision_time()
                #else:
                #    next_imminent_collision_time = 0
                #safe_time_after_maneuver = max(0, next_imminent_collision_time)

                self.sims_this_planning_period.append({
                    'sim': maneuver_sim,
                    'fitness': maneuver_fitness,
                    #'maneuver_length': maneuver_length,
                    'type': 'respawn',
                    'state_type': 'exact' if base_state_is_exact else 'predicted',
                })
                if maneuver_fitness < self.best_fitness_this_planning_period:
                    self.second_best_fitness_this_planning_period = self.best_fitness_this_planning_period
                    self.second_best_fitness_this_planning_period_index = self.best_fitness_this_planning_period_index

                    self.best_fitness_this_planning_period = maneuver_fitness
                    self.best_fitness_this_planning_period_index = len(self.sims_this_planning_period) - 1

                '''
                if search_iterations_count == 1:
                    debug_print(f"The null respawn maneuver gives us a next collision time of {next_imminent_collision_time} s, safe time after maneuver is {safe_time_after_maneuver}, and the fitness is {maneuver_fitness}")
                #if safe_time_after_maneuver > best_safe_time_after_maneuver_found:
                if maneuver_fitness < best_maneuver_fitness_found:
                    debug_print(f"Alright we found a better one with next collision time {next_imminent_collision_time} s, safe time after maneuver is {safe_time_after_maneuver}, and the fitness is {maneuver_fitness}")
                    #best_safe_time_after_maneuver_found = safe_time_after_maneuver
                    best_maneuver_fitness_found = maneuver_fitness
                    best_imminent_collision_time_found = next_imminent_collision_time
                    best_maneuver_sim = maneuver_sim
                    best_safe_time_after_maneuver = safe_time_after_maneuver
                #if safe_time_after_maneuver >= safe_time_threshold:
                if best_maneuver_fitness_found < safe_fitness_threshold:
                    safe_maneuver_found = True
                    debug_print(f"Found safe maneuver! Next imminent collision time is {best_imminent_collision_time_found}")
                    debug_print(f"Maneuver takes this many seconds: {maneuver_length*DELTA_TIME}")
                    #print(f"Projected location to move will be to: {ship_sim_pos_x}, {ship_sim_pos_y} with heading {current_ship_heading + heading_difference_deg}")
                    #if search_iterations_count >= min_search_iterations:
                    #    break
                    #else:
                    #    continue
                '''
            #if search_iterations_count == max_search_iterations:
            #    debug_print("Hit the max iteration count")
            #print(f"Did {search_iterations_count} search iterations to find a respawn maneuver where we're safe for {best_safe_time_after_maneuver}s afterwards and has a fitness of {best_maneuver_fitness_found}. Moving to coordinates {best_maneuver_sim.get_state_sequence()[-1]['position'][0]} {best_maneuver_sim.get_state_sequence()[-1]['position'][1]} at timestep {best_maneuver_sim.get_state_sequence()[-1]['timestep']}")
            #print_explanation(f"Found respawn maneuver where we're safe for {best_safe_time_after_maneuver:0.1f} s afterwards.", self.current_timestep)
        else:
            # Stationary targetting simulation
            #best_stationary_targetting_fitness = math.inf # The lower the better
            #debug_print('Before sim, asteroids shot at:')
            #debug_print(self.asteroids_pending_death)
            #debug_print('And asteroids:')
            #debug_print(game_state['asteroids'])
            #debug_print('Simulating stationary targetting:')
            #print('\n\nAST PENDING DEATH AND FORECASTED SPLITS')
            #print(self.asteroids_pending_death)
            #print(self.forecasted_asteroid_splits)
            #write_to_json(self.asteroids_pending_death, 'BEFORE SIM.txt')
            #write_to_json(self.forecasted_asteroid_splits, 'BEFORE SIM.txt')
            #print(f"Simming stationary targetting, and fire_first_timestep is {self.fire_next_timestep_flag}")
            if plan_stationary:
                # The first list element is the stationary targetting
                #print('game state to base planning:')
                #print(self.game_state_to_base_planning)
                debug_print('Stationary sim ast pending deah:')
                debug_print(self.game_state_to_base_planning['asteroids_pending_death'])
                stationary_targetting_sim = Simulation(self.game_state_to_base_planning['game_state'], self.game_state_to_base_planning['ship_state'], self.game_state_to_base_planning['timestep'], self.game_state_to_base_planning['ship_respawn_timer'], self.game_state_to_base_planning['asteroids_pending_death'], self.game_state_to_base_planning['forecasted_asteroid_splits'], self.game_state_to_base_planning['last_timestep_fired'], self.game_state_to_base_planning['last_timestep_mined'], False, self.game_state_to_base_planning['fire_next_timestep_flag'], self.game_state_plotter)
                #print('\nAST PENDING DEATH AND FORECASTED SPLITS AFTER THE SIM')
                #print(self.asteroids_pending_death)
                #print(self.forecasted_asteroid_splits)
                #write_to_json(self.asteroids_pending_death, 'AFTER SIM.txt')
                #write_to_json(self.forecasted_asteroid_splits, 'AFTER SIM.txt')
                sim_complete_without_crash = stationary_targetting_sim.target_selection()
                debug_print('\nstationary targetting sim move seq')
                debug_print(stationary_targetting_sim.get_move_sequence())
                #best_stationary_targetting_move_sequence = stationary_targetting_sim.get_move_sequence()
                #print("stationary targetting move seq")
                #print(best_stationary_targetting_move_sequence)
                
                #debug_print('After sim, printing asteroids pending death:')
                #debug_print(self.asteroids_pending_death)
                #debug_print('also after sim, forecasted splits:')
                #debug_print(self.forecasted_asteroid_splits)
                #debug_print(f"\nGetting stationary targetting fitness:")
                
                best_stationary_targetting_fitness = stationary_targetting_sim.get_fitness()
                if not sim_complete_without_crash:
                    best_stationary_targetting_fitness += 10
                    #print(f"We got hit by the ship, yikes, so the fitness for sim id {stationary_targetting_sim.get_sim_id()} is being set to {best_stationary_targetting_fitness}")
                #print(f"Stationary targetting fitness: {best_stationary_targetting_fitness}")

                self.sims_this_planning_period.append({
                    'sim': stationary_targetting_sim,
                    'fitness': best_stationary_targetting_fitness,
                    'type': 'targetting',
                    'state_type': 'exact' if base_state_is_exact else 'predicted',
                })
                self.stationary_targetting_sim_index = len(self.sims_this_planning_period) - 1
                if best_stationary_targetting_fitness < self.best_fitness_this_planning_period:
                    self.second_best_fitness_this_planning_period = self.best_fitness_this_planning_period
                    self.second_best_fitness_this_planning_period_index = self.best_fitness_this_planning_period_index

                    self.best_fitness_this_planning_period = best_stationary_targetting_fitness
                    self.best_fitness_this_planning_period_index = self.stationary_targetting_sim_index

                debug_print(f"Planning targetting, and got fitness {best_stationary_targetting_fitness}")

            # Try moving! Run a simulation and find a course of action to put me to safety
            if other_ships_exist:
                if math.isinf(self.best_fitness_this_planning_period):
                    # This is the first timestep we're planning for this period, so we don't really know how many iterations to use. Don't go all out on this first one in case it's an easy one.
                    search_iterations = 2
                elif self.best_fitness_this_planning_period < 0.45:
                    search_iterations = 1
                elif self.best_fitness_this_planning_period < 0.85:
                    search_iterations = 1
                elif self.best_fitness_this_planning_period < 1:
                    search_iterations = 2
                elif self.best_fitness_this_planning_period < 2:
                    search_iterations = 3
                elif self.best_fitness_this_planning_period < 3:
                    search_iterations = 4
                elif self.best_fitness_this_planning_period < 4:
                    search_iterations = 5
                else:
                    search_iterations = 6
            else:
                if math.isinf(self.best_fitness_this_planning_period):
                    raise Exception("If there's no ships, why don't we have any sims this planning period yet? We should have done stationary first.")
                    # This is the first timestep we're planning for this period, so we don't really know how many iterations to use. Don't go all out on this first one in case it's an easy one.
                    search_iterations = 2
                elif self.best_fitness_this_planning_period < 0.45:
                    search_iterations = 1
                elif self.best_fitness_this_planning_period < 1.5:
                    search_iterations = 1
                elif self.best_fitness_this_planning_period < 5:
                    search_iterations = 2
                else:
                    search_iterations = 3
            max_cruise_seconds = 1
            #next_imminent_collision_time = math.inf

            #if not self.sims_this_planning_period:
            #    assert other_ships_exist
            #    search_iterations = min(search_iterations, 2)

            if iterations_boost:
                search_iterations = min(80, (search_iterations + 1)*10)

            if self.game_state_to_base_planning['ship_state']['lives_remaining'] == 1:
                # When down to our last life, try twice as hard to survive
                search_iterations *= 2
            elif self.game_state_to_base_planning['ship_state']['lives_remaining'] == 2:
                search_iterations = floor(search_iterations*1.5)
            for _ in range(search_iterations):
                random_ship_heading_angle = random.uniform(-30.0, 30.0)
                random_ship_accel_turn_rate = random.triangular(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE, 0)
                #random_ship_cruise_speed = random.uniform(-ship_max_speed, ship_max_speed)
                random_ship_cruise_speed = random.triangular(0, SHIP_MAX_SPEED, SHIP_MAX_SPEED)*random.choice([-1, 1])
                random_ship_cruise_turn_rate = random.triangular(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE, 0)
                random_ship_cruise_timesteps = random.randint(1, round(max_cruise_seconds/DELTA_TIME))

                # First do a dummy simulation just to go through the motion, so we have the list of moves
                start_time = time.time()
                #print("Simming maneuver preview")
                dummy_game_state = {'asteroids': [], 'mines': [], 'ships': [], 'bullets': [], 'map_size': self.game_state_to_base_planning['game_state']['map_size']}
                maneuver_preview = Simulation(dummy_game_state, self.game_state_to_base_planning['ship_state'], self.game_state_to_base_planning['timestep'], 0, {}, [], -math.inf, -math.inf, True, False)
                maneuver_preview.rotate_heading(random_ship_heading_angle)
                maneuver_preview.accelerate(random_ship_cruise_speed, random_ship_accel_turn_rate)
                maneuver_preview.cruise(random_ship_cruise_timesteps, random_ship_cruise_turn_rate)
                maneuver_preview.accelerate(0)
                preview_move_sequence = maneuver_preview.get_move_sequence()
                #end_time = time.time()
                #print(f'Cheap preview took {end_time - start_time} s. NOW DOING EXPENSIVE MANEUVER SIM')
                #start_time = time.time()
                #print(f"Mines before going into the polan action sim:", self.game_state_to_base_planning['game_state']['mines'])
                #print("Simming actual maneuver")
                #if self.game_state_to_base_planning['timestep'] == 42:
                #    print(f"Beginning a maneuver sim at ts 42. We're calling the function with these bullets:", self.game_state_to_base_planning['game_state']['bullets'])
                maneuver_sim = Simulation(self.game_state_to_base_planning['game_state'], self.game_state_to_base_planning['ship_state'], self.game_state_to_base_planning['timestep'], self.game_state_to_base_planning['ship_respawn_timer'], self.game_state_to_base_planning['asteroids_pending_death'], self.game_state_to_base_planning['forecasted_asteroid_splits'], self.game_state_to_base_planning['last_timestep_fired'], self.game_state_to_base_planning['last_timestep_mined'], False, self.game_state_to_base_planning['fire_next_timestep_flag'], self.game_state_plotter)
                # This if statement is a doozy. Keep in mind that we evaluate the and clauses left to right, and the moment we find one that's false, we stop.
                # While evaluating, the simulation is advancing, and if it crashes, then it'll evaluate to false and stop the sim.
                if maneuver_sim.simulate_maneuver(preview_move_sequence, True):
                    # The ship went through all the steps without colliding
                    maneuver_complete_without_crash = True
                else:
                    # The ship crashed somewhere before reaching the final resting spot
                    maneuver_complete_without_crash = False
                #end_time = time.time()
                #print(f"Expensive sim took {end_time - start_time} s")
                #move_sequence = maneuver_sim.get_move_sequence()
                #state_sequence = maneuver_sim.get_state_sequence()
                #debug_print(f"\nGetting maneuver fitness:")
                #if (len(self.sims_this_planning_period)) == 1:
                #    print(f"THE PREVIEW MOVE SEQUENCE FOR MANEUVER THATS MESSING UP IS:", preview_move_sequence)
                #    print(f"And the actual move seq is:", maneuver_sim.get_move_sequence())
                maneuver_length = maneuver_sim.get_sequence_length() - 1
                if maneuver_complete_without_crash:
                    maneuver_fitness = maneuver_sim.get_fitness()
                    #next_imminent_collision_time = maneuver_sim.get_next_extrapolated_collision_time()
                else:
                    maneuver_fitness = maneuver_sim.get_fitness() + 10
                    #next_imminent_collision_time = 0
                #safe_time_after_maneuver = max(0, next_imminent_collision_time)

                self.sims_this_planning_period.append({
                    'sim': maneuver_sim,
                    'fitness': maneuver_fitness,
                    'type': 'maneuver',
                    'state_type': 'exact' if base_state_is_exact else 'predicted',
                    #'safe_time_after_maneuver': safe_time_after_maneuver,
                    #'maneuver_length': maneuver_length,
                })
                debug_print(f"Planning random maneuver, and got fitness {maneuver_fitness}")
                if maneuver_fitness < self.best_fitness_this_planning_period:
                    self.second_best_fitness_this_planning_period = self.best_fitness_this_planning_period
                    self.second_best_fitness_this_planning_period_index = self.best_fitness_this_planning_period_index

                    self.best_fitness_this_planning_period = maneuver_fitness
                    self.best_fitness_this_planning_period_index = len(self.sims_this_planning_period) - 1

    def actions(self, ship_state: dict, game_state: dict) -> tuple[float, float, bool, bool]:
        global gamestate_plotting
        thrust_default, turn_rate_default, fire_default, drop_mine_default = 0, 0, False, False
        # Method processed each time step by this controller.
        self.current_timestep += 1
        if self.current_timestep == 0:
            # Only do these on the first timestep
            asteroids_count, current_count = asteroid_counter(game_state['asteroids'])
            print_explanation(f"The starting field has {current_count} asteroids on the screen, with a total of {asteroids_count} counting splits.", self.current_timestep)
            print_explanation(f"At my max shot rate, it'll take {asteroids_count/6:.01f} seconds to clear the field.", self.current_timestep)
            evaluate_scenario(game_state, ship_state)
        debug_print(f"\n\nTimestep {self.current_timestep}, ship id {ship_state['id']} is at {ship_state['position'][0]} {ship_state['position'][1]}")

        if not self.init_done:
            self.finish_init(game_state, ship_state)
            self.init_done = True
        #print("thrust is " + str(thrust) + "\n" + "turn rate is " + str(turn_rate) + "\n" + "fire is " + str(fire) + "\n")
        #if not (self.last_respawn_maneuver_timestep_range[0] <= self.current_timestep <= self.last_respawn_maneuver_timestep_range[1]):
            # We're not in the process of doing our respawn maneuver
        iterations_boost = False
        if self.current_timestep == 0:
            iterations_boost = True
        if self.other_ships_exist:
            # We cannot use deterministic mode to plan ahead
            # We can still try to plan ahead, but we need to compare the predicted state with the actual state
            # Note that there is the possibility we switch from this case, to the case where other ships don't exist, if they die

            # Since other ships exist and the game isn't deterministic, we can die at any time even during the middle of a planned maneuver where we SHOULD survive.
            # Check for that case:
            unexpected_death = False
            if ship_state['is_respawning'] and ship_state['lives_remaining'] not in self.lives_remaining_that_we_did_respawn_maneuver_for:
                print("Ouch, I died in the middle of a maneuver where I expected to survive, due to other ships being present!")
                # Clear the move queue, since previous moves have been invalidated by us taking damage
                self.action_queue.clear()
                self.actioned_timesteps.clear() # If we don't clear it, we'll have duplicated moves since we have to overwrite our planned moves to get to safety, which means enqueuing moves on timesteps we already enqueued moves for.
                self.fire_next_timestep_flag = False # If we were planning on shooting this timestep but we unexpectedly got hit, DO NOT SHOOT! Actually even if we didn't reset this variable here, we'd only shoot after the respawn maneuver is done and then we'd miss a shot. And yes that was a bug that I fixed lmao
                self.game_state_to_base_planning = None
                self.sims_this_planning_period.clear()
                self.best_fitness_this_planning_period_index = None
                self.best_fitness_this_planning_period = math.inf
                self.second_best_fitness_this_planning_period_index = None
                self.second_best_fitness_this_planning_period = math.inf
                unexpected_death = True
                iterations_boost = True
            
            # Set up the actions planning
            if not self.game_state_to_base_planning:
                self.game_state_to_base_planning = {
                    'timestep': self.current_timestep,
                    'respawning': ship_state['is_respawning'] and ship_state['lives_remaining'] not in self.lives_remaining_that_we_did_respawn_maneuver_for,
                    'ship_state': ship_state,
                    'game_state': preprocess_bullets_in_gamestate(game_state),
                    'ship_respawn_timer': 3 if unexpected_death else 0,
                    'asteroids_pending_death': {},
                    'forecasted_asteroid_splits': [],
                    'last_timestep_fired': -math.inf,#self.current_timestep - 1, # TODO: WHY is this not -math.inf? Did I do this due to safety, if it fires too soon? Check for this edgecase!
                    'last_timestep_mined': -math.inf,
                    'fire_next_timestep_flag': False,
                }
                if self.game_state_to_base_planning['respawning']:
                    self.lives_remaining_that_we_did_respawn_maneuver_for.add(ship_state['lives_remaining'])
            
            
            if self.action_queue:
                self.plan_action(self.other_ships_exist, False, iterations_boost, False)
            else:
                # Refresh the base state now that we have the true base state!
                debug_print('REFRESHING BASE STATE FOR STATIONARY ON TS', self.current_timestep)
                self.game_state_to_base_planning['ship_state'] = ship_state
                self.game_state_to_base_planning['game_state'] = preprocess_bullets_in_gamestate(game_state)
                '''
                self.game_state_to_base_planning = {
                    'timestep': self.current_timestep,
                    'respawning': ship_state['is_respawning'] and ship_state['lives_remaining'] not in self.lives_remaining_that_we_did_respawn_maneuver_for,
                    'ship_state': ship_state,
                    'game_state': preprocess_bullets_in_gamestate(game_state),
                    'ship_respawn_timer': 3 if unexpected_death else 0, # We actually don't know exactly when we took damage lmao
                    'asteroids_pending_death': self.game_state_to_base_planning['asteroids_pending_death'],
                    'forecasted_asteroid_splits': self.game_state_to_base_planning['forecasted_asteroid_splits'],
                    'last_timestep_fired': self.game_state_to_base_planning['last_timestep_fired'],
                    'fire_next_timestep_flag': self.game_state_to_base_planning['fire_next_timestep_flag'],
                }
                '''
                self.plan_action(self.other_ships_exist, True, iterations_boost, True)
                assert self.current_timestep == self.game_state_to_base_planning['timestep']
                self.decide_next_action(preprocess_bullets_in_gamestate(game_state), ship_state) # Since other ships exist and this is non-deterministic, we constantly feed in the updated reality
                if len(self.get_other_ships(game_state)) == 0:
                    print_explanation("I'm alone. We can see into the future perfectly now!", self.current_timestep)
                    self.other_ships_exist = False
        else:
            if not self.game_state_to_base_planning:
                # Set up the actions planning
                if ENABLE_ASSERTIONS:
                    assert self.current_timestep == 0
                if self.current_timestep == 0:
                    iterations_boost = True
                # TODO: On the first timestep, spend more time planning actions in case we need to immediately evade!
                self.game_state_to_base_planning = {
                    'timestep': self.current_timestep,
                    'respawning': False, # On the first timestep 0, the is_respawning flag is ALWAYS false, even if we spawn inside asteroids.
                    'ship_state': ship_state,
                    'game_state': preprocess_bullets_in_gamestate(game_state),
                    'ship_respawn_timer': 0,
                    'asteroids_pending_death': {},
                    'forecasted_asteroid_splits': [],
                    'last_timestep_fired': -math.inf,#self.current_timestep - 1, # TODO: CHECK EDGECASE, may need to restore to larger number to be safe
                    'last_timestep_mined': -math.inf,
                    'fire_next_timestep_flag': False,
                }
            # No matter what, spend some time evaluating the best action from the next predicted state
            # When no ships are around, the stationary targetting is the first thing done
            if not self.sims_this_planning_period:
                self.plan_action(self.other_ships_exist, True, iterations_boost, True)
            else:
                self.plan_action(self.other_ships_exist, True, iterations_boost, False)
            if not self.action_queue:
                # Nothing's in the action queue. Evaluate the current situation and figure out the best course of action
                assert self.current_timestep == self.game_state_to_base_planning['timestep']
                debug_print("Decide the next action.")
                self.decide_next_action()
        
        # Execute the actions in the queue for this timestep
        # Initialize defaults. If a component of the action is missing, then the default value will be returned
        thrust_combined, turn_rate_combined, fire_combined, drop_mine_combined = thrust_default, turn_rate_default, fire_default, drop_mine_default

        while self.action_queue and self.action_queue[0][0] == self.current_timestep:
            _, _, thrust, turn_rate, fire, drop_mine = heapq.heappop(self.action_queue)
            thrust_combined = thrust if thrust is not None else thrust_combined
            turn_rate_combined = turn_rate if turn_rate is not None else turn_rate_combined
            fire_combined = fire if (fire is not None and fire_combined is not True) else fire_combined # Fire being true takes priority over false! This lets the fire last timestep enqueue to work, since it sets it to true before the sim's move sequence sets it to false
            drop_mine_combined = drop_mine if drop_mine is not None else drop_mine_combined
        if fire_combined is True and ship_state['can_fire']:
            self.last_timestep_fired = self.current_timestep
        # The next action in the queue is for a future timestep. All actions for this timestep are processed.
        # Bounds check the stuff before the Kessler code complains to me about it
        if thrust_combined < -SHIP_MAX_THRUST or thrust_combined > SHIP_MAX_THRUST:
            thrust_combined = min(max(-SHIP_MAX_THRUST, thrust_combined), SHIP_MAX_THRUST)
            raise Exception("Dude the thrust is too high, go fix your code >:(")
        if turn_rate_combined < -SHIP_MAX_TURN_RATE or turn_rate_combined > SHIP_MAX_TURN_RATE:
            turn_rate_combined = min(max(-SHIP_MAX_TURN_RATE, turn_rate_combined), SHIP_MAX_TURN_RATE)
            raise Exception("Dude the turn rate is too high, go fix your code >:(")
        if fire_combined and not ship_state['can_fire']:
            self.reality_move_sequence.append({'thrust': thrust_combined, 'turn_rate': turn_rate_combined, 'fire': fire_combined, 'drop_mine': drop_mine_combined})
            print(self.reality_move_sequence)
            raise Exception("Why are you trying to fire when you haven't waited out the cooldown yet?")
        #if drop_mine_combined and not ship_state['can_deploy_mine']:
        #    print("You can't deploy mines dude!")
        debug_print(f"Inputs on timestep {self.current_timestep} - thrust: {thrust_combined}, turn_rate: {turn_rate_combined}, fire: {fire_combined}, drop_mine: {drop_mine_combined}")

        if self.current_timestep > slow_down_game_after_second/DELTA_TIME:
            time.sleep(slow_down_game_pause_time)

        if gamestate_plotting and self.game_state_to_base_planning and (start_gamestate_plotting_at_second is None or start_gamestate_plotting_at_second/DELTA_TIME <= self.current_timestep):
            flattened_asteroids_pending_death = [ast for ast_list in self.game_state_to_base_planning['asteroids_pending_death'].values() for ast in ast_list]
            self.game_state_plotter.update_plot(game_state['asteroids'], ship_state, game_state['bullets'], [], [], flattened_asteroids_pending_death, self.game_state_to_base_planning['forecasted_asteroid_splits'], game_state['mines'], True, EPS, f'REALITY TIMESTEP {self.current_timestep}')
        state_dump_dict = {
            'timestep': self.current_timestep,
            'ship_state': ship_state,
            'asteroids': game_state['asteroids'],
            'bullets': game_state['bullets'],
        }
        if REALITY_STATE_DUMP:
            append_dict_to_file(state_dump_dict, 'Reality State Dump.txt')
        if KEY_STATE_DUMP and self.current_timestep in self.set_of_base_gamestate_timesteps:
            append_dict_to_file(state_dump_dict, 'Key Reality State Dump.txt')
        if VALIDATE_ALL_SIMULATED_STATES and not PRUNE_SIM_STATE_SEQUENCE and not self.other_ships_exist:
            debug_print(f"Validating game state for timestep {self.current_timestep}")
            if self.current_timestep in self.simulated_gamestate_history:
                ship_states_match = compare_shipstates(ship_state, self.simulated_gamestate_history[self.current_timestep]['ship_state'])
                if not ship_states_match:
                    print("Actual ship state:", ship_state)
                    print("\nSimulated ship state:", self.simulated_gamestate_history[self.current_timestep]['ship_state'])
                assert ship_states_match
                if not ship_state['is_respawning']:
                    # If respawning, the asteroid states won't match due to an optimization, so skip the check
                    game_states_match = compare_gamestates(game_state, self.simulated_gamestate_history[self.current_timestep]['game_state'])
                    if not game_states_match:
                        print("Actual game state:", game_state)
                        print("\nSimulated game state:", self.simulated_gamestate_history[self.current_timestep]['game_state'])
                    assert game_states_match
            else:
                print(f"Timestep not in list of states!!!")
        if (not VALIDATE_ALL_SIMULATED_STATES or PRUNE_SIM_STATE_SEQUENCE) and VALIDATE_SIMULATED_KEY_STATES and self.current_timestep in self.set_of_base_gamestate_timesteps and not self.other_ships_exist:
            debug_print(f"Validating KEY game state for timestep {self.current_timestep}")
            game_states_match = compare_gamestates(game_state, self.base_gamestates[self.current_timestep]['game_state'])
            if not game_states_match:
                print("Actual game state:", game_state)
                print("\nSimulated game state:", self.base_gamestates[self.current_timestep]['game_state'])
            assert game_states_match
            ship_states_match = compare_shipstates(ship_state, self.base_gamestates[self.current_timestep]['ship_state'])
            if not ship_states_match:
                print("Actual ship state:", ship_state)
                print("\nSimulated ship state:", self.base_gamestates[self.current_timestep]['ship_state'])
            assert ship_states_match
        self.reality_move_sequence.append({'thrust': thrust_combined, 'turn_rate': turn_rate_combined, 'fire': fire_combined, 'drop_mine': drop_mine_combined})
        return thrust_combined, turn_rate_combined, fire_combined, drop_mine_combined

if __name__ == '__main__':
    print("This is a Kessler controller meant to be imported, and not run directly!")

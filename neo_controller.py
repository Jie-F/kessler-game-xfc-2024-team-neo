# XFC 2024 - "Neo" Kessler controller
# Jie Fan 2023-2024
# jie.f@pm.me
# Feel free to reach out if you have questions, suggestions, or find a bug :)

from src.kesslergame import KesslerController
import random
import math
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import heapq
import time
import os
import bisect
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import copy

# TODO: Dynamic culling for sim? Different radii, or maybe direction?

debug_mode = True
gamestate_plotting = False
reality_state_dump = False
simulation_state_dump = False
enable_assertions = True

delta_time = 1/30 # s/ts
#fire_time = 1/10  # seconds
bullet_speed = 800.0 # px/s
ship_max_turn_rate = 180.0 # deg/s
ship_max_thrust = 480.0 # px/s^2
ship_drag = 80.0 # px/s^2
ship_max_speed = 240.0 # px/s
eps = 0.00000001
ship_radius = 20.0 # px
timesteps_until_ship_achieves_max_speed = math.ceil(ship_max_speed/(ship_max_thrust - ship_drag)/delta_time) # Should be 18 timesteps
collision_check_pad = 2 # px
asteroid_aim_buffer_pixels = 7 # px
coordinate_bound_check_padding = 1 # px
mine_blast_radius = 150 # px
mine_radius = 12 # px
mine_blast_pressure = 2000
mine_fuse_time = 3 # s
asteroid_radii_lookup = [8*size for size in range(5)] # asteroid.py
asteroid_mass_lookup = [0.25*math.pi*(8*size)**2 for size in range(5)] # asteroid.py
respawn_invincibility_time = 3 # s
bullet_mass = 1
ship_avoidance_padding = 25
ship_avoidance_speed_padding_ratio = 1/100
bullet_length = 12 # px

class GameStatePlotter:
    def __init__(self, game_state):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, game_state['map_size'][0]])
        self.ax.set_ylim([0, game_state['map_size'][1]])
        self.ax.set_aspect('equal', adjustable='box')
        self.game_state = game_state

    def update_plot(self, asteroids: dict = [], ship_state: dict | None = None, bullets: list = [], circled_asteroids: list = [], ghost_asteroids: list = [], forecasted_asteroids: list = [], clear_plot=True, pause_time=eps, plot_title=""):
        if clear_plot:
            self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.set_xlim([0, self.game_state['map_size'][0]])
        self.ax.set_ylim([0, self.game_state['map_size'][1]])
        self.ax.set_aspect('equal', adjustable='box')

        self.ax.set_title(plot_title, fontsize=14, color='black')

        # Draw asteroids and their velocities
        for a in asteroids:
            asteroid_circle = patches.Circle(a['position'], a['radius'], color='gray', fill=True)
            self.ax.add_patch(asteroid_circle)
            self.ax.arrow(a['position'][0], a['position'][1], a['velocity'][0]*delta_time, a['velocity'][1]*delta_time, head_width=3, head_length=5, fc='white', ec='white')
        for a in ghost_asteroids:
            asteroid_circle = patches.Circle(a['position'], a['radius'], color='#333333', fill=True, zorder=-100)
            self.ax.add_patch(asteroid_circle)
            #self.ax.arrow(a['position'][0], a['position'][1], a['velocity'][0]*delta_time, a['velocity'][1]*delta_time, head_width=3, head_length=5, fc='white', ec='white')
        for a in forecasted_asteroids:
            asteroid_circle = patches.Circle(a['position'], a['radius'], color='#440000', fill=True, zorder=100)
            self.ax.add_patch(asteroid_circle)
        #print(highlighted_asteroids)
        for a in circled_asteroids:
            #print('asteroid', a)
            highlight_circle = patches.Circle(a['position'], a['radius'] + 5, color='orange', fill=False)
            self.ax.add_patch(highlight_circle)

        if ship_state:
            # Draw the ship as an elongated triangle
            ship_size_base = ship_radius
            ship_size_tip = ship_radius
            ship_heading = ship_state['heading']
            ship_position = ship_state['position']
            angle_rad = np.radians(ship_heading)
            ship_vertices = [
                (ship_position[0] + ship_size_tip*np.cos(angle_rad), ship_position[1] + ship_size_tip*np.sin(angle_rad)),
                (ship_position[0] + ship_size_base*np.cos(angle_rad + np.pi*3/4), ship_position[1] + ship_size_base*np.sin(angle_rad + np.pi*3/4)),
                (ship_position[0] + ship_size_base*np.cos(angle_rad - np.pi*3/4), ship_position[1] + ship_size_base*np.sin(angle_rad - np.pi*3/4)),
            ]
            ship = patches.Polygon(ship_vertices, color='green', fill=True)
            self.ax.add_patch(ship)

            # Draw the ship's hitbox as a blue circle
            ship_circle = patches.Circle(ship_position, ship_radius, color='blue', fill=False)
            self.ax.add_patch(ship_circle)

        # Draw arrow line segments for bullets
        for b in bullets:
            bullet_tail = (b['position'][0] - bullet_length*math.cos(math.radians(b['heading'])), b['position'][1] - bullet_length*math.sin(math.radians(b['heading'])))
            self.ax.arrow(bullet_tail[0], bullet_tail[1], b['position'][0] - bullet_tail[0], b['position'][1] - bullet_tail[1], head_width=3, head_length=5, fc='red', ec='red')

        plt.draw()
        plt.pause(pause_time)

    def is_asteroid_in_list(self, asteroid_list, asteroid):
        # Assuming you have a function to check if an asteroid is in the list
        return asteroid in asteroid_list

    def start_animation(self):
        # This can be an empty function if you are calling update_plot manually in your game loop
        pass

def debug_print(message):
    if debug_mode:
        print(message)

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
    return f"Pos: ({a['position'][0]:0.2f}, {a['position'][1]:0.2f}), Vel: ({a['velocity'][0]:0.2f}, {a['velocity'][1]:0.2f}), Size: {a['size']}"

def angle_difference_rad(angle1, angle2):
    # Calculate the raw difference
    raw_diff = angle1 - angle2

    # Adjust for wraparound using modulo
    adjusted_diff = raw_diff % (2*math.pi)

    # If the difference is greater than pi, adjust to keep within -pi to pi
    if adjusted_diff > math.pi:
        adjusted_diff -= 2*math.pi

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

def min_rotation_to_align_deg(current_heading, target_heading):
    # Current and target headings are within 0 to 360 degrees

    # Calculate the differences for both direct and reverse alignment
    #current + (target-current) = target
    direct_diff = angle_difference_deg(target_heading, current_heading)
    reverse_heading = (current_heading + 180) % 360
    reverse_diff = angle_difference_deg(target_heading, reverse_heading)
    # Choose the one with the minimum absolute rotation
    if abs(direct_diff) <= abs(reverse_diff):
        return direct_diff, 1
    else:
        return reverse_diff, -1

def check_collision(a_x, a_y, a_r, b_x, b_y, b_r):
    if (a_x - b_x)**2 + (a_y - b_y)**2 < (a_r + b_r + collision_check_pad)**2:
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
    # If both objects are stationary, then we only have to check the collision right now and not do any fancy math
    # This should speed up scenarios where most asteroids are stationary
    if math.isclose(Dax, 0, abs_tol=eps) and math.isclose(Day, 0, abs_tol=eps) and math.isclose(Dbx, 0, abs_tol=eps) and math.isclose(Dby, 0, abs_tol=eps):
        if check_collision(Oax, Oay, ra, Obx, Oby, rb):
            t1 = -math.inf
            t2 = math.inf
        else:
            t1 = math.nan
            t2 = math.nan
        return t1, t2
    a = Dax**2 + Dbx**2 + Day**2 + Dby**2 - 2*Dax*Dbx - 2*Day*Dby
    b = 2*(Oax*Dax - Oax*Dbx - Obx*Dax + Obx*Dbx + Oay*Day - Oay*Dby - Oby*Day + Oby*Dby)
    c = Oax**2 + Obx**2 + Oay**2 + Oby**2 - 2*(Oax*Obx + Oay*Oby) - (ra + rb)**2
    d = b**2 - 4*a*c
    if (a != 0) and (d >= 0):
        t1 = (-b + math.sqrt(d)) / (2*a)
        t2 = (-b - math.sqrt(d)) / (2*a)
    else:
        if enable_assertions:
            assert a != 0
        t1 = math.nan
        t2 = math.nan
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

def unwrap_asteroid(asteroid, max_x, max_y, pattern='half_surround', directional_culling=True):
    duplicates = []
    # Original X and Y coordinates
    orig_x, orig_y = asteroid["position"]
    # Generate positions for the unwrapped duplicates
    for col, dx in enumerate([-max_x, 0, max_x]):
        for row, dy in enumerate([-max_y, 0, max_y]):
            if pattern == 'surround':
                # The default surround pattern multiplies the number of asteroids by 9
                pass
            elif pattern == 'cross':
                # This pattern multiplies the number of asteroids by 5
                if (col, row) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                    continue
            elif pattern == 'stack':
                # This pattern multiplies the number of asteroids by 3
                if (col, row) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)]:
                    continue
            elif pattern == 'half_surround':
                # This pattern multiplies the number of asteroids by 4
                if orig_x*2 <= max_x:
                    if orig_y*2 <= max_y:
                        # Bottom left
                        if not (dx >= 0 and dy >= 0):
                            continue
                    else:
                        # Top left
                        if not (dx >= 0 and dy <= 0):
                            continue
                else:
                    if orig_y*2 <= max_y:
                        # Bottom right
                        if not (dx <= 0 and dy >= 0):
                            continue
                    else:
                        # Top right
                        if not (dx <= 0 and dy <= 0):
                            continue
            elif pattern == 'none':
                # This pattern multiplies the number of asteroids by 1
                if (dx, dy) != (0, 0):
                    continue
            if directional_culling:
                # Check to make sure the asteroids are headed in the direction of the main game bounds
                # This optimization multiplies the amount of duplicates asteroids by 0.375 approximately, for the surround/half_surround pattern
                # TODO: THIS DOESN'T CONSIDER the case where we want to duplicate for searching for a safe spot to move, since I want to have a bubble around me that extends beyond the bounds, and I don't want to cull those away
                if (dx, dy) != (0, 0):
                    if (dx != 0 and np.sign(dx) == np.sign(asteroid['velocity'][0])) or (dy != 0 and np.sign(dy) == np.sign(asteroid['velocity'][1])):
                        # This unwrapped asteroid's heading out into space
                        continue
            #print(f"It: col{col} row{row}")
            duplicate = dict(asteroid) # TODO: Check whether this copy is necessary
            duplicate['position'] = (orig_x + dx, orig_y + dy)
            
            duplicates.append(duplicate)
    return duplicates

def check_coordinate_bounds(game_state, x, y):
    if 0 <= x < game_state['map_size'][0] and 0 <= y < game_state['map_size'][1]:
        return True
    else:
        return False

def alternative_interception_calc(ship_pos_x, ship_pos_y, asteroid_pos_x, asteroid_pos_y, asteroid_vel_x, asteroid_vel_y, asteroid_r, ship_heading, game_state, future_shooting_timesteps=0):
    pass

def calculate_interception(ship_pos_x, ship_pos_y, asteroid_pos_x, asteroid_pos_y, asteroid_vel_x, asteroid_vel_y, asteroid_r, ship_heading, game_state, future_shooting_timesteps=0):
    # The interception time returned is the time in seconds from when the bullet is shot to when it hits the center of the asteroid

    # Use future shooting time to extrapolate asteroid location to the firing time
    asteroid_pos_x += asteroid_vel_x*future_shooting_timesteps*delta_time
    asteroid_pos_y += asteroid_vel_y*future_shooting_timesteps*delta_time
    # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
    asteroid_ship_x = ship_pos_x - asteroid_pos_x
    asteroid_ship_y = ship_pos_y - asteroid_pos_y
    
    asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)
    
    asteroid_direction = math.atan2(asteroid_vel_y, asteroid_vel_x) # Velocity is a 2-element array [vx,vy].
    theta = asteroid_ship_theta - asteroid_direction
    cos_theta = math.cos(theta)
    # Need the speeds of the asteroid and bullet. speed*time is distance to the intercept point
    asteroid_speed_square = asteroid_vel_x**2 + asteroid_vel_y**2
    asteroid_speed = math.sqrt(asteroid_speed_square)
    
    # Discriminant of the quadratic formula b^2-4ac
    asteroid_dist_square = (ship_pos_x - asteroid_pos_x)**2 + (ship_pos_y - asteroid_pos_y)**2
    asteroid_dist = math.sqrt(asteroid_dist_square)
    bullet_speed_square = bullet_speed**2
    discriminant = 4*asteroid_dist_square*asteroid_speed_square*(cos_theta)**2 - 4*(asteroid_speed_square - bullet_speed_square)*asteroid_dist_square
    if discriminant < 0:
        # There is no intercept.
        return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
    sqrt_discriminant = math.sqrt(discriminant)
    intercept1_s = ((2*asteroid_dist*asteroid_speed*cos_theta) + sqrt_discriminant) / (2*(asteroid_speed_square - bullet_speed_square))
    intercept2_s = ((2*asteroid_dist*asteroid_speed*cos_theta) - sqrt_discriminant) / (2*(asteroid_speed_square - bullet_speed_square))
    #print(f"intercept 1: {intercept1}, intercept2: {intercept2}")
    # Take the smaller intercept time, as long as it is a positive time FROM THE SHOOTING TIME; if not, take the larger one.
    # The intercept time is time in seconds counting from the shooting time, not from the current time
    # Therefore we accept negative times, because it could still be in the future relative to the current time
    if not (intercept1_s >= 0 or intercept2_s >= 0):
        # There is no future intercept.
        return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    elif intercept1_s >= 0 and intercept2_s >= 0:
        interception_time_s = min(intercept1_s, intercept2_s)
    else:
        interception_time_s = max(intercept1_s, intercept2_s)
    
    # Calculate the intercept point. The work backwards to find the ship's firing angle shot_heading.
    intercept_x = asteroid_pos_x + asteroid_vel_x*interception_time_s
    intercept_y = asteroid_pos_y + asteroid_vel_y*interception_time_s

    # If the asteroid is out of bounds, then it will wrap around and this shot isn't feasible
    # However, if an unwrapped asteroid was passed into this function and the interception is inbounds, then it's a feasible shot
    feasible = check_coordinate_bounds(game_state, intercept_x, intercept_y)
    
    shot_heading = math.atan2(intercept_y - ship_pos_y, intercept_x - ship_pos_x)
    
    # Lastly, find the difference between firing angle and the ship's current orientation.
    shot_heading_error_rad = angle_difference_rad(shot_heading, math.radians(ship_heading))

    # Calculate the amount off of the shot heading I can be and still hit the asteroid
    #print(f"Asteroid radius {asteroid_r} asteroid distance {asteroid_dist}")
    if asteroid_r < asteroid_dist:
        shot_heading_tolerance_rad = math.asin((asteroid_r - asteroid_aim_buffer_pixels)/asteroid_dist)
    else:
        shot_heading_tolerance_rad = math.pi/4
    
    asteroid_dist_during_interception = math.sqrt((ship_pos_x - intercept_x)**2 + (ship_pos_y - intercept_y)**2)
    return feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time_s + 0*future_shooting_timesteps*delta_time, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception

def target_priority(game_state, ship_state, a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y, imminent_collision_time_s, next_imminent_collision_time_stationary_s, close_proximity_asteroid_count):
    # OBSOLETE!!!!!!
    # TODO: Use a fuzzy system to optimize this using rules and stuff
    # The lower the score, the better the priority!

    #asteroid_aiming_score = min(6, aiming_timesteps_required/5 + abs(shooting_angle_error_deg)/10 + interception_time_s)
    if close_proximity_asteroid_count > 6:
        asteroid_aiming_score = min(10, aiming_timesteps_required/2)
    else:
        asteroid_aiming_score = min(20, aiming_timesteps_required**2)
    #print(f"aiming_timesteps_required: {aiming_timesteps_required/5:0.2f} shooting_angle_error_deg: {abs(shooting_angle_error_deg)/10:0.2f} interception_time_s: {interception_time_s:0.2f} imminent_collision_time_s: {imminent_collision_time_s:0.2f} asteroid_dist_during_interception: {asteroid_dist_during_interception/500:0.2f}")
    # Target selection works differently depending on whether we have mines left, and whether this asteroid will get hit by a mine
    mines_list = game_state['mines']

    asteroid_mine_score = 0
    if len(mines_list) > 0:
        # There are currently mines on the board
        asteroid_in_mine_blast = False
        mine_is_mine = False
        for m in mines_list:
            if count_asteroids_in_mine_blast_radius(game_state, [a], m['position'][0], m['position'][1], 0) == 1:
                asteroid_in_mine_blast = True
                mine_is_mine = True # TODO: TEMPORARY, DETERMINE THIS BY TRACKING THE MINES I LAID
                break # Don't care about the other mines, since this asteroid is inside at least one mine
        # Might be smart to differentiate between asteroids that are within one mine blast, and asteroids within two.
        # But this makes things super complicate so yeah let's not worry about that. This is already considering a lot of factors.
        if asteroid_in_mine_blast:
            if mine_is_mine:
                # I want to maximize my score by splitting up asteroids within my mine's blast radius
                if a['size'] == 1:
                    asteroid_mine_score += 3
                elif a['size'] == 2:
                    asteroid_mine_score += 0
                elif a['size'] == 3:
                    asteroid_mine_score += 1
                elif a['size'] == 4:
                    asteroid_mine_score += 2
            else:
                # I want to minimize my opponent's score by shooting individual asteroids within their blast radius, so their mine is less effective
                if a['size'] == 1:
                    asteroid_mine_score += 0
                elif a['size'] == 2:
                    asteroid_mine_score += 3
                elif a['size'] == 3:
                    asteroid_mine_score += 2
                elif a['size'] == 4:
                    asteroid_mine_score += 1
        else:
            # Penalize the asteroid for not being in a mine blast, since shooting large asteroids in my own mine blast is better for my score than shooting ones outside
            asteroid_mine_score += 2
    else:
        # There are no active mines on the board
        if ship_state['mines_remaining'] > 0:
            # If I have mines left, I want to create more mining opportunities by shooting asteroids of size >1, and leave asteroids of size 1
            if a['size'] == 1:
                asteroid_mine_score += 3
            elif a['size'] == 2:
                asteroid_mine_score += 0
            elif a['size'] == 3:
                asteroid_mine_score += 1
            elif a['size'] == 4:
                asteroid_mine_score += 2
        else:
            # I don't have mines left. Shoot asteroids of size 1. Don't shoot larger ones.
            if a['size'] == 1:
                asteroid_mine_score += 0
            elif a['size'] == 2:
                asteroid_mine_score += 3
            elif a['size'] == 3:
                asteroid_mine_score += 2
            elif a['size'] == 4:
                asteroid_mine_score += 1
    
    if imminent_collision_time_s <= 2:
        imminent_collision_score = 0
    elif imminent_collision_time_s >= 10:
        imminent_collision_score = 4
    else:
        imminent_collision_score = (imminent_collision_time_s - 2)/2

    close_asteroid_threshold = 200
    # Special case for if we're in survival mode, no mine, no lives to spare, and we want to clear out small asteroids around us to make a bubble so we have more evasion options
    if next_imminent_collision_time_stationary_s < 5 and ship_state['mines_remaining'] == 0 and (close_proximity_asteroid_count > 6 or len(game_state['asteroids']) > 50) and asteroid_dist_during_interception < close_asteroid_threshold:
        if a['size'] == 1:
            high_density_survival_score = -7
        elif a['size'] == 2:
            high_density_survival_score = 1
        elif a['size'] == 3:
            high_density_survival_score = 2
        elif a['size'] == 4:
            high_density_survival_score = 3
    else:
        high_density_survival_score = 0
    
    if a['size'] == 1:
        # If the asteroid is small, I prefer to shoot small asteroids close to me to make a safe personal bubble around me
        if asteroid_dist_during_interception < close_asteroid_threshold:
            asteroid_distance_score = 0
        else:
            asteroid_distance_score = (asteroid_dist_during_interception - close_asteroid_threshold)/500
    else:
        # If this asteroid is close to me, shooting it could unleash debris in my face, so try not to shoot large asteroids which are super close
        if asteroid_dist_during_interception < close_asteroid_threshold:
            asteroid_distance_score = 2
        else:
            # Asteroid is far
            asteroid_distance_score = (asteroid_dist_during_interception - close_asteroid_threshold)/500

    if 'timesteps_until_appearance' in a:
        # We want to penalize shooting at asteroids that haven't spawned in yet, because the farther ahead they are, the more things that could change, and that future asteroid may get destroyed by something else or destroyed by my other bullets accidentally, so we'd be shooting at an asteroid that no longer exists
        target_future_score = a['timesteps_until_appearance']/10 + 0.5
    else:
        target_future_score = 0
    #print(f"Target priority scores. Aiming: {asteroid_aiming_score}, Mines: {asteroid_mine_score}, Imminent Collision: {imminent_collision_score}, Distance: {asteroid_distance_score}, Future: {target_future_score}, High density survival: {high_density_survival_score}")
    return asteroid_aiming_score + asteroid_mine_score + imminent_collision_score + asteroid_distance_score + target_future_score + high_density_survival_score

def forecast_asteroid_bullet_splits(a, timesteps_until_appearance, bullet_heading_deg, game_state=None, wrap=False):
    if a['size'] == 1:
        # Asteroids of size 1 don't split
        return []
    # Look at asteroid.py in the Kessler game's code
    bullet_vel_x = math.cos(math.radians(bullet_heading_deg))*bullet_speed
    bullet_vel_y = math.sin(math.radians(bullet_heading_deg))*bullet_speed
    vfx = (1/(bullet_mass + a['mass']))*(bullet_mass*bullet_vel_x + a['mass']*a['velocity'][0])
    vfy = (1/(bullet_mass + a['mass']))*(bullet_mass*bullet_vel_y + a['mass']*a['velocity'][1])
    return forecast_asteroid_splits(a, timesteps_until_appearance, vfx, vfy, game_state, wrap)

def forecast_asteroid_mine_splits(asteroid, timesteps_until_appearance, mine, game_state=None, wrap=False):
    if a['size'] == 1:
        # Asteroids of size 1 don't split
        return []
    dist_slow = np.sqrt((mine['position'][0] - asteroid['position'][0])**2 + (mine['position'][1] - asteroid['position'][1])**2)
    dist = math.dist(mine['position'], asteroid['position'])
    if enable_assertions:
        assert math.isclose(dist, dist_slow)
    F = (-dist/mine_blast_radius + 1)*mine_blast_pressure*2*mine_radius
    a = F/asteroid['mass']
    # calculate "impulse" based on acc
    if dist == 0:
        debug_print(f"Dist is 0! Kessler will spit out a runtime warning, and this asteroid will disappear without splitting.")
        return []
    vfx = asteroid['velocity'][0] + a*(asteroid['position'][0] - mine['position'][0])/dist
    vfy = asteroid['velocity'][1] + a*(asteroid['position'][1] - mine['position'][1])/dist
    return forecast_asteroid_splits(asteroid, timesteps_until_appearance, vfx, vfy, game_state, wrap)

def forecast_asteroid_splits(a, timesteps_until_appearance, vfx, vfy, game_state=None, wrap=False):
    # Calculate speed of resultant asteroid(s) based on velocity vector
    v = np.sqrt(vfx**2 + vfy**2)
    # Calculate angle of center asteroid for split (degrees)
    theta = math.degrees(np.arctan2(vfy, vfx))
    # Split angle is the angle off of the new velocity vector for the two asteroids to the sides, the center child
    # asteroid continues on the new velocity path
    split_angle = 15
    angles = [(theta + split_angle)%360, theta, (theta - split_angle)%360]
    # This is wacky because we're back-extrapolation the position of the asteroid BEFORE IT WAS BORN!!!!11!

    forecasted_asteroids = [{'position': ((a['position'][0] + a['velocity'][0]*delta_time*timesteps_until_appearance - timesteps_until_appearance*math.cos(math.radians(angle))*v*delta_time)%game_state['map_size'][0], (a['position'][1] + a['velocity'][1]*delta_time*timesteps_until_appearance - timesteps_until_appearance*math.sin(math.radians(angle))*v*delta_time)%game_state['map_size'][1]) if game_state and wrap else (a['position'][0] + a['velocity'][0]*delta_time*timesteps_until_appearance - timesteps_until_appearance*math.cos(math.radians(angle))*v*delta_time, a['position'][1] + a['velocity'][1]*delta_time*timesteps_until_appearance - timesteps_until_appearance*math.sin(math.radians(angle))*v*delta_time),
                             'velocity': (math.cos(math.radians(angle))*v, math.sin(math.radians(angle))*v),
                             'size': a['size'] - 1,
                             'mass': asteroid_mass_lookup[a['size'] - 1],
                             'radius': asteroid_radii_lookup[a['size'] - 1],
                             'timesteps_until_appearance': timesteps_until_appearance}
                               for angle in angles]
    return forecasted_asteroids

def maintain_forecasted_asteroids(forecasted_asteroid_splits, game_state=None, wrap=False):
    # Maintain the list of projected split asteroids by advancing the position, decreasing the timestep, and facilitate removal
    new_forecasted_asteroids = []
    for forecasted_asteroid in forecasted_asteroid_splits:
        if forecasted_asteroid['timesteps_until_appearance'] > 1:
            new_a = {
                'position': ((forecasted_asteroid['position'][0] + forecasted_asteroid['velocity'][0]*delta_time)%game_state['map_size'][0], (forecasted_asteroid['position'][1] + forecasted_asteroid['velocity'][1]*delta_time)%game_state['map_size'][1]) if wrap and game_state is not None else (forecasted_asteroid['position'][0] + forecasted_asteroid['velocity'][0]*delta_time, forecasted_asteroid['position'][1] + forecasted_asteroid['velocity'][1]*delta_time),
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
            print(f"INSIDE COMP FUNCTION, ASTEROID {ast_to_string(a)} IS CLOSE TO {ast_to_string(asteroid)}")
            return True
    return False

def count_asteroids_in_mine_blast_radius(game_state, asteroids_list, mine_x, mine_y, future_check_timesteps):
    count = 0
    for a in asteroids_list:
        # Extrapolate the asteroid position into the time of the mine detonation to check its bounds
        asteroid_future_x = a['position'][0] + future_check_timesteps*delta_time*a['velocity'][0]
        asteroid_future_y = a['position'][1] + future_check_timesteps*delta_time*a['velocity'][1]
        if check_coordinate_bounds(game_state, asteroid_future_x, asteroid_future_y):
            # Use the same collision prediction function as we use with the ship
            t1, t2 = collision_prediction(mine_x, mine_y, 0, 0, mine_blast_radius, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'])
            #print(t1, t2)
            # Assuming the two times exist, the first time is when the collision starts, and the second time is when the collision ends
            # All in between times is where the circles are inside of each other (intersects)
            # We want to check whether the mine's blast radius is intersecting with the asteroid at the future time
            if not math.isnan(t1) and not math.isnan(t2) and min(t1, t2) < future_check_timesteps*delta_time < max(t1, t2):
                # A collision exists, and it'll happen when the mine is detonating
                count += 1
    return count

def predict_ship_mine_collision(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, mine, future_timesteps=0):
    if mine['remaining_time'] >= future_timesteps*delta_time:
        # Project the ship to its future location when the mine is blowing up
        ship_pos_x += (mine['remaining_time'] - future_timesteps*delta_time)*ship_vel_x
        ship_pos_y += (mine['remaining_time'] - future_timesteps*delta_time)*ship_vel_y
        if check_collision(ship_pos_x, ship_pos_y, ship_radius, mine['position'][0], mine['position'][1], mine_blast_radius):
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
    return 1 + math.ceil((time_until_asteroid_center_s*bullet_speed - asteroid_radius - ship_radius)/bullet_speed/delta_time)

def asteroid_bullet_collision(bullet_head_position, bullet_heading_angle, asteroid_center, asteroid_radius):
    # This is an optimized version of circle_line_collision() from the Kessler source code
    bullet_tail = (bullet_head_position[0] - bullet_length*math.cos(math.radians(bullet_heading_angle)), bullet_head_position[1] - bullet_length*math.sin(math.radians(bullet_heading_angle)))
    # First, do a rough check if there's no chance the collision can occur
    # Avoid the use of min/max because it should be a bit faster
    if bullet_head_position[0] < bullet_tail[0]:
        x_min = bullet_head_position[0] - asteroid_radius
        x_max = bullet_tail[0] + asteroid_radius
    else:
        x_min = bullet_tail[0] - asteroid_radius
        x_max = bullet_head_position[0] + asteroid_radius

    if asteroid_center[0] < x_min or asteroid_center[0] > x_max:
        return False

    if bullet_head_position[1] < bullet_tail[1]:
        y_min = bullet_head_position[1] - asteroid_radius
        y_max = bullet_tail[1] + asteroid_radius
    else:
        y_min = bullet_tail[1] - asteroid_radius
        y_max = bullet_head_position[1] + asteroid_radius

    if asteroid_center[1] < y_min or asteroid_center[1] > y_max:
        return False

    # A collision is possible.
    # Create a triangle between the center of the asteroid, and the two ends of the bullet.
    a = math.dist(bullet_head_position, asteroid_center)
    b = math.dist(bullet_tail, asteroid_center)
    c = bullet_length

    # Heron's formula to calculate area of triangle and resultant height (distance from circle center to line segment)
    s = 0.5*(a + b + c)

    squared_area = s*(s - a)*(s - b)*(s - c)
    triangle_height = 2/c*math.sqrt(max(squared_area, 0))

    # If triangle's height is less than the asteroid's radius, the bullet is colliding with it
    return triangle_height < asteroid_radius

@lru_cache() # This function gets called with the same params all the time, so just cache the return value the first time
def get_simulated_ship_max_range(max_cruise_seconds):
    dummy_game_state = {}
    dummy_ship_state = {'speed': 0, 'position': (0, 0), 'velocity': (0, 0), 'heading': 0, 'bullets_remaining': 0, 'lives_remaining': 1}
    max_ship_range_test = Simulation(dummy_game_state, dummy_ship_state, 0)
    max_ship_range_test.accelerate(ship_max_speed)
    max_ship_range_test.cruise(round(max_cruise_seconds/delta_time))
    max_ship_range_test.accelerate(0)
    state_sequence = max_ship_range_test.get_state_sequence()
    #print(state_sequence[0])
    ship_random_range = math.dist(state_sequence[0]['position'], state_sequence[-1]['position'])
    ship_random_max_maneuver_length = len(state_sequence)
    return ship_random_range, ship_random_max_maneuver_length

def simulate_ship_movement_with_inputs(game_state, ship_state, move_sequence):
    dummy_game_state = {'asteroids': [], 'map_size': game_state['map_size']}
    ship_movement_sim = Simulation(dummy_game_state, ship_state, 0, [], {}, [], [], [], [], -math.inf, math.inf, False)
    ship_movement_sim.apply_move_sequence(move_sequence)
    return ship_movement_sim.get_ship_state()

def get_feasible_intercept_angle_and_turn_time(a, ship_state, game_state, timesteps_until_can_fire=0):
    def count_intersections(a, b):
        if len(a) != len(b):
            raise ValueError("Lists must be of the same length.")
        intersection_count = 0
        # Iterate through the lists, checking for changes in relative ordering
        for i in range(len(a) - 1):
            if (a[i] < b[i] and a[i + 1] > b[i + 1]) or (a[i] > b[i] and a[i + 1] < b[i + 1]):
                intersection_count += 1
        return intersection_count

    spam = False
    debug = False
    if spam: debug_print(f"\nGETTING FEASIBLE INTERCEPT FOR ASTEROID {ast_to_string(a)}, THIS IS A BIG HONKER OF A FUNCTION")
    # a could be a virtual asteroid, and we'll bounds check it for feasibility
    #timesteps_until_can_fire += 2
    # This function will check whether it's feasible to intercept an asteroid, whether real or virtual, within the bounds of the game
    # It will find exactly how long it'll take to turn toward the right heading before shooting, and consider turning both directions and taking the min turn amount/time
    # There are possible cases where you counterintuitively want to turn the opposite way of the asteroid, especially in crazy cases where the asteroid is moving faster than the bullet, so the ship doesn't chase down the asteroid and instead preempts it

    # Alright so what the heck is this convergence stuff all about?
    # So let's say at this timestep we calculate that if we want to shoot an asteroid, we have to shoot at X degrees from our heading.
    # But the thing is, we can only hit the asteroid if we were already looking there and we shoot at this exact instant.
    # By the time we can turn our ship to the correct heading, the future correct heading will have moved even farther! We have Zeno's paradox.
    # So what we need to find, is a future timestep at which between now and then, we have enough time to turn our ship, to achieve the future correct heading to shoot at the bullet
    # There may be a way to solve this algebraically, but since we're working with discrete numbers, we can hone in on this value within a few iterations.

    # The idea for this linear/binary search is that, for the linear search case:
    # First we see how far off the heading is. We're assuming 0 aiming timesteps required before shooting.
    # We get that, say the heading is off by an amount that requires turning for 10 timesteps.
    # We then check, if we shoot 10 timesteps in the future, how far off would the heading be from the current time?
    # We may get that the heading is off by an amount that requires 11 timesteps of turning.
    # Alright, if we shoot 11 timesteps in the future, how far off is the heading? Oh, it's an amount that's turnable in 11 timesteps, 11 == 11 so we're done iterating and we found our answer! We need to turn 11 timesteps, and the angle is the last thing we found.
    # TODO: CULLING TO ELIMINATE ONES WHERE IT ISNT FEASIBLE AND ITS CLEAR OFF THE BAT
    
    aiming_timesteps_required = None

    if debug:
        ts_data_array = []
        angle_err_data_array = []
        aiming_ts_array = []
        aiming_ts_rounded_array = []
    
    # TODO: USE BINARY SEARCH INSTEAD OF LINEAR SEARCH
    for turning_timesteps in range(0, 31):
        feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time_s, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception = calculate_interception(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'], ship_state['heading'], game_state, turning_timesteps)
        if spam: print(feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time_s, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception)
        if math.isnan(shot_heading_error_rad):
            # No intercept exists
            if debug:
                ts_data_array.append(turning_timesteps)
                angle_err_data_array.append(math.nan)
                aiming_ts_array.append(math.nan)
                aiming_ts_rounded_array.append(math.nan)
            continue
        #print(f"Feasible intercept ({intercept_x}, {intercept_y})")
        #print(f'Still not converged, extra timesteps: {aiming_timesteps_required}, shot_heading_error_rad: {shot_heading_error_rad}, tol: {shot_heading_tolerance_rad}')
        # For each asteroid, because the asteroid has a size, there is a range in angles in which we can shoot it.
        # We don't have to hit the very center, just close enough to it so that it's still the thick part of the circle and it won't skim the tangent
        # TODO: If we can aim at the center of the asteroid with no additional timesteps required, then just might as well do it cuz yeah
        #print(f"Shot heading error {shot_heading_error_rad}, shot heading tol: {shot_heading_tolerance_rad}, whether it's within: {abs(shot_heading_error_rad) <= shot_heading_tolerance_rad}")
        if abs(shot_heading_error_rad) <= shot_heading_tolerance_rad:
            shooting_angle_error_rad_using_tolerance = 0
        else:
            if shot_heading_error_rad > 0: # TODO: UNCOMMENT THE FOLLOWING AND USE THE TOLERANCE TO MAKE SHOOTING EASIER
                shooting_angle_error_rad_using_tolerance = shot_heading_error_rad# - shot_heading_tolerance_rad
            else:
                shooting_angle_error_rad_using_tolerance = shot_heading_error_rad# + shot_heading_tolerance_rad
        # shooting_angle_error is the amount we need to move our heading by, in radians
        # If there's some timesteps until we can fire, then we can get a head start on the aiming
        new_aiming_timesteps_required = math.ceil(math.degrees(abs(shooting_angle_error_rad_using_tolerance))/(ship_max_turn_rate*delta_time) - eps)
        new_aiming_timesteps_required_exact_aiming = math.ceil(math.degrees(abs(shot_heading_error_rad))/(ship_max_turn_rate*delta_time) - eps)
        if debug:
            ts_data_array.append(turning_timesteps)
            angle_err_data_array.append(shot_heading_error_rad)
            aiming_ts_array.append(math.degrees(abs(shot_heading_error_rad))/(ship_max_turn_rate*delta_time))
            aiming_ts_rounded_array.append(new_aiming_timesteps_required_exact_aiming)

        if new_aiming_timesteps_required == new_aiming_timesteps_required_exact_aiming:
            # If it doesn't save any time to not use exact aiming, just exactly aim at the center of the asteroid. It also prevents potential edge cases where it aims at the side of the asteroid but it doesn't shoot it (see XFC 2021 scenario_2_still_corridors scenario)
            shooting_angle_error_rad = shot_heading_error_rad
        else:
            shooting_angle_error_rad = shooting_angle_error_rad_using_tolerance
        if turning_timesteps >= new_aiming_timesteps_required and feasible and turning_timesteps >= timesteps_until_can_fire:
            if turning_timesteps == new_aiming_timesteps_required:
                if spam: debug_print(f"Converged. Aiming timesteps required is {turning_timesteps} which was the same as the previous iteration")
            elif turning_timesteps > new_aiming_timesteps_required:
                # Wacky oscillation case
                if spam: debug_print(f"WACKY OSCILLATION CASE WHERE aiming_timesteps_required ({turning_timesteps}) > new_aiming_timesteps_required ({new_aiming_timesteps_required})")
            # Found an answer
            if not aiming_timesteps_required:
                # Only set the first answer
                aiming_timesteps_required = turning_timesteps
                if not debug:
                    break
        else:
            if spam: debug_print(f"Not converged yet. aiming_timesteps_required ({turning_timesteps}) != new_aiming_timesteps_required ({new_aiming_timesteps_required})")
            continue
    
    if aiming_timesteps_required:
        feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time_s, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception = calculate_interception(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'], ship_state['heading'], game_state, aiming_timesteps_required)
    else:
        return False, None, None, None, None, None, None
    if abs(shot_heading_error_rad) <= shot_heading_tolerance_rad:
        shooting_angle_error_rad_using_tolerance = 0
    else:
        if shot_heading_error_rad > 0: # TODO: UNCOMMENT THE FOLLOWING AND USE THE TOLERANCE TO MAKE SHOOTING EASIER
            shooting_angle_error_rad_using_tolerance = shot_heading_error_rad# - shot_heading_tolerance_rad
        else:
            shooting_angle_error_rad_using_tolerance = shot_heading_error_rad# + shot_heading_tolerance_rad
    new_aiming_timesteps_required = math.ceil(math.degrees(abs(shooting_angle_error_rad_using_tolerance))/(ship_max_turn_rate*delta_time) - eps)
    new_aiming_timesteps_required_exact_aiming = math.ceil(math.degrees(abs(shot_heading_error_rad))/(ship_max_turn_rate*delta_time) - eps)
    
    if new_aiming_timesteps_required == new_aiming_timesteps_required_exact_aiming:
        # If it doesn't save any time to not use exact aiming, just exactly aim at the center of the asteroid. It also prevents potential edge cases where it aims at the side of the asteroid but it doesn't shoot it (see XFC 2021 scenario_2_still_corridors scenario)
        shooting_angle_error_rad = shot_heading_error_rad
    else:
        shooting_angle_error_rad = shooting_angle_error_rad_using_tolerance
    shooting_angle_error_deg = math.degrees(shooting_angle_error_rad)
    # TODO: OPPOSITNG CASE SHOOTING OTHER WAY IT'S MORE COMPLICATED TO CONSIDER, OR PROVE ITS IMPOSSIBLE
    # OH WAIT THIS CASE ISN'T REALLY POSSIBLE BECAUSE THE ASTEROID WOULD HAVE TO BE MOVING SUPER FAST. The highest I've seen the absolute shooting angle error is like 135 degrees, and that's an extreme synthetic scenario
    if enable_assertions:
        assert abs(shooting_angle_error_deg) <= 180
    if not feasible:
        return False, None, None, None, None, None, None
    #time.sleep(1)
    if spam: debug_print("Final answer is:")
    if spam: print(feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception)
    
    if debug and feasible and count_intersections(ts_data_array, aiming_ts_array) > 2:
        # Plot the answer
        plt.figure(figsize=(10, 5))
        # Plotting T against F(T)
        plt.plot(ts_data_array, aiming_ts_array, marker='o', label='TS Required (Fractional)')
        plt.plot(ts_data_array, aiming_ts_rounded_array, marker='o', label='TS Required (Rounded)')
        # Plotting T against T
        plt.plot(ts_data_array, ts_data_array, marker='x', color='red', label='Turning TS')

        # Adding titles and labels
        plt.title('Plot of Turning Timesteps Required for Interception')
        plt.xlabel('Timesteps')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()
    return feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception

def track_asteroid_we_shot_at(asteroids_pending_death, current_timestep, game_state, bullet_travel_timesteps, asteroid, wrap=True):
    asteroid = dict(asteroid)
    #bullet_travel_timesteps += 1 # TODO: Fudge? ahfsdufoisdfoi doesn't matter
    # Project the asteroid into the future, to where it would be on the timestep of its death
    
    for future_timesteps in range(0, bullet_travel_timesteps + 1):
        if wrap:
            # Wrap asteroid position to get the canonical asteroid
            asteroid['position'] = (asteroid['position'][0]%game_state['map_size'][0], asteroid['position'][1]%game_state['map_size'][1])
        #print(asteroid)
        timestep = current_timestep + future_timesteps
        if timestep not in asteroids_pending_death:
            asteroids_pending_death[timestep] = [dict(asteroid)]
        else:
            #debug_print(f"Future ts: {future_timesteps}")
            #if is_asteroid_in_list(asteroids_pending_death[timestep], asteroid):
                #debug_print(f"WARNING: ASTEREOID ALREWADY  IN LIST ")
            if enable_assertions:
               if is_asteroid_in_list(asteroids_pending_death[timestep], asteroid):
                   print(f'ABOUT TO FAIL ASSERTION, we are in the future by {future_timesteps} timesteps, LIST FOR THIS TS IS:')
                   print(asteroids_pending_death[timestep])
               assert not is_asteroid_in_list(asteroids_pending_death[timestep], asteroid)
            asteroids_pending_death[timestep].append(dict(asteroid))
        # Advance the asteroid to the next position
        # TODO: Remove this redundant operation on the last iteration
        asteroid['position'] = (asteroid['position'][0] + asteroid['velocity'][0]*delta_time, asteroid['position'][1] + asteroid['velocity'][1]*delta_time)
    return asteroids_pending_death

def check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(asteroids_pending_death, current_timestep, game_state, asteroid, wrap=False):
    # Check whether the asteroid has already been shot at, or if we can shoot at it again
    if wrap:
        asteroid = dict(asteroid)
        asteroid['position'] = (asteroid['position'][0]%game_state['map_size'][0], asteroid['position'][1]%game_state['map_size'][1])
    else:
        if enable_assertions:
            assert check_coordinate_bounds(game_state, asteroid['position'][0], asteroid['position'][1])
    if current_timestep not in asteroids_pending_death:
        print(f"This asteroid was NOT shot at and IS IN THE CLEAR! {ast_to_string(asteroid)}")
        return True
    else:
        if not is_asteroid_in_list(asteroids_pending_death[current_timestep], asteroid):
            print(f"This asteroid was NOT shot at and IS IN THE CLEAR! {ast_to_string(asteroid)}")
        return not is_asteroid_in_list(asteroids_pending_death[current_timestep], asteroid)

def extrapolate_asteroid_forward(asteroid, timesteps, game_state=None, wrap=False):
    asteroid = dict(asteroid) # TODO: Unsure whether this copy is necessary
    if wrap and game_state is not None:
        asteroid['position'] = ((asteroid['position'][0] + asteroid['velocity'][0]*timesteps*delta_time)%game_state['map_size'][0], (asteroid['position'][1] + asteroid['velocity'][1]*timesteps*delta_time)%game_state['map_size'][1])
    else:
        asteroid['position'] = (asteroid['position'][0] + asteroid['velocity'][0]*timesteps*delta_time, asteroid['position'][1] + asteroid['velocity'][1]*timesteps*delta_time)
    return asteroid

class Simulation():
    # Simulates kessler_game.py and ship.py and other game mechanics
    def __init__(self, game_state, ship_state, initial_timestep, asteroids=[], asteroids_pending_death={}, forecasted_asteroid_splits=[], mines=[], other_ships=[], bullets=[], last_timestep_fired=-math.inf, timesteps_to_not_check_collision_for=0, fire_first_timestep=False, game_state_plotter: GameStatePlotter | None = None):
        #debug_print("INITIALIZING SIMULATION, ship info is printed below:")
        #print(ship_state)
        self.speed = ship_state['speed']
        self.position = ship_state['position']
        self.velocity = ship_state['velocity']
        self.heading = ship_state['heading']
        self.initial_timestep = initial_timestep
        self.future_timesteps = 0
        self.last_timestep_fired = last_timestep_fired
        #debug_print('Heres the asteroid list when coming in:')
        #debug_print(asteroids)
        self.asteroids = [dict(a) for a in asteroids]
        self.mines = [dict(m) for m in mines]
        #print(other_ships)
        self.other_ships = [dict(s) for s in other_ships]
        self.bullets = [dict(b) for b in bullets]
        self.bullets_remaining = math.inf if ship_state['bullets_remaining'] == -1 else ship_state['bullets_remaining']
        self.game_state = game_state
        self.ship_move_sequence = []
        self.state_sequence = []
        self.asteroids_shot = 0
        self.asteroids_pending_death = asteroids_pending_death
        self.forecasted_asteroid_splits = forecasted_asteroid_splits
        self.timesteps_to_not_check_collision_for = timesteps_to_not_check_collision_for
        self.fire_next_timestep_flag = False
        self.fire_first_timestep = fire_first_timestep
        self.game_state_plotter = game_state_plotter

    def get_ship_state(self):
        return {'position': self.position, 'velocity': self.velocity, 'speed': self.speed, 'heading': self.heading, 'bullets_remaining': self.bullets_remaining}

    def get_fire_next_timestep_flag(self):
        return self.fire_next_timestep_flag

    #def get_shot_at_asteroids(self):
    #    return self.asteroids_pending_death
    def get_asteroids_pending_death(self):
        return self.asteroids_pending_death

    def get_forecasted_asteroid_splits(self):
        return self.forecasted_asteroid_splits

    def get_instantaneous_asteroid_collision(self):
        for a in self.asteroids:
            if check_collision(self.position[0], self.position[1], ship_radius, a['position'][0], a['position'][1], a['radius']):
                return True
        return False

    def get_instantaneous_ship_collision(self):
        ship_avoidance_padding
        for ship in self.other_ships:
            # The faster the other ship is going, the bigger of a bubble around it I'm going to draw, since they can deviate from their path very quickly and run into me even though I thought I was in the clear
            if check_collision(self.position[0], self.position[1], ship_radius, ship['position'][0], ship['position'][1], ship['radius'] + ship_avoidance_padding + math.sqrt(ship['velocity'][0]**2 + ship['velocity'][1]**2)*ship_avoidance_speed_padding_ratio):
                return True
        return False

    def get_instantaneous_mine_collision(self):
        # TODO: MIGHT HAVE OFF BY ONE ERROR WHERE THE MINE EXPLODES ONE FRAME BEFORE OR AFTER THE ONE I EXPECT, VALIDATE THE MINE EXPLOSION TIME LATER
        mine_collision = False
        mine_remove_idxs = []
        for i, m in enumerate(self.mines):
            if m['remaining_time'] < eps: # TODO: Off by 1 error?
                if check_collision(self.position[0], self.position[1], ship_radius, m['position'][0], m['position'][1], mine_blast_radius):
                    mine_collision = True
                mine_remove_idxs.append(i)
        self.mines = [mine for idx, mine in enumerate(self.mines) if idx not in mine_remove_idxs]
        return mine_collision

    def get_next_extrapolated_collision_time(self):
        # Assume constant velocity from here
        next_imminent_collision_time = math.inf
        #print('Extrapolating stuff at rest in end')
        # TODO: Unsure if we need the forecasted asteroids, but I figured it can't hurt
        for asteroid in (self.asteroids + self.forecasted_asteroid_splits):
            #debug_print("Checking collision with asteroid:")
            #debug_print(a)
            #debug_print(f"Future timesteps: {self.future_timesteps}, timesteps to not check collision for: {self.timesteps_to_not_check_collision_for}")
            unwrapped_asteroids = unwrap_asteroid(asteroid, self.game_state['map_size'][0], self.game_state['map_size'][1], 'surround', True)
            for a in unwrapped_asteroids:
                if self.future_timesteps >= self.timesteps_to_not_check_collision_for:
                    next_imminent_collision_time = min(next_imminent_collision_time, predict_next_imminent_collision_time_with_asteroid(self.position[0], self.position[1], self.velocity[0], self.velocity[1], ship_radius, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius']))
                else:
                    # The asteroids position was never updated, so we need to extrapolate it right now in one step
                    #debug_print("Extrapolated collision time: The asteroids position was never updated, so we need to extrapolate it right now in one step")
                    next_imminent_collision_time = min(next_imminent_collision_time, predict_next_imminent_collision_time_with_asteroid(self.position[0] + self.future_timesteps*delta_time*self.velocity[0], self.position[1] + self.future_timesteps*delta_time*self.velocity[1], self.velocity[0], self.velocity[1], ship_radius, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius']))
        for m in self.mines:
            next_imminent_collision_time = min(predict_ship_mine_collision(self.position[0], self.position[1], self.velocity[0], self.velocity[1], m, 0), next_imminent_collision_time)
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
        move_sequence_length_s = self.get_sequence_length()*delta_time
        safe_time_after_maneuver_s = self.get_next_extrapolated_collision_time()
        asteroids_shot = self.asteroids_shot

        if asteroids_shot == 0:
            asteroids_shot = 0.5
        time_per_asteroids_shot = move_sequence_length_s/asteroids_shot
        asteroids_score = time_per_asteroids_shot/2
        
        states = self.get_state_sequence()

        if len(states) >= 2:
            displacement = math.dist(states[0]['position'], states[-1]['position'])
        else:
            displacement = 0
        displacement_score = displacement/1000 # TODO: If wrapped, this score will be disporportionately big. Ideally account for that by unwrapping somehow

        if displacement < eps:
            # Stationary
            safe_time_score = max(0, min(5, 5 - 5/3*safe_time_after_maneuver_s))
        else:
            # Maneuvering
            safe_time_score = max(0, min(5, 5 - safe_time_after_maneuver_s))

        sequence_length_score = move_sequence_length_s/10
        debug_print(f"Safe time score: {safe_time_score}, asteroids score: {asteroids_score}, sequence length score: {sequence_length_score}, displacement score: {displacement_score}")
        return safe_time_score + asteroids_score + sequence_length_score + displacement_score

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

    def target_selection(self):
        print('\n\nGETTING INTO TARGET SELECTION')
        def simulate_shooting_at_target(target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps):
            # Just because we're lined up for a shot doesn't mean our shot will hit, unfortunately.
            # Bullets and asteroids travel in discrete timesteps, and it's possible for the bullet to miss the asteroid hitbox between timesteps, where the interception would have occurred on an intermediate timestep.
            # This is unavoidable, and we just have to choose targets that don't do this.
            # If the asteroids are moving slow enough, this should be rare, but especially if small asteroids are moving very quickly, this issue is common.
            # A simulation will easily show whether this will happen or not
            # TODO: CULLING
            aiming_move_sequence = self.get_rotate_heading_move_sequence(target_asteroid_shooting_angle_error_deg)
            if self.fire_first_timestep:
                timesteps_until_can_fire = max(0, 5 - len(aiming_move_sequence))
            else:
                timesteps_until_can_fire = max(0, 5 - (self.initial_timestep + self.future_timesteps + len(aiming_move_sequence) - self.last_timestep_fired))
            #if enable_assertions:
            #   assert timesteps_until_can_fire == 0
            #print(f'aiming move seq before append, and ts until can fire is {timesteps_until_can_fire}')
            #print(aiming_move_sequence)
            aiming_move_sequence.extend([{'thrust': 0, 'turn_rate': 0, 'fire': False} for _ in range(timesteps_until_can_fire)])
            #print('aiming move seq')
            #print(aiming_move_sequence)
            asteroid_advance_timesteps = len(aiming_move_sequence)
            debug_print(f"Asteroid advanced timesteps: {asteroid_advance_timesteps}")
            debug_print(f"Targetting turning timesteps: {target_asteroid_turning_timesteps}")
            if enable_assertions:
                assert asteroid_advance_timesteps <= target_asteroid_turning_timesteps
            if asteroid_advance_timesteps < target_asteroid_turning_timesteps:
                debug_print(f"asteroid_advance_timesteps {asteroid_advance_timesteps} < target_asteroid_turning_timesteps {target_asteroid_turning_timesteps}")
                #time.sleep(2)
                for _ in range(target_asteroid_turning_timesteps - asteroid_advance_timesteps):
                    debug_print('Waiting one more timestep!')
                    #print(f"APPENDING ANOTHER WAITING MOVE WHERE WE SHOULDNT FIRE")
                    aiming_move_sequence.append({'thrust': 0, 'turn_rate': 0, 'fire': False})
            target_asteroid = dict(target_asteroid)
            target_asteroid = extrapolate_asteroid_forward(target_asteroid, asteroid_advance_timesteps, self.game_state, True)
            #debug_print(f"We're targetting asteroid {ast_to_string(target_asteroid)}")
            #debug_print(f"Entering the bullet target sim, we're on timestep {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)}")
            #debug_print(self.asteroids)
            #print('CURRENT SHIP STATE:')
            #print(self.get_ship_state())
            #print('APPLYING AIMING MOVE SEQ')
            #print(aiming_move_sequence)
            # TODO: TBH we could just set the ship heading instead of simulating, because simulating tells us nothing new!
            ship_state_after_aiming = simulate_ship_movement_with_inputs(self.game_state, self.get_ship_state(), aiming_move_sequence)
            #ship_state_after_aiming = self.get_ship_state()
            #ship_state_after_aiming['heading'] = (ship_state_after_aiming['heading'] + target_asteroid_shooting_angle_error_deg)%360
            #print('SHIP STTAT AFTER AIMING')
            #print(ship_state_after_aiming)
            actual_asteroid_hit, timesteps_until_bullet_hit_asteroid = self.bullet_target_sim(ship_state_after_aiming, self.fire_first_timestep, len(aiming_move_sequence))
            return actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid

        aiming_underturn_allowance_deg = 10
        # First, find the most imminent asteroid
        
        target_asteroids_list = []
        dummy_ship_state = {'speed': 0, 'position': self.position, 'velocity': (0, 0), 'heading': self.heading, 'bullets_remaining': 0, 'lives_remaining': 1}
        if self.fire_first_timestep:
            timesteps_until_can_fire = 5
        else:
            timesteps_until_can_fire = max(0, 5 - (self.initial_timestep + self.future_timesteps - self.last_timestep_fired))
        debug_print(f"\nSimulation starting from timestep {self.initial_timestep + self.future_timesteps}, and we need to wait this many until we can fire: {timesteps_until_can_fire}")
        #debug_print('asteroids')
        #debug_print(self.asteroids)
        #debug_print('Forecasetd splits')
        #debug_print(self.forecasted_asteroid_splits)
        #debug_print('bullets')
        #debug_print(self.bullets)
        most_imminent_asteroid_exists = False
        for asteroid in self.asteroids + self.forecasted_asteroid_splits:
            if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, asteroid, True):
                print(f"\nOn TS {self.initial_timestep + self.future_timesteps} We do not have a pending shot for the asteroid {ast_to_string(asteroid)}")
                unwrapped_asteroids = unwrap_asteroid(asteroid, self.game_state['map_size'][0], self.game_state['map_size'][1], 'surround', True)
                # Iterate through all unwrapped asteroids to find which one of the unwraps is the best feasible target.
                # 99% of the time, only one of the unwraps will have a feasible target, but there's situations where we could either shoot the asteroid before it wraps, or wait for it to wrap and then shoot it.
                # In these cases, we need to pick whichever option is the fastest when factoring in turn time and waiting time.
                best_feasible_unwrapped_target = None
                for a in unwrapped_asteroids:
                    feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = get_feasible_intercept_angle_and_turn_time(a, dummy_ship_state, self.game_state, timesteps_until_can_fire)
                    if feasible:
                        if best_feasible_unwrapped_target is None or aiming_timesteps_required < best_feasible_unwrapped_target[2]:
                            best_feasible_unwrapped_target = (feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception)
                if best_feasible_unwrapped_target is not None:
                    feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = best_feasible_unwrapped_target
                    imminent_collision_time_s = math.inf
                    for a in unwrapped_asteroids:
                        imminent_collision_time_s = min(imminent_collision_time_s, predict_next_imminent_collision_time_with_asteroid(self.position[0], self.position[1], self.velocity[0], self.velocity[1], ship_radius, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'] + collision_check_pad))
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
                        most_imminent_asteroid_exists = True
        # Check whether we have enough time to aim at it and shoot it down
        # TODO: PROBLEM, what if the asteroid breaks into pieces and I need to shoot those down too? But I have plenty of time, and I still want the fitness function to be good in that case
        
        turn_angle_deg_until_can_fire = timesteps_until_can_fire*ship_max_turn_rate*delta_time # Can be up to 30 degrees
        #print(target_asteroids_list)
        # if there’s an imminent shot coming toward me, I will aim at the asteroid that gets me CLOSEST to the direction of the imminent shot.
        # So it only plans one shot at a time instead of a series of shots, and it’ll keep things simpler
        #debug_print(f"Least angular dist: {least_angular_distance_asteroid_shooting_angle_error_deg}")
        #debug_print(target_asteroids_list)
        actual_asteroid_hit = None
        aiming_move_sequence = []

        if most_imminent_asteroid_exists:
            # First try to shoot the most imminent asteroids, if they exist
            #debug_print('YAYAYAYAYAYAYYAYAYAYAYAYYAYAYAYAAYAYAY SHOOOOTING AT IMMIEMNT ASDFOIASDJFOIJWETIJISOERJFIOSDJFIOSJFIODSJFIOSDJFIOSDJFIOJFDSIOIJO')
            #debug_print(f"Shooting at most imminent asteroids. Most imminent collision time is {most_imminent_collision_time_s}s with turn angle error {most_imminent_asteroid_shooting_angle_error_deg}")
            #debug_print(most_imminent_asteroid)
            # Find the asteroid I can shoot at that gets me closest to the imminent shot, if I can't reach the imminent shot in time until I can shoot
            #print('\n\ntarget asteroids list')
            #print(target_asteroids_list)
            sorted_imminent_targets = sorted(target_asteroids_list, key=lambda a: a['imminent_collision_time_s'])
            #print('\nsorted imminent targets')
            #print(sorted_imminent_targets)
            # TODO: For each asteroid, give it a couple feasible times where we wait longer and longer. This way we can choose to wait a timestep to fire again if we'll get luckier with the bullet lining up
            most_imminent_asteroid_aiming_timesteps = sorted_imminent_targets[0]['aiming_timesteps_required']
            most_imminent_asteroid = sorted_imminent_targets[0]['asteroid']
            most_imminent_asteroid_shooting_angle_error_deg = sorted_imminent_targets[0]['shooting_angle_error_deg']
            most_imminent_asteroid_interception_time_s = sorted_imminent_targets[0]['interception_time_s']
            if most_imminent_asteroid_aiming_timesteps <= timesteps_until_can_fire:
                if enable_assertions:
                    assert most_imminent_asteroid_aiming_timesteps == timesteps_until_can_fire
                # I can reach the imminent shot without wasting a shot opportunity, so do it
                actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid = simulate_shooting_at_target(most_imminent_asteroid, most_imminent_asteroid_shooting_angle_error_deg, most_imminent_asteroid_interception_time_s, most_imminent_asteroid_aiming_timesteps)
                if not actual_asteroid_hit or not check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, actual_asteroid_hit, True):
                    print(f"DANG IT, the most imminent shot exists but we'll miss if we shoot now!")
            else:
                # Between now and when I can shoot, I don't have enough time to aim at the imminent asteroid.
                # Instead, find the closest asteroid along the way to shoot
                sorted_targets = sorted(target_asteroids_list, key=lambda a: a['shooting_angle_error_deg'])
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
                    actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid = simulate_shooting_at_target(target['asteroid'], target['shooting_angle_error_deg'], target['interception_time_s'], target['aiming_timesteps_required'])
                    if not actual_asteroid_hit or not check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, actual_asteroid_hit, True):
                        print(f"DANG IT, we're shooting something on the way to the most imminent asteroid, but we'll miss this particular one!")
                else:
                    # Just gonna have to waste shot opportunities and turn all the way
                    actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid = simulate_shooting_at_target(most_imminent_asteroid, most_imminent_asteroid_shooting_angle_error_deg, most_imminent_asteroid_interception_time_s, most_imminent_asteroid_aiming_timesteps)
                    if not actual_asteroid_hit or not check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, actual_asteroid_hit, True):
                        print(f"DANG IT, we're forced to turn to the most imminent shot, but we'll still miss if we do!")
        elif target_asteroids_list:
            debug_print("Shooting at asteroid with least shot delay since there aren't imminent asteroids")
            # TODO: Can sort tiebreakers like key=lambda a: (a['aiming_timesteps_required'], a['size'], a['distance']))
            # Once I get more prioritization, I can make use of this choice to prioritize small asteroids, or ones that are far from me, or whatever!
            #print('BEFOER AND AFTER SORTING')
            #print(target_asteroids_list)
            sorted_targets = sorted(target_asteroids_list, key=lambda a: a['aiming_timesteps_required'])
            #print(sorted_targets)
            for target in sorted_targets:
                least_shot_delay_asteroid = target['asteroid']
                least_shot_delay_asteroid_shooting_angle_error_deg = target['shooting_angle_error_deg']
                least_shot_delay_asteroid_interception_time_s = target['interception_time_s']
                least_shot_delay_asteroid_aiming_timesteps = target['aiming_timesteps_required']
                #print(sorted_targets)
                #time.sleep(5)
                actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid = simulate_shooting_at_target(least_shot_delay_asteroid, least_shot_delay_asteroid_shooting_angle_error_deg, least_shot_delay_asteroid_interception_time_s, least_shot_delay_asteroid_aiming_timesteps)
                if actual_asteroid_hit is not None and check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, actual_asteroid_hit, True):
                    break
            if not actual_asteroid_hit:
                print(f"DANG IT, we went through all asteroids with least shot delay and we'll miss them ALL!")
        else:
            # Ain't nothing to shoot at
            # TODO: IF THERE'S NOTHING TO SHOOT AT, TURN TOWARD THE CENTER! THIS WAY MAYBE there's a super fast one that I just keep not getting.
            print('Nothing to shoot at!')
            # We still simulate one iteration of this, because if we had a pending shot from before, this will do the shot!
            self.update(0, 0, False)
            #print("NOT SHOOTING AT ANYTHING, THE MOVE SEQUENCE IS BELOW AND HOPEFULLY WE DONT SHOOT ON THIS TIMESTEP")
            #print(self.ship_move_sequence)
            return

        #debug_print('Closest ang asteroid:')
        #debug_print(target_asteroids_list[least_angular_distance_asteroid_index])
        #debug_print('Second closest ang asteroid:')
        #debug_print(target_asteroids_list[second_least_angular_distance_asteroid_index])

        #print(f"Bullet should have been fired on simulated timestep {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)}")
        if actual_asteroid_hit is None or not check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, actual_asteroid_hit, True):
            self.fire_next_timestep_flag = False
            #print("Dang it, the sim showed that we won't actually hit anything :(")
            self.update(0, 0, False)
        else:
            #actual_asteroid_hit_UNEXTRAPOLATED = dict(actual_asteroid_hit)
            actual_asteroid_hit_on_collision = dict(actual_asteroid_hit)
            actual_asteroid_hit = extrapolate_asteroid_forward(actual_asteroid_hit, len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid, self.game_state, True)
            debug_print(f"We used a sim to forecast the asteroid that we'll hit being the one at pos ({actual_asteroid_hit['position'][0]}, {actual_asteroid_hit['position'][1]}) with vel ({actual_asteroid_hit['velocity'][0]}, {actual_asteroid_hit['velocity'][1]}) in {timesteps_until_bullet_hit_asteroid - len(aiming_move_sequence)} timesteps")
            debug_print(f"The primitive forecast would have said we'd hit asteroid at pos ({target_asteroid['position'][0]}, {target_asteroid['position'][1]}) with vel ({target_asteroid['velocity'][0]}, {target_asteroid['velocity'][1]}) in {calculate_timesteps_until_bullet_hits_asteroid(target_asteroid_interception_time_s, target_asteroid['radius'])} timesteps")
            #print(self.asteroids_pending_death)
            print(f"\nTracking that we just shot at the asteroid {ast_to_string(actual_asteroid_hit)}, our intended target was {target_asteroid}")
            #actual_asteroid_hit_UNEXTRAPOLATED = extrapolate_asteroid_forward(actual_asteroid_hit, -(len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid), self.game_state, True)
            actual_asteroid_hit_at_present_time = extrapolate_asteroid_forward(actual_asteroid_hit, -len(aiming_move_sequence), self.game_state, True)
            if gamestate_plotting:
                self.game_state_plotter.update_plot([], [], [], [actual_asteroid_hit_at_present_time], [], [], False, 0.5, 'FEASIBLE TARGETS') #[dict(a['asteroid']) for a in sorted_targets]
            #actual_asteroid_hit_tracking_purposes_super_early = extrapolate_asteroid_forward(actual_asteroid_hit, )
            self.asteroids_pending_death = track_asteroid_we_shot_at(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + 0*len(aiming_move_sequence), self.game_state, timesteps_until_bullet_hit_asteroid - 0*len(aiming_move_sequence), actual_asteroid_hit_at_present_time)
            # Forecasted splits get progressed while doing the move sequence which includes rotation, so we need to start the forecast before the rotation even starts
            self.forecasted_asteroid_splits.extend(forecast_asteroid_bullet_splits(actual_asteroid_hit_at_present_time, timesteps_until_bullet_hit_asteroid, self.heading, self.game_state, True))
            forecast_backup = copy.deepcopy(self.forecasted_asteroid_splits)
            #print('extending  list with')
            #print(forecast_asteroid_bullet_splits(actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, self.heading, self.game_state, True))
            
            #if enable_assertions:
            #   assert(is_asteroid_in_list(self.asteroids, actual_asteroid_hit))
            #debug_print(f"Length of asteroids list before removing: {len(self.asteroids)}")
            # To make it so that checking for the next imminent collision no longer considers the asteroid we just shot at, we'll remove it from the list of asteroids as if the ship instantly shot it down
            self.asteroids = [a for a in self.asteroids if not is_asteroid_in_list([actual_asteroid_hit], a)]
            #debug_print(f"Length of asteroids list after removing: {len(self.asteroids)}")
            #if locked_in:
            #debug_print('Were gonna fire the next timestep! Printing out the move sequence:')
            
            self.asteroids_shot += 1
            #self.ship_move_sequence.append(aiming_move_sequence) # Bit of a hack...
            #print(aiming_move_sequence)
            self.apply_move_sequence(aiming_move_sequence)
            self.forecasted_asteroid_splits = forecast_backup
            self.fire_next_timestep_flag = True
            #print('THE ACTUAL MOVE SEQUENCE WE GET BACK FROM THE SIM:')
            #print(self.ship_move_sequence)

    def bullet_target_sim(self, ship_state=None, fire_first_timestep=False, fire_after_timesteps=0):
        # Assume we shoot on the next timestep, so we'll create a bullet and then track it and simulate it to see what it hits, if anything
        # This sim doesn't modify the state of the simulation class. Everything here is discarded after the sim is over, and this is just to see what my bullet hits, if anything.
        asteroids = [dict(a) for a in self.asteroids] # TODO: CULL ASTEROIDS? But then we won't be considering other bullets so maybe this is a bad idea. Maybe cull for all bullets and mines combined?
        mines = [dict(m) for m in self.mines]
        bullets = [dict(b) for b in self.bullets]
        initial_ship_state = self.get_ship_state()
        #debug_print(f"Beginning sim, and here's the midair bullets we have before the sim started:")
        #debug_print(bullets)
        bullet_created = False
        initial_timestep_fire_bullet = None
        timesteps_until_bullet_hit_asteroid = 0
        # Keep iterating until our bullet flies off the edge of the screen, or it hits an asteroid
        while True:
            # Simplified simulation loop
            timesteps_until_bullet_hit_asteroid += 1
            #print(f"BULLET SIM STATE DUMP FOR FUTURE TIMESTEPS {timesteps_until_bullet_hit_asteroid}")
            #print("BULLETS")
            #print(bullets + (bullet_created)*[my_bullet])
            #print("ASTEOIRDS")
            #print(asteroids)
            #if gamestate_plotting:
            #   self.game_state_plotter.update_plot(asteroids, None, bullets + (bullet_created)*[my_bullet], [], [], [], True, eps, 'BULLET SIMULATION')

            # Simulate bullets
            bullet_remove_idxs = []
            for b_ind, b in enumerate(bullets):
                new_bullet_pos = (b['position'][0] + delta_time*b['velocity'][0], b['position'][1] + delta_time*b['velocity'][1])
                if check_coordinate_bounds(self.game_state, new_bullet_pos[0], new_bullet_pos[1]):
                    b['position'] = new_bullet_pos
                else:
                    bullet_remove_idxs.append(b_ind)
            bullets = [bullet for idx, bullet in enumerate(bullets) if idx not in bullet_remove_idxs]
            if bullet_created:
                my_new_bullet_pos = (initial_timestep_fire_bullet['position'][0] + delta_time*initial_timestep_fire_bullet['velocity'][0], initial_timestep_fire_bullet['position'][1] + delta_time*initial_timestep_fire_bullet['velocity'][1])
                if check_coordinate_bounds(self.game_state, my_new_bullet_pos[0], my_new_bullet_pos[1]):
                    initial_timestep_fire_bullet['position'] = my_new_bullet_pos
                else:
                    return None, None # The bullet got shot into the void without hitting anything :(
            
            for m in mines:
                m['fuse_time'] -= delta_time
            for a in asteroids:
                a['position'] = ((a['position'][0] + delta_time*a['velocity'][0])%self.game_state['map_size'][0], (a['position'][1] + delta_time*a['velocity'][1])%self.game_state['map_size'][1])
            #debug_print(f"TS ahead of sim end: {timesteps_until_bullet_hit_asteroid}")
            #debug_print(asteroids)
            # Create the initial bullet we fire, if we're locked in
            if fire_first_timestep and timesteps_until_bullet_hit_asteroid == 1:
                # TODO: LRU CACHE THE SIN AND COS FUNCTIONS OR JUST MODIFY THE BULLET DICT TO INCLUDE VELS AND PROBABLY ALSO THE TAIL
                rad_heading = math.radians(initial_ship_state['heading'])
                cos_heading = math.cos(rad_heading)
                sin_heading = math.sin(rad_heading)
                bullet_x = initial_ship_state['position'][0] + ship_radius*cos_heading
                bullet_y = initial_ship_state['position'][1] + ship_radius*sin_heading
                # Make sure the bullet isn't being fired out into the void
                if check_coordinate_bounds(self.game_state, bullet_x, bullet_y):
                    vx = bullet_speed*cos_heading
                    vy = bullet_speed*sin_heading
                    initial_timestep_fire_bullet = {
                        'position': (bullet_x, bullet_y),
                        'velocity': (vx, vy),
                        'heading': initial_ship_state['heading'],
                        'mass': bullet_mass,
                    }
                    bullets.append(initial_timestep_fire_bullet)
            # The new bullet we create will theoretically end up at the end of the list of bullets
            if not bullet_created and timesteps_until_bullet_hit_asteroid == fire_after_timesteps + 1:
                if ship_state is not None:
                    bullet_fired_from_ship_heading = ship_state['heading']
                    bullet_fired_from_ship_position = ship_state['position']
                else:
                    bullet_fired_from_ship_heading = self.heading
                    bullet_fired_from_ship_position = self.position
                # TODO: LRU CACHE THE SIN AND COS FUNCTIONS OR JUST MODIFY THE BULLET DICT TO INCLUDE VELS AND PROBABLY ALSO THE TAIL
                rad_heading = math.radians(bullet_fired_from_ship_heading)
                cos_heading = math.cos(rad_heading)
                sin_heading = math.sin(rad_heading)
                bullet_x = bullet_fired_from_ship_position[0] + ship_radius*cos_heading
                bullet_y = bullet_fired_from_ship_position[1] + ship_radius*sin_heading
                # Make sure the bullet isn't being fired out into the void
                if not check_coordinate_bounds(self.game_state, bullet_x, bullet_y):
                    return None, None # The bullet got shot into the void without hitting anything :(
                vx = bullet_speed*cos_heading
                vy = bullet_speed*sin_heading
                initial_timestep_fire_bullet = {
                    'position': (bullet_x, bullet_y),
                    'velocity': (vx, vy),
                    'heading': bullet_fired_from_ship_heading,
                    'mass': bullet_mass,
                }
                bullet_created = True
                #print(f"We created our own simulated bullet, we're on timestep actually idk and idc, and the bullet is:")
                #print(my_bullet)
                #print(f"Also on this timestep, the asteroids are")
                #print(asteroids)
            #debug_print(f"My sim bullet is at on timestep {timesteps_until_bullet_hit_asteroid}:")
            #debug_print(my_bullet)
            # Skip simulating ship

            # Check bullet/asteroid collisions
            bullet_remove_idxs = []
            asteroid_remove_idxs = []
            for idx_bul, b in enumerate(bullets + (bullet_created)*[initial_timestep_fire_bullet]):
                for idx_ast, a in enumerate(asteroids):
                    # If collision occurs
                    if asteroid_bullet_collision(b['position'], b['heading'], a['position'], a['radius']):
                        if idx_bul == len(bullets):
                            # This bullet is my bullet!
                            return a, timesteps_until_bullet_hit_asteroid
                        else:
                            # Mark bullet for removal
                            bullet_remove_idxs.append(idx_bul)
                            # Create asteroid splits and mark it for removal
                            asteroids.extend(forecast_asteroid_bullet_splits(a, 0, b['heading']))
                            asteroid_remove_idxs.append(idx_ast)
                            # Stop checking this bullet
                            break
            # Remove bullets and asteroids
            bullets = [bullet for idx, bullet in enumerate(bullets) if idx not in bullet_remove_idxs]
            asteroids = [asteroid for idx, asteroid in enumerate(asteroids) if idx not in asteroid_remove_idxs]

            # Check mine/asteroid collisions
            mine_remove_idxs = []
            asteroid_remove_idxs = []
            new_asteroids = []
            for idx_mine, mine in enumerate(mines):
                if mine['remaining_time'] < eps:
                    # Mine is detonating
                    mine_remove_idxs.append(idx_mine)
                    for idx_ast, asteroid in enumerate(asteroids):
                        if check_collision(asteroid['position'][0], asteroid['position'][1], asteroid['radius'], mine['position'][0], mine['position'][1], mine_blast_radius):
                            new_asteroids.extend(forecast_asteroid_mine_splits(a, 0, mine, self.game_state, True))
                            asteroid_remove_idxs.append(idx_ast)
            mines = [mine for idx, mine in enumerate(mines) if idx not in mine_remove_idxs]
            asteroids = [asteroid for idx, asteroid in enumerate(asteroids) if idx not in asteroid_remove_idxs]
            asteroids.extend(new_asteroids)

    def apply_move_sequence(self, move_sequence=[]):
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
            self.update(thrust, turn_rate, fire)

    def update(self, thrust=0, turn_rate=0, fire=None):
        # This is a highly optimized simulation of what kessler_game.py does, and should match exactly its behavior
        # Being even one timestep off is the difference between life and death!!!
        self.state_sequence.append({'timestep': self.initial_timestep + self.future_timesteps, 'position': self.position, 'velocity': self.velocity, 'speed': self.speed, 'heading': self.heading, 'asteroids': [dict(a) for a in self.asteroids], 'bullets': [dict(b) for b in self.bullets], 'asteroids_pending_death': dict(self.asteroids_pending_death), 'forecasted_asteroid_splits': [dict(a) for a in self.forecasted_asteroid_splits]})
        #print(f"Current SIM STATE DUMP FOR TIMESTEP {self.initial_timestep + self.future_timesteps}")
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
        for b_ind, b in enumerate(self.bullets):
            new_bullet_pos = (b['position'][0] + delta_time*b['velocity'][0], b['position'][1] + delta_time*b['velocity'][1])
            if check_coordinate_bounds(self.game_state, new_bullet_pos[0], new_bullet_pos[1]):
                b['position'] = new_bullet_pos
            else:
                bullet_remove_idxs.append(b_ind)
        self.bullets = [bullet for idx, bullet in enumerate(self.bullets) if idx not in bullet_remove_idxs]

        # Update mines
        for m in self.mines:
            m['fuse_time'] -= delta_time
            # If the timer is below eps, it'll detonate this timestep

        # Simulate dynamics of asteroids
        # Wrap the asteroid positions in the same operation
        if self.future_timesteps >= self.timesteps_to_not_check_collision_for:
            for a in self.asteroids:
                a['position'] = ((a['position'][0] + delta_time*a['velocity'][0])%self.game_state['map_size'][0], (a['position'][1] + delta_time*a['velocity'][1])%self.game_state['map_size'][1])

        # Simulate the ship!
        # Bullet firing happens before we turn the ship
        # Check whether we want to shoot a simulated bullet
        print(f"SIMULATION TS {self.future_timesteps}")
        if self.fire_first_timestep and self.future_timesteps == 0:
            print('FIRE FIRST TIMETSEP IS TRUE SO WERE FIRING')
            fire_this_timestep = True
        elif fire is None:
            timesteps_until_can_fire = max(0, 5 - (self.initial_timestep + self.future_timesteps - self.last_timestep_fired))
            fire_this_timestep = False
            if timesteps_until_can_fire == 0 and self.bullets_remaining > 0 and self.future_timesteps >= self.timesteps_to_not_check_collision_for:
                for asteroid in self.asteroids:
                    if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, asteroid, True):
                        unwrapped_asteroids = unwrap_asteroid(asteroid, self.game_state['map_size'][0], self.game_state['map_size'][1], 'half_surround', True)
                        for a in unwrapped_asteroids:
                            feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception = calculate_interception(self.position[0], self.position[1], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'], self.heading, self.game_state)
                            if feasible and abs(shot_heading_error_rad) <= shot_heading_tolerance_rad:
                                fire_this_timestep = True
                                self.asteroids_shot += 1
                                print(f"Tracking that we shot at the asteroid {ast_to_string(asteroid)}")
                                print(self.asteroids_pending_death)
                                self.asteroids_pending_death = track_asteroid_we_shot_at(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, calculate_timesteps_until_bullet_hits_asteroid(interception_time, asteroid['radius']), asteroid)
                                break
        else:
            fire_this_timestep = fire
        
        if fire_this_timestep:
            self.last_timestep_fired = self.initial_timestep + self.future_timesteps
            # Create new bullets/mines
            self.bullets_remaining -= 1
            rad_heading = math.radians(self.heading)
            cos_heading = math.cos(rad_heading)
            sin_heading = math.sin(rad_heading)
            bullet_x = self.position[0] + ship_radius*cos_heading
            bullet_y = self.position[1] + ship_radius*sin_heading
            # Make sure the bullet isn't being fired out into the void
            if check_coordinate_bounds(self.game_state, bullet_x, bullet_y):
                vx = bullet_speed*cos_heading
                vy = bullet_speed*sin_heading
                # TODO: Add component vels and dx dy for the tail, for efficient computation
                new_bullet = {
                    'position': (bullet_x, bullet_y),
                    'velocity': (vx, vy),
                    'heading': self.heading,
                    'mass': bullet_mass,
                }
                self.bullets.append(new_bullet)

        # TODO: Let the ship drop mines

        # Simulate ship dynamics
        drag_amount = ship_drag*delta_time
        if drag_amount > abs(self.speed):
            self.speed = 0
        else:
            self.speed -= drag_amount*np.sign(self.speed)
        # Bounds check the thrust
        thrust = min(max(-ship_max_thrust, thrust), ship_max_thrust)
        # Apply thrust
        self.speed += thrust*delta_time
        # Bounds check the speed
        self.speed = min(max(-ship_max_speed, self.speed), ship_max_speed)
        # Bounds check the turn rate
        turn_rate = min(max(-ship_max_turn_rate, turn_rate), ship_max_turn_rate)
        # Update the angle based on turning rate
        self.heading += turn_rate*delta_time
        # Keep the angle within (0, 360)
        self.heading %= 360
        # Use speed magnitude to get velocity vector
        self.velocity = (math.cos(math.radians(self.heading))*self.speed, math.sin(math.radians(self.heading))*self.speed)
        # Update the position based off the velocities
        # Do the wrap in the same operation
        self.position = ((self.position[0] + self.velocity[0]*delta_time)%self.game_state['map_size'][0], (self.position[1] + self.velocity[1]*delta_time)%self.game_state['map_size'][1])
        

        # Check bullet/asteroid collisions
        bullet_remove_idxs = []
        asteroid_remove_idxs = []
        for idx_bul, b in enumerate(self.bullets):
            for idx_ast, a in enumerate(self.asteroids):
                # If collision occurs
                if asteroid_bullet_collision(b['position'], b['heading'], a['position'], a['radius']):
                    # Mark bullet for removal
                    bullet_remove_idxs.append(idx_bul)
                    # Create asteroid splits and mark it for removal
                    self.asteroids.extend(forecast_asteroid_bullet_splits(a, 0, b['heading']))
                    asteroid_remove_idxs.append(idx_ast)
                    # Stop checking this bullet
                    break
        # Cull bullets and asteroids that are marked for removal
        self.bullets = [bullet for idx, bullet in enumerate(self.bullets) if idx not in bullet_remove_idxs]
        self.asteroids = [asteroid for idx, asteroid in enumerate(self.asteroids) if idx not in asteroid_remove_idxs]

        self.ship_move_sequence.append({'timestep': self.initial_timestep + self.future_timesteps, 'thrust': thrust, 'turn_rate': turn_rate, 'fire': fire_this_timestep})

        # Check mine/asteroid and mine/ship collisions
        mine_remove_idxs = []
        asteroid_remove_idxs = []
        new_asteroids = []
        for idx_mine, mine in enumerate(self.mines):
            if mine['remaining_time'] < eps:
                # Mine is detonating
                mine_remove_idxs.append(idx_mine)
                for idx_ast, asteroid in enumerate(self.asteroids):
                    if check_collision(asteroid['position'][0], asteroid['position'][1], asteroid['radius'], mine['position'][0], mine['position'][1], mine_blast_radius):
                        new_asteroids.extend(forecast_asteroid_mine_splits(a, 0, mine, self.game_state, True))
                        asteroid_remove_idxs.append(idx_ast)
                if check_collision(self.position[0], self.position[1], ship_radius, mine['position'][0], mine['position'][1], mine_blast_radius):
                    # Ship got hit by mine, RIP
                    # Even if the player is invincible, we still need to check for these collisions since invincibility is only for asteroids and ship-ship collisions
                    return False
        self.mines = [mine for idx, mine in enumerate(self.mines) if idx not in mine_remove_idxs]
        self.asteroids = [asteroid for idx, asteroid in enumerate(self.asteroids) if idx not in asteroid_remove_idxs]
        self.asteroids.extend(new_asteroids)

        # Check asteroid/ship and ship/ship collisions
        if self.future_timesteps >= self.timesteps_to_not_check_collision_for:
            if self.get_instantaneous_asteroid_collision() or self.get_instantaneous_ship_collision():
                return False

        self.future_timesteps += 1

        self.forecasted_asteroid_splits = maintain_forecasted_asteroids(self.forecasted_asteroid_splits, self.game_state, True)
        return True

    def rotate_heading(self, heading_difference_deg, shoot_on_first_timestep=False):
        target_heading = (self.heading + heading_difference_deg)%360
        still_need_to_turn = heading_difference_deg
        while abs(still_need_to_turn) > ship_max_turn_rate*delta_time + eps:
            if not self.update(0, ship_max_turn_rate*np.sign(heading_difference_deg), shoot_on_first_timestep):
                return False
            shoot_on_first_timestep = False
            still_need_to_turn -= ship_max_turn_rate*np.sign(heading_difference_deg)*delta_time
        if not self.update(0, still_need_to_turn/delta_time, shoot_on_first_timestep):
            return False
        # TODO: FIGURE OUT WHY THIS FAILS FOR CLOSING RING SCENARIO
        #if enable_assertions:
        #   assert(abs(target_heading - self.heading) < eps)
        return True

    def get_rotate_heading_move_sequence(self, heading_difference_deg, shoot_on_first_timestep=False):
        move_sequence = []
        still_need_to_turn = heading_difference_deg
        while abs(still_need_to_turn) > ship_max_turn_rate*delta_time + eps:
            move_sequence.append({'thrust': 0, 'turn_rate': ship_max_turn_rate*np.sign(heading_difference_deg), 'fire': shoot_on_first_timestep})
            shoot_on_first_timestep = False
            still_need_to_turn -= ship_max_turn_rate*np.sign(heading_difference_deg)*delta_time
        move_sequence.append({'thrust': 0, 'turn_rate': still_need_to_turn/delta_time, 'fire': shoot_on_first_timestep})
        return move_sequence

    def accelerate(self, target_speed, turn_rate=0):
        # Keep in mind speed can be negative
        # Drag will always slow down the ship
        while abs(self.speed - target_speed) > eps:
            drag = -ship_drag*np.sign(self.speed)
            drag_amount = ship_drag*delta_time
            if drag_amount > abs(self.speed):
                # The drag amount is reduced if it would make the ship cross 0 speed on its own
                adjust_drag_by = abs((drag_amount - abs(self.speed))/delta_time)
                drag -= adjust_drag_by*np.sign(drag)
            delta_speed_to_target = target_speed - self.speed
            thrust_amount = delta_speed_to_target/delta_time - drag
            #print(thrust_amount, self.speed, target_speed, delta_speed_to_target)
            thrust_amount = min(max(-ship_max_thrust, thrust_amount), ship_max_thrust)
            if not self.update(thrust_amount, turn_rate):
                return False
        return True

    def cruise(self, cruise_time, cruise_turn_rate=0):
        # Maintain current speed
        for _ in range(cruise_time):
            if not self.update(np.sign(self.speed)*ship_drag, cruise_turn_rate):
                return False
        return True

    def get_move_sequence(self):
        return self.ship_move_sequence
    
    #def get_move_sequence_length(self):
    #    return len(self.ship_move_sequence)

    def get_state_sequence(self):
        return self.state_sequence

    def get_sequence_length(self):
        #debug_print(f"Length of move seq: {len(self.ship_move_sequence)}, length of state seq: {len(self.state_sequence)}")
        if enable_assertions:
            assert len(self.ship_move_sequence) == len(self.state_sequence)
        return len(self.state_sequence)

    def get_future_timesteps(self):
        return self.future_timesteps
    
    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_heading(self):
        return self.heading

class Neo(KesslerController):
    @property
    def name(self) -> str:
        return "Neo"

    def __init__(self):
        self.init_done = False
        self.ship_id = None
        self.current_timestep = 0
        self.previously_targetted_asteroid = None
        self.last_timestep_fired = -math.inf
        self.action_queue = []  # This will become our heap
        self.asteroids_pending_death = {} # Keys are timesteps, and the values are the asteroids that still have midair bullets travelling toward them, so we don't want to shoot at them again
        self.forecasted_asteroid_splits = [] # List of asteroids that are forecasted to appear, but aren't guaranteed to
        os.system('color 2') # REMOVE THIS LATER, THIS IS JUST FOR FUN
        self.previous_asteroids_list = []
        self.last_respawn_maneuver_timestep_range = (-math.inf, 0)
        self.last_respawn_invincible_timestep_range = (-math.inf, 0)
        self.fire_next_timestep_flag = False
        self.previous_bullets = []
        self.game_state_plotter = None

    def finish_init(self, game_state, ship_state):
        # If we need the game state or ship state to finish init, we can use this function to do that
        if self.ship_id is None:
            self.ship_id = ship_state['id']
        if gamestate_plotting:
            self.game_state_plotter = GameStatePlotter(game_state)
        asteroid_density = ctrl.Antecedent(np.arange(0, 11, 1), 'asteroid_density')
        asteroids_entropy = ctrl.Antecedent(np.arange(0, 11, 1), 'asteroids_entropy')
        other_ship_lives = ctrl.Antecedent(np.arange(0, 4, 1), 'other_ship_lives')
        
        aggression = ctrl.Consequent(np.arange(0, 1, 1), 'asteroid_growth_factor')
        #if self.process_pool is None and __name__ != '__main__':
            #print(f"NAAAAAAAAAAAAAAAAAAMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE: {__name__}")
            #self.process_pool = multiprocessing.Pool(processes=4)


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

    def simulate_maneuver(self, ship_state, game_state, initial_turn_angle, accel_turn_rate, cruise_speed, cruise_turn_rate, cruise_timesteps, safe_timesteps=0, fire_first_timestep=False):
        maneuver_sim = Simulation(game_state, ship_state, self.current_timestep, game_state['asteroids'], self.asteroids_pending_death, self.forecasted_asteroid_splits, game_state['mines'], self.get_other_ships(game_state), game_state['bullets'], self.last_timestep_fired, safe_timesteps, fire_first_timestep, self.game_state_plotter)
        # This if statement is a doozy. Keep in mind that we evaluate the and clauses left to right, and the moment we find one that's false, we stop.
        # While evaluating, the simulation is advancing, and if it crashes, then it'll evaluate to false and stop the sim.
        if maneuver_sim.rotate_heading(initial_turn_angle) and maneuver_sim.accelerate(cruise_speed, accel_turn_rate) and maneuver_sim.cruise(cruise_timesteps, cruise_turn_rate) and maneuver_sim.accelerate(0):
            # The ship went through all the steps without colliding
            maneuver_complete_without_crash = True
        else:
            # The ship crashed somewhere before reaching the final resting spot
            maneuver_complete_without_crash = False
        
        #move_sequence = maneuver_sim.get_move_sequence()
        #state_sequence = maneuver_sim.get_state_sequence()
        maneuver_fitness = maneuver_sim.get_fitness()
        maneuver_length = maneuver_sim.get_sequence_length()
        next_imminent_collision_time = maneuver_length*delta_time
        if maneuver_complete_without_crash:
            next_imminent_collision_time += maneuver_sim.get_next_extrapolated_collision_time()
        safe_time_after_maneuver = max(0, next_imminent_collision_time - maneuver_length*delta_time)
        return maneuver_sim, maneuver_length, next_imminent_collision_time, safe_time_after_maneuver, maneuver_fitness

    def plan_action(self, ship_state: Dict, game_state: Dict):
        # Simulate and look for a good move
        # We have two options. Stay put and focus on targetting asteroids, or we can come up with an avoidance maneuver and target asteroids along the way if convenient
        # We simulate both options, and take the one with the higher fitness score
        # If we stay still, we can potentially keep shooting asteroids that are on collision course with us without having to move
        # But if we're overwhelmed, it may be a lot better to move to a safer spot
        # The third scenario is that even if we're safe where we are, we may be able to be on the offensive and seek out asteroids to lay mines, so that can also increase the fitness function of moving, making it better than staying still
        # Our number one priority is to stay alive. Second priority is to shoot as much as possible. And if we can, lay mines without putting ourselves in danger.

        # Stationary targetting simulation
        best_stationary_targetting_move_sequence = []
        best_stationary_targetting_fitness = math.inf # The lower the better
        #debug_print('Before sim, asteroids shot at:')
        #debug_print(self.asteroids_pending_death)
        #debug_print('And asteroids:')
        #debug_print(game_state['asteroids'])
        debug_print('Simulating stationary targetting:')
        stationary_targetting_sim = Simulation(game_state, ship_state, self.current_timestep, game_state['asteroids'], self.asteroids_pending_death, self.forecasted_asteroid_splits, game_state['mines'], self.get_other_ships(game_state), game_state['bullets'], self.last_timestep_fired, 0, self.fire_next_timestep_flag, self.game_state_plotter)
        stationary_targetting_sim.target_selection()
        best_stationary_targetting_move_sequence = stationary_targetting_sim.get_move_sequence()
        #print("stationary targetting move seq")
        #print(best_stationary_targetting_move_sequence)
        
        #debug_print('After sim, printing asteroids pending death:')
        #debug_print(self.asteroids_pending_death)
        #debug_print('also after sim, forecasted splits:')
        #debug_print(self.forecasted_asteroid_splits)
        if simulation_state_dump:
            append_dict_to_file(stationary_targetting_sim.get_state_sequence(), 'Simulation State Dump.txt')
        best_stationary_targetting_fitness = stationary_targetting_sim.get_fitness()

        # TODO: Make getting stuff from the sim more efficient by just keeping the object, and not pulling out data unnecessarily
        
        # Check for danger
        if best_stationary_targetting_fitness > 5:
            max_search_iterations = 100
            min_search_iterations = 10
        elif best_stationary_targetting_fitness > 4:
            max_search_iterations = 50
            min_search_iterations = 8
        elif best_stationary_targetting_fitness > 3:
            max_search_iterations = 30
            min_search_iterations = 5
        elif best_stationary_targetting_fitness > 1:
            max_search_iterations = 20
            min_search_iterations = 4
        else:
            # Stationary targetting is already very good, but we'll simulate a maneuver just in case we get something suuuuuper good
            max_search_iterations = 0
            min_search_iterations = 0
        max_cruise_seconds = 1
        next_imminent_collision_time = math.inf
        good_maneuver_fitness_threshold = 1




        #max_search_iterations = 0
        #min_search_iterations = 0



        # Try moving!
        #print("Look for a course of action to potentially move")
        # Run a simulation and find a course of action to put me to safety
        very_good_maneuver_found = False # If we can already achieve a very good fitness function, just call it a day and don't look for anything even better
        best_imminent_collision_time_found = -1
        best_maneuver_fitness = math.inf # The lower the better
        best_maneuver_sim = None
        best_safe_time_after_maneuver = -math.inf
        search_iterations_count = 0
        while search_iterations_count < min_search_iterations or (not very_good_maneuver_found and search_iterations_count < max_search_iterations):
            search_iterations_count += 1
            if search_iterations_count%5 == 0:
                debug_print(f"Search iteration {search_iterations_count}")
                pass
            random_ship_heading_angle = random.uniform(-30.0, 30.0)
            random_ship_accel_turn_rate = random.uniform(-ship_max_turn_rate, ship_max_turn_rate)
            #random_ship_cruise_speed = random.uniform(-ship_max_speed, ship_max_speed)
            random_ship_cruise_speed = random.triangular(0, ship_max_speed, ship_max_speed)*random.choice([-1, 1])
            random_ship_cruise_turn_rate = random.uniform(-ship_max_turn_rate, ship_max_turn_rate)
            random_ship_cruise_timesteps = random.randint(1, round(max_cruise_seconds/delta_time))
            maneuver_sim, maneuver_length, next_imminent_collision_time, safe_time_after_maneuver, maneuver_fitness = self.simulate_maneuver(ship_state, game_state, random_ship_heading_angle, random_ship_accel_turn_rate, random_ship_cruise_speed, random_ship_cruise_turn_rate, random_ship_cruise_timesteps, 0, self.fire_next_timestep_flag)
            if maneuver_fitness < best_maneuver_fitness:
                #print(f"Alright we found a better one with time {next_imminent_collision_time}")
                best_maneuver_fitness = maneuver_fitness
                best_imminent_collision_time_found = next_imminent_collision_time
                best_maneuver_sim = maneuver_sim
                best_safe_time_after_maneuver = safe_time_after_maneuver
                #best_maneuver_tuple = (random_ship_heading_angle, random_ship_accel_turn_rate, random_ship_cruise_speed, random_ship_cruise_turn_rate, random_ship_cruise_timesteps, next_imminent_collision_time, safe_time_after_maneuver, best_maneuver_move_sequence, best_maneuver_state_sequence)
            if best_maneuver_fitness < good_maneuver_fitness_threshold: #safe_time_after_maneuver >= safe_time_threshold:
                very_good_maneuver_found = True
                #print(f"Found safe maneuver! Next imminent collision time is {best_imminent_collision_time_found}")
                #print(f"Maneuver takes this many seconds: {maneuver_length*delta_time}")
                #print(f"Projected location to move will be to: {ship_sim_pos_x}, {ship_sim_pos_y} with heading {current_ship_heading + heading_difference_deg}")
                if search_iterations_count >= min_search_iterations:
                    break
                else:
                    continue
        if best_maneuver_sim is not None:
            debug_print(f"Did {search_iterations_count} search iterations to find a maneuver that buys us another {best_imminent_collision_time_found}s of life, and we're safe for {best_safe_time_after_maneuver}s after we come to a stop at coordinates {best_maneuver_sim.get_state_sequence()[-1]['position'][0]} {best_maneuver_sim.get_state_sequence()[-1]['position'][1]} at timestep {best_maneuver_sim.get_state_sequence()[-1]['timestep']}")

        #best_maneuver_fitness = 1000
        #end_state = sim_ship.get_state_sequence()[-1]
        debug_print(f"Maneuver fitness: {best_maneuver_fitness}, stationary fitness: {best_stationary_targetting_fitness}")
        if best_maneuver_fitness < best_stationary_targetting_fitness:
            #print("Decided that maneuvering is the best action")
            self.fire_next_timestep_flag = False
            best_move_sequence = best_maneuver_sim.get_move_sequence()
            self.asteroids_pending_death = best_maneuver_sim.get_asteroids_pending_death()
            self.forecasted_asteroid_splits = best_maneuver_sim.get_forecasted_asteroid_splits()
        else:
            #print("Decided that stationary targetting is the best action")
            best_move_sequence = best_stationary_targetting_move_sequence
            self.asteroids_pending_death = stationary_targetting_sim.get_asteroids_pending_death()
            self.forecasted_asteroid_splits = stationary_targetting_sim.get_forecasted_asteroid_splits()
            self.fire_next_timestep_flag = stationary_targetting_sim.get_fire_next_timestep_flag()
        
        for move in best_move_sequence:
            self.enqueue_action(move['timestep'], move['thrust'], move['turn_rate'], move['fire'])

        # Record the maneuver for statistical analysis
        '''
        if best_maneuver_tuple:
            #log_tuple_to_file(best_maneuver_tuple, 'Simulation State Dump.txt')
            move_seq = best_maneuver_tuple[7]
            state_seq = best_maneuver_tuple[8]
            state_dump_dict = {
                'timestep': self.current_timestep,
                'move_sequence': move_seq,
                'state_sequence': state_seq,
            }
            if simulation_state_dump:
                append_dict_to_file(state_dump_dict, 'Simulation State Dump.txt')
        '''

    def plan_respawn_maneuver_to_safety(self, ship_state, game_state):
        # Simulate and look for a good move
        #print(f"Checking for imminent danger. We're currently at position {ship_state['position'][0]} {ship_state['position'][1]}")
        #print(f"Current ship location: {ship_state['position'][0]}, {ship_state['position'][1]}, ship heading: {ship_state['heading']}")

        # Check for danger
        max_search_iterations = 40
        min_search_iterations = 10
        max_cruise_seconds = 1 + 26*delta_time
        #ship_random_range, ship_random_max_maneuver_length = get_simulated_ship_max_range(max_cruise_seconds)
        #print(f"Respawn maneuver max length: {ship_random_max_maneuver_length}s")
        next_imminent_collision_time = math.inf
        safe_time_threshold = 5

        debug_print("Look for a respawn maneuver")
        # Run a simulation and find a course of action to put me to safety
        safe_maneuver_found = False
        best_imminent_collision_time_found = -math.inf
        best_maneuver_sim = None
        best_safe_time_after_maneuver = -math.inf
        search_iterations_count = 0
        
        while search_iterations_count < min_search_iterations or (not safe_maneuver_found and search_iterations_count < max_search_iterations):
            search_iterations_count += 1
            if search_iterations_count % 5 == 0:
                debug_print(f"Search iteration {search_iterations_count}")
                pass
            random_ship_heading_angle = random.uniform(-20.0, 20.0)
            random_ship_accel_turn_rate = random.uniform(-ship_max_turn_rate, ship_max_turn_rate)
            random_ship_cruise_speed = ship_max_speed*np.sign(random.random() - 0.5)
            random_ship_cruise_turn_rate = 0
            random_ship_cruise_timesteps = random.randint(round(max_cruise_seconds/delta_time), round(max_cruise_seconds/delta_time))
            maneuver_sim, maneuver_length, next_imminent_collision_time, safe_time_after_maneuver, maneuver_fitness = self.simulate_maneuver(ship_state, game_state, random_ship_heading_angle, random_ship_accel_turn_rate, random_ship_cruise_speed, random_ship_cruise_turn_rate, random_ship_cruise_timesteps, math.inf)
            
            if next_imminent_collision_time > best_imminent_collision_time_found:
                debug_print(f"Alright we found a better one with time {next_imminent_collision_time}")
                best_imminent_collision_time_found = next_imminent_collision_time
                best_maneuver_sim = maneuver_sim
                best_safe_time_after_maneuver = safe_time_after_maneuver
            if safe_time_after_maneuver >= safe_time_threshold:
                safe_maneuver_found = True
                debug_print(f"Found safe maneuver! Next imminent collision time is {best_imminent_collision_time_found}")
                debug_print(f"Maneuver takes this many seconds: {maneuver_length*delta_time}")
                #print(f"Projected location to move will be to: {ship_sim_pos_x}, {ship_sim_pos_y} with heading {current_ship_heading + heading_difference_deg}")
                if search_iterations_count >= min_search_iterations:
                    break
                else:
                    continue
        if search_iterations_count == max_search_iterations:
            debug_print("Hit the max iteration count")
        debug_print(f"Did {search_iterations_count} search iterations to find a respawn maneuver where we're safe for {best_safe_time_after_maneuver}s afterwards. Moving to coordinates {best_maneuver_sim.get_state_sequence()[-1]['position'][0]} {best_maneuver_sim.get_state_sequence()[-1]['position'][1]} at timestep {best_maneuver_sim.get_state_sequence()[-1]['timestep']}")

        # Enqueue the respawn maneuver
        #print(best_maneuver_move_sequence)
        for move in best_maneuver_sim.get_move_sequence():
            self.enqueue_action(move['timestep'], move['thrust'], move['turn_rate'])
        
        self.last_respawn_maneuver_timestep_range = (self.current_timestep, self.current_timestep + best_maneuver_sim.get_sequence_length()) # TODO: CHECK FOR OFF BY ONE ERROR
        self.last_respawn_invincible_timestep_range = (self.current_timestep, self.current_timestep + 2 + round(respawn_invincibility_time/delta_time))
        #print(self.last_respawn_maneuver_timestep_range, self.last_respawn_invincible_timestep_range)
        return True

    def check_mine_opportunity(self, ship_state, game_state):
        average_asteroid_density = len(game_state['asteroids'])/(game_state['map_size'][0]*game_state['map_size'][1])
        average_asteroids_inside_blast_radius = average_asteroid_density*math.pi*mine_blast_radius**2
        mine_ast_count = count_asteroids_in_mine_blast_radius(game_state, game_state['asteroids'], ship_state['position'][0], ship_state['position'][1], round(mine_fuse_time/delta_time))
        debug_print(f"Mine count inside: {mine_ast_count} compared to average density amount inside: {average_asteroids_inside_blast_radius}")
        if (len(game_state['asteroids']) > 40 and mine_ast_count > 1.5*average_asteroids_inside_blast_radius or mine_ast_count > 20 or mine_ast_count > 2*average_asteroids_inside_blast_radius) and len(game_state['mines']) == 0:
            self.enqueue_action(self.current_timestep, None, None, None, True)

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        thrust_default, turn_rate_default, fire_default, drop_mine_default = 0, 0, False, False
        # Method processed each time step by this controller.
        self.current_timestep += 1
        debug_print(f"\n\nTimestep {self.current_timestep}, ship is at {ship_state['position'][0]} {ship_state['position'][1]}")
        if not self.init_done:
            self.finish_init(game_state, ship_state)
            self.init_done = True
        #print("thrust is " + str(thrust) + "\n" + "turn rate is " + str(turn_rate) + "\n" + "fire is " + str(fire) + "\n")
        if ship_state['is_respawning'] and not (self.last_respawn_invincible_timestep_range[0] <= self.current_timestep <= self.last_respawn_invincible_timestep_range[1]):
            debug_print("OUCH WE LOST A LIFE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Clear the move queue, since previous moves have been invalidated by us taking damage
            self.action_queue = []
            self.plan_respawn_maneuver_to_safety(ship_state, game_state)
        if not (self.last_respawn_maneuver_timestep_range[0] <= self.current_timestep <= self.last_respawn_maneuver_timestep_range[1]):
            # We're not in the process of doing our respawn maneuver
            if not self.action_queue:
                # Nothing's in the action queue. Evaluate the current situation and figure out the best course of action
                debug_print("Plan the next action.")
                self.plan_action(ship_state, game_state)
            #print(len(self.action_queue))
            if ship_state['mines_remaining'] > 0:
                pass
                self.check_mine_opportunity(ship_state, game_state)
        # Prune out the list of asteroids we shot at if the timestep (key) is in the past
        self.asteroids_pending_death = {timestep: asteroids for timestep, asteroids in self.asteroids_pending_death.items() if timestep > self.current_timestep}

        # Execute the actions already in the queue for this timestep

        # Initialize defaults. If a component of the action is missing, then the default value will be returned
        thrust_combined, turn_rate_combined, fire_combined, drop_mine_combined = thrust_default, turn_rate_default, fire_default, drop_mine_default

        while self.action_queue and self.action_queue[0][0] == self.current_timestep:
            #print("Stuff is in the queue!")
            _, _, thrust, turn_rate, fire, drop_mine = heapq.heappop(self.action_queue)
            thrust_combined = thrust if thrust is not None else thrust_combined
            turn_rate_combined = turn_rate if turn_rate is not None else turn_rate_combined
            fire_combined = fire if (fire is not None and fire_combined is not True) else fire_combined # Fire being true takes priority over false! This lets the fire last timestep enqueue to work, since it sets it to true before the sim's move sequence sets it to false
            drop_mine_combined = drop_mine if drop_mine is not None else drop_mine_combined
        if fire_combined == True and ship_state['can_fire']:
            self.last_timestep_fired = self.current_timestep
        # The next action in the queue is for a future timestep. All actions for this timestep are processed.
        # Bounds check the stuff before the Kessler code complains to me about it
        if thrust_combined < -ship_max_thrust or thrust_combined > ship_max_thrust:
            thrust_combined = min(max(-ship_max_thrust, thrust_combined), ship_max_thrust)
            print("Dude the thrust is too high, go fix your code >:(")
        if turn_rate_combined < -ship_max_turn_rate or turn_rate_combined > ship_max_turn_rate:
            turn_rate_combined = min(max(-ship_max_turn_rate, turn_rate_combined), ship_max_turn_rate)
            print("Dude the turn rate is too high, go fix your code >:(")
        if fire_combined and not ship_state['can_fire']:
            print("Why are you trying to fire when you haven't waited out the cooldown yet?")
        #print(ship_state)
        #if drop_mine_combined and not ship_state['can_deploy_mine']:
        #    print("You can't deploy mines dude!")
        debug_print(f"Inputs on timestep {self.current_timestep} - thrust: {thrust_combined}, turn_rate: {turn_rate_combined}, fire: {fire_combined}, drop_mine: {drop_mine_combined}")
        #time.sleep(0.15)
        #print(game_state, ship_state)
        #drop_mine_combined = random.random() < 0.01
        #debug_print(f"Asteroids on timestep {self.current_timestep}")
        #debug_print(game_state['asteroids'])

        def missing_elements(list1, list2):
            """
            Returns the elements that are in list1 but not in list2.

            Parameters:
            list1 (list): The first list.
            list2 (list): The second list, which is a subset of the first list.

            Returns:
            list: A list of elements that are in list1 but not in list2.
            """
            return [element for element in list1 if element not in list2 and 'timesteps_until_appearance' not in element]
        #if self.current_timestep == 1:
        #time.sleep(0.1)
        #debug_print('Asteroids killed this timestep:')
        #for a in self.previous_asteroids_list:
        #    a['position'] = (a['position'][0] + a['velocity'][0]*delta_time, a['position'][1] + a['velocity'][1]*delta_time)
        #self.previous_bullets = game_state['bullets']
        #print(f'CURRENT BULLETS INGAME ON TIMESTEP {self.current_timestep}:')
        #print(game_state['bullets'])
        #print("CURRENT ASTEROIDS INGAME:")
        #print(game_state['asteroids'])
        #debug_print(missing_elements(self.previous_asteroids_list, game_state['asteroids']))
        #self.previous_asteroids_list = game_state['asteroids']
        if gamestate_plotting:
            flattened_asteroids_pending_death = [ast for ast_list in self.asteroids_pending_death.values() for ast in ast_list]
            self.game_state_plotter.update_plot(game_state['asteroids'], ship_state, game_state['bullets'], [], flattened_asteroids_pending_death, self.forecasted_asteroid_splits, True, eps, 'REALITY')
        state_dump_dict = {
            'timestep': self.current_timestep,
            'ship_state': ship_state,
            'asteroids': game_state['asteroids'],
            'bullets': game_state['bullets'],
        }
        self.forecasted_asteroid_splits = maintain_forecasted_asteroids(self.forecasted_asteroid_splits, game_state, True)
        if reality_state_dump:
            append_dict_to_file(state_dump_dict, 'Reality State Dump.txt')
        return thrust_combined, turn_rate_combined, fire_combined, drop_mine_combined

if __name__ == '__main__':
    pass

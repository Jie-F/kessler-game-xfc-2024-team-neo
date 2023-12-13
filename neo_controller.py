# XFC 2024 - "Neo" Kessler controller
# Jie Fan 2023-2024
# jie.f@pm.me
# Feel free to reach out if you have questions, suggestions, or find a bug :)

import random
import math
from src.kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from collections import deque
import heapq
import sys
import time
import os
import copy

# TODO: Once we get hit, use the 3 seconds of invincibility and do a global search to find a spot on the map that we can stay for a long time, and then beeline it over there.
# Bonus points if we can also do a search and find a good spot to lay a mine and lay the mine as well as get to the safe spot all within 3 seconds

# TODO: Line of sight analysis, and use it to see which shots are feasible, before target selection/prioritization.

# TODO: Circular path ship cruising, constant time collision prediction?

# TODO: Dynamic culling for sim? Different radii, or maybe direction?

# TODO: Simulate the dynamics of the bullet and asteroids, so we know whether the bullet will hit, and exactly which one it'll hit

debug_mode = True
reality_state_dump = False
simulation_state_dump = False
delta_time = 1/30 # s/ts
#fire_time = 1/10  # seconds
bullet_speed = 800.0 # px/s
ship_max_turn_rate = 180.0 # deg/s
ship_max_thrust = 480.0 # px/s^2
ship_drag = 80.0 # px/s^2
ship_max_speed = 240.0 # px/s
eps = 0.00000001
ship_radius = 20.0 # px
timesteps_until_terminal_velocity = math.ceil(ship_max_speed/(ship_max_thrust - ship_drag)/delta_time) # Should be 18 timesteps
collision_check_pad = 3 # px
asteroid_aim_buffer_pixels = 7 # px
coordinate_bound_check_padding = 1 # px
mine_blast_radius = 150 # px
mine_fuse_time = 3 # s
asteroid_radii_lookup = [8*size for size in range(5)] # asteroid.py
asteroid_mass_lookup = [0.25*math.pi*(8*size)**2 for size in range(5)] # asteroid.py
respawn_invincibility_time = 3 # s
bullet_mass = 1
ship_avoidance_padding = 25
ship_avoidance_speed_padding_ratio = 1/100
bullet_length = 12 # px

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
    if math.isclose(Dax, 0, abs_tol=eps) and math.isclose(Day, 0, abs_tol=eps) and math.isclose(Dbx, 0, abs_tol=eps) and math.isclose(Dby, 0, abs_tol=eps) == 0:
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
        if a == 0:
            print("WARNING: This should never have happened, especially since we're already checking the case where all velocities are 0!")
        t1 = math.nan
        t2 = math.nan
    return t1, t2

def predict_next_imminent_collision_time_with_asteroid(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius):
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

def duplicate_asteroids_for_wraparound(asteroid, max_x, max_y, pattern='half_surround', directional_culling=True):
    duplicates = []

    # Original X and Y coordinates
    orig_x, orig_y = asteroid["position"]

    # Generate positions for the duplicates
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
                        # This imaginary asteroid's heading out into space
                        continue
            #print(f"It: col{col} row{row}")
            duplicate = asteroid.copy() # TODO: Check whether this copy is necessary
            #duplicate['dx'] = dx
            #duplicate['dy'] = dy
            duplicate['position'] = (orig_x + dx, orig_y + dy)
            
            duplicates.append(duplicate)
    return duplicates

def check_coordinate_bounds(game_state, x, y):
    if 0 <= x < game_state['map_size'][0] and 0 <= y < game_state['map_size'][1]:
        return True
    else:
        return False

def calculate_interception(ship_pos_x, ship_pos_y, asteroid_pos_x, asteroid_pos_y, asteroid_vel_x, asteroid_vel_y, asteroid_r, ship_heading, game_state, future_shooting_timesteps=0):
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
    if not check_coordinate_bounds(game_state, intercept_x, intercept_y):
        return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    
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
    return True, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time_s + future_shooting_timesteps*delta_time, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception

def target_priority(game_state, ship_state, a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y, imminent_collision_time_s, next_imminent_collision_time_stationary_s, close_proximity_asteroid_count):
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

def forecast_asteroid_splits(game_state, ship_state, a, timesteps_until_appearance, bullet_heading_deg):
    if a['size'] == 1:
        # Asteroids of size 1 don't split
        return []
    # Look at asteroid.py in the Kessler game's code
    bullet_vel_x = math.cos(math.radians(bullet_heading_deg))*bullet_speed
    bullet_vel_y = math.sin(math.radians(bullet_heading_deg))*bullet_speed
    vfx = (1/(bullet_mass + a['mass']))*(bullet_mass*bullet_vel_x + a['mass']*a['velocity'][0])
    vfy = (1/(bullet_mass + a['mass']))*(bullet_mass*bullet_vel_y + a['mass']*a['velocity'][1])
    # Calculate speed of resultant asteroid(s) based on velocity vector
    v = np.sqrt(vfx**2 + vfy**2)
    # Calculate angle of center asteroid for split (degrees)
    theta = math.degrees(np.arctan2(vfy, vfx))
    # Split angle is the angle off of the new velocity vector for the two asteroids to the sides, the center child
    # asteroid continues on the new velocity path
    split_angle = 15
    angles = [(theta + split_angle)%360, theta, (theta - split_angle)%360]
    # This is wacky because we're back-extrapolation the position of the asteroid BEFORE IT WAS BORN!!!!11!
    forecasted_asteroids = [{'position': (a['position'][0] + a['velocity'][0]*delta_time*timesteps_until_appearance - timesteps_until_appearance*math.cos(math.radians(angle))*v*delta_time, a['position'][1] + a['velocity'][1]*delta_time*timesteps_until_appearance - timesteps_until_appearance*math.sin(math.radians(angle))*v*delta_time),
                             'velocity': (math.cos(math.radians(angle))*v, math.sin(math.radians(angle))*v),
                             'size': a['size'] - 1,
                             'mass': asteroid_mass_lookup[a['size'] - 1],
                             'radius': asteroid_radii_lookup[a['size'] - 1],
                             'timesteps_until_appearance': timesteps_until_appearance}
                               for angle in angles]
    return forecasted_asteroids

def is_asteroid_in_list(list_of_asteroids, a, tolerance=1e-9):
    # Since floating point comparison isn't a good idea, break apart the asteroid dict and compare each element manually in a fuzzy way
    for asteroid in list_of_asteroids:
        if math.isclose(a['position'][0], asteroid['position'][0], abs_tol=tolerance) and math.isclose(a['position'][1], asteroid['position'][1], abs_tol=tolerance) and math.isclose(a['velocity'][0], asteroid['velocity'][0], abs_tol=tolerance) and math.isclose(a['velocity'][1], asteroid['velocity'][1], abs_tol=tolerance) and math.isclose(a['size'], asteroid['size'], abs_tol=tolerance) and math.isclose(a['mass'], asteroid['mass'], abs_tol=tolerance) and math.isclose(a['radius'], asteroid['radius'], abs_tol=tolerance):
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

def circle_line_collision(line_A, line_B, center, radius):
    # Check if circle edge is within the outer bounds of the line segment (offset for radius)
    # Not 100% accurate (some false positives) but fast and rare inaccuracies
    x_bounds = [min(line_A[0], line_B[0])-radius, max(line_A[0], line_B[0])+radius]
    if center[0] < x_bounds[0] or center[0] > x_bounds[1]:
        return False
    y_bounds = [min(line_A[1], line_B[1])-radius, max(line_A[1], line_B[1])+radius]
    if center[1] < y_bounds[0] or center[1] > y_bounds[1]:
        return False

    # calculate side lengths of triangle formed from the line segment and circle center point
    # round to 4 decimal places to prevent floating point error sqrt(-0) later
    a = round(math.dist(line_A, center), 4)
    b = round(math.dist(line_B, center), 4)
    c = round(math.dist(line_A, line_B), 4)

    # Heron's formula to calculate area of triangle and resultant height (distance from circle center to line segment)
    s = 0.5 * (a + b + c)

    cen_dist = 2 / c * np.sqrt(s * (s-a) * (s-b) * (s-c))

    # If circle distance to line segment is less than circle radius, they are colliding
    if cen_dist < radius:
        return True
    return False

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

class Simulation():
    # Simulates kessler_game.py and ship.py and other game mechanics
    def __init__(self, game_state, ship_state, initial_timestep, asteroids=[], mines=[], other_ships=[], bullets=[], last_timestep_fired=-math.inf, timesteps_to_not_check_collision_for=0):
        #debug_print("INITIALIZING SIMULATION")
        self.speed = ship_state['speed']
        self.position = ship_state['position']
        self.velocity = ship_state['velocity']
        self.heading = ship_state['heading']
        self.initial_timestep = initial_timestep
        self.future_timesteps = 0
        self.last_timestep_fired = last_timestep_fired
        self.asteroids = copy.deepcopy(asteroids)
        self.mines = copy.deepcopy(mines)
        self.other_ships = copy.deepcopy(other_ships)
        self.bullets = copy.deepcopy(bullets)
        self.bullets_remaining = math.inf if ship_state['bullets_remaining'] == -1 else ship_state['bullets_remaining']
        self.game_state = game_state
        self.ship_move_sequence = []
        self.state_sequence = [] #[{'timestep': self.initial_timestep, 'position': self.position, 'velocity': self.velocity, 'speed': self.speed, 'heading': self.heading, 'asteroids': copy.deepcopy(self.asteroids), 'bullets': copy.deepcopy(self.bullets)}]
        self.timesteps_to_not_check_collision_for = timesteps_to_not_check_collision_for

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
        # TODO: MAKE SURE THE MINES HAVE WRAPPING LOGIC
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
        for a in self.asteroids:
            if self.future_timesteps >= self.timesteps_to_not_check_collision_for:
                next_imminent_collision_time = min(predict_next_imminent_collision_time_with_asteroid(self.position[0], self.position[1], self.velocity[0], self.velocity[1], ship_radius, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius']), next_imminent_collision_time)
            else:
                # The asteroids position was never updated, so we need to extrapolate it right now in one step
                next_imminent_collision_time = min(predict_next_imminent_collision_time_with_asteroid(self.position[0] + self.future_timesteps*delta_time*self.velocity[0], self.position[1] + self.future_timesteps*delta_time*self.velocity[1], self.velocity[0], self.velocity[1], ship_radius, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius']), next_imminent_collision_time)
        for m in self.mines:
            next_imminent_collision_time = min(predict_ship_mine_collision(self.position[0], self.position[1], self.velocity[0], self.velocity[1], m, 0), next_imminent_collision_time)
        return next_imminent_collision_time

    def coordinates_in_same_wrap(self, pos1, pos2):
        # Checks whether the coordinates are in the same universe
        max_width = self.game_state['map_size'][0]
        max_height = self.game_state['map_size'][1]
        return pos1[0]//max_width == pos2[0]//max_width and pos1[1]//max_height == pos2[1]//max_height

    def update(self, thrust=0, turn_rate=0):
        # This is a highly optimized simulation of what kessler_game.py does, and should match exactly its behavior
        # Being even one timestep off is the difference between life and death!!!
        self.state_sequence.append({'timestep': self.initial_timestep + self.future_timesteps, 'position': self.position, 'velocity': self.velocity, 'speed': self.speed, 'heading': self.heading, 'asteroids': copy.deepcopy(self.asteroids), 'bullets': copy.deepcopy(self.bullets)})
        # The simulation starts by evaluating actions and dynamics of the current present timestep, and then steps into the future
        # The game state we're given is actually what we had at the end of the previous timestep
        # The game will take the previous state, and apply current actions and then update to get the result of this timestep
        # Apply drag. Fully stop the ship if it would cross zero speed in this time (prevents oscillation)

        # Simulation order:
        # Get controller outputs and apply to ship
        # Update bullet/mine/asteroid/ship positions
        # Check for collisions

        # TODO: Simplify simulation for respawning, and disable shooting

        # Simulate dynamics of bullets
        # It gets screwy when the ship can move between the bounds but the bullets can't.
        # The bounds are effectively walls that the bullets can't shoot through, but everything else can pass through
        # So we need to detect when the bullets are crossing the boundary, and delete them if they try to
        # Enumerate and track indices to delete
        bullet_remove_idxs = []
        for b_ind, b in enumerate(self.bullets):
            new_bullet_pos = (b['position'][0] + delta_time*b['velocity'][0], b['position'][1] + delta_time*b['velocity'][1])
            if self.coordinates_in_same_wrap(b['position'], new_bullet_pos):
                b['position'] = new_bullet_pos
            else:
                bullet_remove_idxs.append(b_ind)
            #b['tail'] = (b['tail'][0] + delta_time*b['velocity'][0], b['tail'][1] + delta_time*b['velocity'][1])
        self.bullets = [bullet for idx, bullet in enumerate(self.bullets) if idx not in bullet_remove_idxs]

        # Update mines
        if self.mines:
            debug_print(self.mines)
        for m in self.mines:
            m['fuse_time'] -= delta_time
            # If the timer is below eps, it'll detonate this timestep

        # Simulate dynamics of asteroids
        #print('before updating asteroids')
        #print(self.asteroids)
        #if self.asteroids:
        #    debug_print(self.asteroids[0])
        if self.future_timesteps >= self.timesteps_to_not_check_collision_for:
            for a in self.asteroids:
                a['position'] = (a['position'][0] + delta_time*a['velocity'][0], a['position'][1] + delta_time*a['velocity'][1])
        #print('after updating asteroids')
        #print(self.asteroids)

        
        # Bullet firing happens before we turn the ship
        # Check whether we want to shoot a simulated bullet
        timesteps_until_can_fire = max(0, 5 - (self.initial_timestep + self.future_timesteps - self.last_timestep_fired))
        fire_this_timestep = False
        if timesteps_until_can_fire == 0 and self.bullets_remaining > 0 and self.future_timesteps >= self.timesteps_to_not_check_collision_for:
            for a in self.asteroids:
                feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception = calculate_interception(self.position[0], self.position[1], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'], self.heading, self.game_state)
                if feasible and -shot_heading_tolerance_rad < shot_heading_error_rad < shot_heading_tolerance_rad:
                    fire_this_timestep = True
                    break
        
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
            if self.coordinates_in_same_wrap(self.position, (bullet_x, bullet_y)):
                self.tail = [self.position[0] - bullet_length*cos_heading, self.position[1] - bullet_length*sin_heading]
                vx = bullet_speed*cos_heading
                vy = bullet_speed*sin_heading
                new_bullet = {
                                'position': (bullet_x, bullet_y),
                                'velocity': (vx, vy),
                                'heading': self.heading,
                                'mass': bullet_mass,
                }
                self.bullets.append(new_bullet)

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
        self.velocity = [math.cos(math.radians(self.heading))*self.speed, math.sin(math.radians(self.heading))*self.speed]
        # Update the position based off the velocities
        self.position = [pos + vel*delta_time for pos, vel in zip(self.position, self.velocity)]
        # We don't wrap the position of the ship! We duplicate the asteroids around it, and collide with out of bounds asteroids to handle wrapping

        # Check asteroid-bullet collisions
        bullet_remove_idxs = []
        asteroid_remove_idxs = []
        for idx_bul, b in enumerate(self.bullets):
            for idx_ast, a in enumerate(self.asteroids):
                # If collision occurs
                # TODO: CREATE BULLET TAIL, or even better replace this with my own freaking function
                b_tail = (b['position'][0] - bullet_length*math.cos(math.radians(b['heading'])), b['position'][1] - bullet_length*math.sin(math.radians(b['heading'])))
                if circle_line_collision(b['position'], b_tail, a['position'], a['radius']):
                    # Increment hit values on ship that fired bullet then destruct bullet and mark for removal
                    bullet_remove_idxs.append(idx_bul)
                    # Asteroid destruct function and mark for removal
                    self.asteroids.extend(forecast_asteroid_splits(None, None, a, 0, b['heading']))
                    asteroid_remove_idxs.append(idx_ast)
                    # Stop checking this bullet
                    break
        # Cull bullets and asteroids that are marked for removal
        self.bullets = [bullet for idx, bullet in enumerate(self.bullets) if idx not in bullet_remove_idxs]
        self.asteroids = [asteroid for idx, asteroid in enumerate(self.asteroids) if idx not in asteroid_remove_idxs]

        # Check asteroid-ship and ship-ship collisions
        if self.future_timesteps >= self.timesteps_to_not_check_collision_for:
            if self.get_instantaneous_asteroid_collision() or self.get_instantaneous_ship_collision():
                return False

        # Check mine-ship collisions
        # Even if the player is invincible, we still need to check for these collisions since invincibility is only for asteroids and ship-ship collisions
        if self.get_instantaneous_mine_collision():
            return False

        # Update mine-asteroid collisions
        # TODO

        self.ship_move_sequence.append({'timestep': self.initial_timestep + self.future_timesteps, 'thrust': thrust, 'turn_rate': turn_rate, 'fire': fire_this_timestep})
        self.future_timesteps += 1
        return True

    def rotate_heading(self, heading_difference_deg):
        target_heading = (self.heading + heading_difference_deg)%360
        still_need_to_turn = heading_difference_deg
        while abs(still_need_to_turn) > ship_max_turn_rate*delta_time:
            if not self.update(0, ship_max_turn_rate*np.sign(heading_difference_deg)):
                return False
            still_need_to_turn -= ship_max_turn_rate*np.sign(heading_difference_deg)*delta_time
        if not self.update(0, still_need_to_turn/delta_time):
            return False
        assert(abs(target_heading - self.heading) < eps)
        return True

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
    
    def get_state_sequence(self):
        return self.state_sequence

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

    def finish_init(self, game_state, ship_state):
        # If we need the game state or ship state to finish init, we can use this function to do that
        if self.ship_id is None:
            self.ship_id = ship_state['id']

    def find_other_ship_positions(self, game_state):
        other_positions = []
        for ship in game_state['ships']:
            if ship['id'] != self.ship_id:
                other_positions.append(ship['position'])
        return other_positions

    def track_asteroid_we_shot_at(self, game_state, future_hit_timesteps, asteroid):
        asteroid = asteroid.copy()
        future_hit_timesteps += 1 # Fudge ahfsdufoisdfoij
        #print('track asteroid')
        # Project the asteroid into the future, to where it would be on the timestep of its death
        for future_timesteps in range(0, future_hit_timesteps + 1):
            #print(asteroid)
            timestep = self.current_timestep + future_timesteps
            if timestep not in self.asteroids_pending_death:
                self.asteroids_pending_death[timestep] = [asteroid.copy()]
            else:
                self.asteroids_pending_death[timestep].append(asteroid.copy())
            # Advance the asteroid to the next position
            # TODO: Remove this redundant operation on the last iteration
            '''
            asteroid_new_x = asteroid['position'][0] + asteroid['velocity'][0]*delta_time
            asteroid_new_y = asteroid['position'][1] + asteroid['velocity'][1]*delta_time
            # Wrap the exact same way the game does it (can't use %, it's inaccurate)
            asteroid_wrapped_x = asteroid_new_x
            offset = game_state['map_size'][0] - asteroid_new_x
            if offset < 0 or offset > game_state['map_size'][0]:
                asteroid_wrapped_x += game_state['map_size'][0] * np.sign(offset)
            asteroid_wrapped_y = asteroid_new_y
            offset = game_state['map_size'][1] - asteroid_new_y
            '''
            #if offset < 0 or offset > game_state['map_size'][1]:
            #    asteroid_wrapped_y += game_state['map_size'][1] * np.sign(offset)
            asteroid['position'] = ((asteroid['position'][0] + asteroid['velocity'][0]*delta_time)%game_state['map_size'][0], (asteroid['position'][1] + asteroid['velocity'][1]*delta_time)%game_state['map_size'][1])
        #print('done tracking asteroid')

    def check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self, game_state, asteroid):
        # Check whether the asteroid has already been shot at, or if we can shoot at it again
        asteroid = asteroid.copy()
        # Wrap the exact same way the game does it (can't use %, it's inaccurate)
        '''
        asteroid_wrapped_x = asteroid['position'][0]
        offset = game_state['map_size'][0] - asteroid['position'][0]
        if offset < 0 or offset > game_state['map_size'][0]:
            asteroid_wrapped_x += game_state['map_size'][0] * np.sign(offset)
        asteroid_wrapped_y = asteroid['position'][1]
        offset = game_state['map_size'][1] - asteroid['position'][1]
        if offset < 0 or offset > game_state['map_size'][1]:
            asteroid_wrapped_y += game_state['map_size'][1] * np.sign(offset)
        '''
        asteroid['position'] = (asteroid['position'][0]%game_state['map_size'][0], asteroid['position'][1]%game_state['map_size'][1]) #Dangit this wrapping code isn't as precise I think
        #asteroid['position'] = (asteroid_wrapped_x, asteroid_wrapped_y)
        if self.current_timestep not in self.asteroids_pending_death:
            return True
        elif is_asteroid_in_list(self.asteroids_pending_death[self.current_timestep], asteroid):
            #print('NOPE WE CANT SHOOT AT THIS AGANI')
            return False
        else:
            #print(asteroid)
            #print('IS NOT IN THE LIST')
            #print(self.asteroids_pending_death)
            return True

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

    def simulate_maneuver(self, ship_state, game_state, asteroids, safe_time_threshold, initial_turn_angle, accel_turn_rate, cruise_speed, cruise_turn_rate, cruise_timesteps, safe_timesteps=0):
        sim_ship = Simulation(game_state, ship_state, self.current_timestep, asteroids, game_state['mines'], self.find_other_ship_positions(game_state), game_state['bullets'], self.last_timestep_fired, safe_timesteps)
        # This if statement is a doozy. Keep in mind that we evaluate the and clauses left to right, and the moment we find one that's false, we stop.
        # While evaluating, the simulation is advancing, and if it crashes, then it'll evaluate to false and stop the sim.
        if sim_ship.rotate_heading(initial_turn_angle) and sim_ship.accelerate(cruise_speed, accel_turn_rate) and sim_ship.cruise(cruise_timesteps, cruise_turn_rate) and sim_ship.accelerate(0):
            # The ship went through all the steps without colliding
            maneuver_complete_without_crash = True
        else:
            # The ship crashed somewhere before reaching the final resting spot
            maneuver_complete_without_crash = False
        
        move_sequence = sim_ship.get_move_sequence()
        state_sequence = sim_ship.get_state_sequence()
        maneuver_length = len(move_sequence)
        next_imminent_collision_time = maneuver_length*delta_time
        if maneuver_complete_without_crash:
            next_imminent_collision_time += sim_ship.get_next_extrapolated_collision_time()
        safe_time_after_maneuver = max(0, next_imminent_collision_time - maneuver_length*delta_time)
        return maneuver_length, next_imminent_collision_time, safe_time_after_maneuver, move_sequence, state_sequence

    def plan_maneuver(self, ship_state: Dict, game_state: Dict):
        # Simulate and look for a good move
        #print(f"Checking for imminent danger. We're currently at position {ship_state['position'][0]} {ship_state['position'][1]}")
        #print(f"Current ship location: {ship_state['position'][0]}, {ship_state['position'][1]}, ship heading: {ship_state['heading']}")

        # Check for danger
        max_search_iterations = 10
        min_search_iterations = 5
        max_cruise_seconds = 1
        ship_random_range, ship_random_max_maneuver_length = get_simulated_ship_max_range(max_cruise_seconds)
        next_imminent_collision_time_stationary = math.inf
        next_imminent_collision_time = math.inf
        safe_time_threshold = 5
        # Check ship-asteroid imminent collisions
        relevant_asteroids = []
        for real_asteroid in game_state['asteroids']:
            duplicated_asteroids = duplicate_asteroids_for_wraparound(real_asteroid, game_state['map_size'][0], game_state['map_size'][1], pattern='surround', directional_culling=False)
            for a in duplicated_asteroids:
                next_imminent_collision_time_stationary = min(predict_next_imminent_collision_time_with_asteroid(ship_state['position'][0], ship_state['position'][1], ship_state['velocity'][0], ship_state['velocity'][1], ship_state['radius'], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius']), next_imminent_collision_time_stationary)
                # Check whether this asteroid will get inside the "bubble" that the ship could possibly move in, and if not, cull it
                #asteroid_gets_inside_range = check_collision(ship_state['position'][0], ship_state['position'][1], ship_state['radius'] + ship_random_range, a['position'][0], a['position'][1], a['radius'])
                t1, t2 = collision_prediction(ship_state['position'][0], ship_state['position'][1], ship_state['velocity'][0], ship_state['velocity'][1], ship_state['radius'] + ship_random_range, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'])
                if not math.isnan(t1) and not math.isnan(t2) and not ((t1 < 0 and t2 < 0) or (t1 > ship_random_max_maneuver_length*delta_time + safe_time_threshold and t2 > ship_random_max_maneuver_length*delta_time + safe_time_threshold)):
                    relevant_asteroids.append(a)
        # Check ship-mine imminent collisions
        for mine in game_state['mines']:
            next_imminent_collision_time_stationary = min(predict_ship_mine_collision(ship_state['position'][0], ship_state['position'][1], ship_state['velocity'][0], ship_state['velocity'][1], mine), next_imminent_collision_time_stationary)
        #print(f"Next imminent collision is in {next_imminent_collision_time_stationary}s if we don't move")
        #print(f"Amount of asteroids before culling: {len(game_state['asteroids'])} and after culling, but including duplicates: {len(relevant_asteroids)}")
        
        very_dangerous_time = 1.2
        moving_multiplies_safe_time_by_at_least = 1.5
        if next_imminent_collision_time_stationary > safe_time_threshold:
            # Nothing's gonna hit us for a long time
            #print(f"We're safe for the next {safe_time_threshold} seconds")
            #self.enqueue_action(self.current_timestep)
            return False
        else:
            #print("Look for a course of action to potentially move")
            # Run a simulation and find a course of action to put me to safety
            safe_maneuver_found = False
            best_imminent_collision_time_found = -1
            best_maneuver_move_sequence = []
            best_maneuver_state_sequence = []
            best_safe_time_after_maneuver = -math.inf
            search_iterations_count = 0
            best_maneuver_tuple = None
            while search_iterations_count < min_search_iterations or (not safe_maneuver_found and search_iterations_count < max_search_iterations):
                search_iterations_count += 1
                if search_iterations_count % 5 == 0:
                    debug_print(f"Search iteration {search_iterations_count}")
                    pass
                random_ship_heading_angle = random.uniform(-30.0, 30.0)
                random_ship_accel_turn_rate = random.uniform(-ship_max_turn_rate, ship_max_turn_rate)
                random_ship_cruise_speed = random.uniform(-ship_max_speed, ship_max_speed)
                random_ship_cruise_turn_rate = random.uniform(-ship_max_turn_rate, ship_max_turn_rate)
                random_ship_cruise_timesteps = random.randint(1, round(max_cruise_seconds/delta_time))
                maneuver_length, next_imminent_collision_time, safe_time_after_maneuver, move_sequence, state_sequence = self.simulate_maneuver(ship_state, game_state, relevant_asteroids, safe_time_threshold, random_ship_heading_angle, random_ship_accel_turn_rate, random_ship_cruise_speed, random_ship_cruise_turn_rate, random_ship_cruise_timesteps)
                
                if next_imminent_collision_time > best_imminent_collision_time_found:
                    #print(f"Alright we found a better one with time {next_imminent_collision_time}")
                    best_imminent_collision_time_found = next_imminent_collision_time
                    best_maneuver_move_sequence = move_sequence
                    best_maneuver_state_sequence = state_sequence
                    best_safe_time_after_maneuver = safe_time_after_maneuver
                    best_maneuver_tuple = (random_ship_heading_angle, random_ship_accel_turn_rate, random_ship_cruise_speed, random_ship_cruise_turn_rate, random_ship_cruise_timesteps, next_imminent_collision_time, safe_time_after_maneuver, best_maneuver_move_sequence, best_maneuver_state_sequence)
                if safe_time_after_maneuver >= safe_time_threshold:
                    safe_maneuver_found = True
                    #print(f"Found safe maneuver! Next imminent collision time is {best_imminent_collision_time_found}")
                    #print(f"Maneuver takes this many seconds: {maneuver_length*delta_time}")
                    #print(f"Projected location to move will be to: {ship_sim_pos_x}, {ship_sim_pos_y} with heading {current_ship_heading + heading_difference_deg}")
                    if search_iterations_count >= min_search_iterations:
                        break
                    else:
                        continue
            if search_iterations_count == max_search_iterations:
                #print("Hit the max iteration count")
                pass
            debug_print(f"Did {search_iterations_count} search iterations to find a maneuver that buys us another {best_imminent_collision_time_found}s of life, and we're safe for {best_safe_time_after_maneuver}s after we come to a stop at coordinates {best_maneuver_state_sequence[-1]['position'][0]} {best_maneuver_state_sequence[-1]['position'][1]} at timestep {best_maneuver_state_sequence[-1]['timestep']}")
            
            #end_state = sim_ship.get_state_sequence()[-1]
            if best_safe_time_after_maneuver > next_imminent_collision_time_stationary or (best_safe_time_after_maneuver > very_dangerous_time and len(game_state['mines']) > 0) or (next_imminent_collision_time_stationary < very_dangerous_time and best_imminent_collision_time_found > next_imminent_collision_time_stationary and best_safe_time_after_maneuver > 0):
                # K yeah it's worth moving, this will keep us alive for at least as long as we'll live if we stay here
                #print("K yeah it's worth moving")
                #print(f"We're gonna end up at {end_state[1][0]}, {end_state[1][1]}")
                # Enqueue the safe maneuver
                for move in best_maneuver_move_sequence:
                    self.enqueue_action(move['timestep'], move['thrust'], move['turn_rate'], move['fire'])
                # Record the maneuver for statistical analysis
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
                if not best_maneuver_move_sequence:
                    # Null action, just so I can take damage and the sim doesn't crash
                    #self.enqueue_action(self.current_timestep)
                    return False
                else:
                    return True
            else:
                # Stay put, not worth moving
                #print("Stay put, not worth moving")
                return False
                #self.enqueue_action(self.current_timestep)
            #print(self.action_queue)

    def check_convenient_shots(self, ship_state, game_state):
        # Iterate over all asteroids including duplicated ones, and check whether we're perfectly in line to hit any
        feasible_to_hit = []
        for real_asteroid in (game_state['asteroids'] + self.forecasted_asteroid_splits):
            duplicated_asteroids = duplicate_asteroids_for_wraparound(real_asteroid, game_state['map_size'][0], game_state['map_size'][1], pattern='half_surround')
            for a in duplicated_asteroids:
                feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception = calculate_interception(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'], ship_state['heading'], game_state)
                if feasible and a['size'] == 1:
                    feasible_to_hit.append((a, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, asteroid_dist_during_interception))
        #print(feasible_to_hit)
        # Iterate through the feasible ones and figure out which one we're actually going to hit if we shoot
        if ship_state['can_fire']:
            most_direct_asteroid_shot_tuple = None
            for feasible in feasible_to_hit:
                a, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, asteroid_dist_during_interception = feasible
                if self.check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(game_state, a):
                    #print(f"Shot tolerance is and the ship's heading is {math.radians(ship_state['heading'])}")
                    if abs(shot_heading_error_rad) < shot_heading_tolerance_rad:
                        if most_direct_asteroid_shot_tuple == None or asteroid_dist_during_interception < most_direct_asteroid_shot_tuple[4]:
                            most_direct_asteroid_shot_tuple = feasible
            if most_direct_asteroid_shot_tuple is not None:
                #print("Shooting the convenient shot!")
                self.enqueue_action(self.current_timestep, None, None, True)
                self.track_asteroid_we_shot_at(game_state, calculate_timesteps_until_bullet_hits_asteroid(most_direct_asteroid_shot_tuple[3], most_direct_asteroid_shot_tuple[0]['radius']), most_direct_asteroid_shot_tuple[0])
                self.forecasted_asteroid_splits.extend(forecast_asteroid_splits(game_state, ship_state, most_direct_asteroid_shot_tuple[0], calculate_timesteps_until_bullet_hits_asteroid(most_direct_asteroid_shot_tuple[3], most_direct_asteroid_shot_tuple[0]['radius']), ship_state['heading'] + math.degrees(shot_heading_error_rad)))

    def get_feasible_intercept_angle_and_turn_time(self, a, ship_state, game_state, timesteps_until_can_fire=0):
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
        feasible = True
        converged = False
        aiming_timesteps_required = 0
        shot_heading_error_rad = math.inf
        shot_heading_tolerance_rad = 0
        shooting_angle_error_rad = math.nan
        # TODO: USE BINARY SEARCH INSTEAD OF LINEAR SEARCH
        while feasible and not converged:
            feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time_s, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception = calculate_interception(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'], ship_state['heading'], game_state, timesteps_until_can_fire + aiming_timesteps_required)
            #print(f"Feasible intercept ({intercept_x}, {intercept_y})")
            #print(f'Still not converged, extra timesteps: {aiming_timesteps_required}, shot_heading_error_rad: {shot_heading_error_rad}, tol: {shot_heading_tolerance_rad}')
            if not feasible:
                break
            # For each asteroid, because the asteroid has a size, there is a range in angles in which we can shoot it.
            # We don't have to hit the very center, just close enough to it so that it's still the thick part of the circle and it won't skim the tangent
            # TODO: If we can aim at the center of the asteroid with no additional timesteps required, then just might as well do it cuz yeah
            #print(f"Shot heading error {shot_heading_error_rad}, shot heading tol: {shot_heading_tolerance_rad}, whether it's within: {abs(shot_heading_error_rad) <= shot_heading_tolerance_rad}")
            if abs(shot_heading_error_rad) <= shot_heading_tolerance_rad:
                shooting_angle_error_rad_using_tolerance = 0
            else:
                if shot_heading_error_rad > 0:
                    shooting_angle_error_rad_using_tolerance = shot_heading_error_rad - shot_heading_tolerance_rad
                else:
                    shooting_angle_error_rad_using_tolerance = shot_heading_error_rad + shot_heading_tolerance_rad
            # shooting_angle_error is the amount we need to move our heading by, in radians
            # If there's some timesteps until we can fire, then we can get a head start on the aiming
            new_aiming_timesteps_required = max(0, math.ceil(math.degrees(abs(shooting_angle_error_rad_using_tolerance))/(ship_max_turn_rate*delta_time) - eps) - timesteps_until_can_fire)
            new_aiming_timesteps_required_exact_aiming = max(0, math.ceil(math.degrees(abs(shot_heading_error_rad))/(ship_max_turn_rate*delta_time) - eps) - timesteps_until_can_fire)
            if new_aiming_timesteps_required == new_aiming_timesteps_required_exact_aiming:
                # If it doesn't save any time to not use exact aiming, just exactly aim at the center of the asteroid. It also prevents potential edge cases where it aims at the side of the asteroid but it doesn't shoot it (see XFC 2021 scenario_2_still_corridors scenario)
                shooting_angle_error_rad = shot_heading_error_rad
            else:
                shooting_angle_error_rad = shooting_angle_error_rad_using_tolerance
            if aiming_timesteps_required == new_aiming_timesteps_required:
                #print(f"Converged. Aiming timesteps required is {aiming_timesteps_required}")
                converged = True
            elif aiming_timesteps_required > new_aiming_timesteps_required:
                # Wacky oscillation case, IGNORE FOR NOW
                converged = True
            else:
                aiming_timesteps_required += 1
        shooting_angle_error_deg = math.degrees(shooting_angle_error_rad)
        # TODO: OPPOSITNG CASE SHOOTING OTHER WAY IT'S MORE COMPLICATED TO CONSIDER
        # OH WAIT THIS CASE ISN'T REALLY POSSIBLE BECAUSE THE ASTEROID WOULD HAVE TO BE MOVING SUPER FAST. The highest I've seen the absolute shooting angle error is like 135 degrees, and that's an extreme made up scenario
        if abs(shooting_angle_error_deg) >= 180:
            #print(f'shooting_angle_error_deg: {shooting_angle_error_deg} asdy8f9y7asdf89asdy789fads789y9asdf78y7asdfy79asdfy789asdfy789asdfy789asdfy978asdfy789asdfy78adfsy78asdfy78y78asdfyf87asdyf78dsay9f78dsayf78sy78fsya78fysa78fy78sdayf78ysa78yf78asdy78fas')
            pass
        if not feasible:
            return False, None, None, None, None, None, None
        return feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception

    def plan_targetting(self, ship_state, game_state):
        # Iterate over all asteroids including duplicated ones to get the ones that can be feasibly aimed for on the current as well as next timestep, not considering line of sight
        feasible_to_hit_current_timestep = []
        feasible_to_hit_next_timestep = []
        next_imminent_collision_time_stationary = math.inf
        #print(self.last_time_fired, self.current_timestep)
        timesteps_until_can_fire = max(0, 5 - (self.current_timestep - self.last_timestep_fired))
        #print(f"Timesteps until we can fire: {timesteps_until_can_fire}")
        for real_asteroid in (game_state['asteroids'] + self.forecasted_asteroid_splits):
            duplicated_asteroids = duplicate_asteroids_for_wraparound(real_asteroid, game_state['map_size'][0], game_state['map_size'][1], 'half_surround')
            #print(duplicated_asteroids)
            for a in duplicated_asteroids:
                # Find imminent collision time of the ship, to make targetting decisions later
                next_imminent_collision_time_stationary = min(predict_next_imminent_collision_time_with_asteroid(ship_state['position'][0], ship_state['position'][1], ship_state['velocity'][0], ship_state['velocity'][1], ship_state['radius'], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius']), next_imminent_collision_time_stationary)

                if self.check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(game_state, a):
                    imminent_collision_time_s = predict_next_imminent_collision_time_with_asteroid(ship_state['position'][0], ship_state['position'][1], ship_state['velocity'][0], ship_state['velocity'][1], ship_radius, a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'])

                    feasible_current_timestep, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = self.get_feasible_intercept_angle_and_turn_time(a, ship_state, game_state, timesteps_until_can_fire)
                    #print(f"Converged. Need extra timesteps: {aiming_timesteps_required}")
                    if feasible_current_timestep:
                        feasible_to_hit_current_timestep.append((a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y, imminent_collision_time_s))
                    if timesteps_until_can_fire == 0:
                        # Also calculate for the next timestep in case we can fire on this timestep and begin turning toward our next target for the future
                        feasible_next_timestep, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = self.get_feasible_intercept_angle_and_turn_time(a, ship_state, game_state, timesteps_until_can_fire + 1)
                        if feasible_next_timestep:
                            feasible_to_hit_next_timestep.append((a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y, imminent_collision_time_s - delta_time))
        #print("Feasible to hit list:")
        #print(feasible_to_hit_current_timestep)
        #print(feasible_to_hit_next_timestep)

        close_proximity_asteroid_count = count_asteroids_in_mine_blast_radius(game_state, game_state['asteroids'], ship_state['position'][0], ship_state['position'][1], 0)

        # Iterate through the feasible asteroids and figure out which one we're actually probably going to hit if we shoot (since some asteroids can cover up other asteroids and you may not hit your intended target)
        best_target_priority = math.inf
        most_direct_asteroid_shot_tuple = None
        # TODO: min_abs_shooting_angle_error is kind of a useless temp variable, but it might be good to keep anyway
        for feasible in feasible_to_hit_current_timestep:
            #print(shot_heading_error_rad)
            # Shooting_angle_deg already considers that we don't have to aim at the center of the asteroid. It shoots the side of the asteroid that requires us to aim less
            a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y, imminent_collision_time_s = feasible
            # TODO: Clean up the if statements below and perhaps combine redundant cases
            current_target_priority = target_priority(game_state, ship_state, a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y, imminent_collision_time_s, next_imminent_collision_time_stationary, close_proximity_asteroid_count)
            if most_direct_asteroid_shot_tuple is None:
                most_direct_asteroid_shot_tuple = feasible
                best_target_priority = current_target_priority
            elif current_target_priority < best_target_priority:
                #print(f"{shooting_angle_error_deg} is better than {min_abs_shooting_angle_error_deg}")
                # New best target that requires the least movement to hit
                most_direct_asteroid_shot_tuple = feasible
                best_target_priority = current_target_priority
        #print(timesteps_until_can_fire)
        #assert((timesteps_until_can_fire == 0) == ship_state['can_fire'])

        if most_direct_asteroid_shot_tuple is not None:
            best_target_priority = math.inf
            second_most_direct_asteroid_shot_tuple = None
            for feasible in feasible_to_hit_next_timestep:
                #print(shot_heading_error_rad)
                # Shooting_angle_deg already considers that we don't have to aim at the center of the asteroid. It shoots the side of the asteroid that requires us to aim less
                a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y, imminent_collision_time_s = feasible
                if most_direct_asteroid_shot_tuple[0]['velocity'][0] != a['velocity'][0] and most_direct_asteroid_shot_tuple[0]['velocity'][1] != a['velocity'][1] and most_direct_asteroid_shot_tuple[0]['position'][0] != a['position'][0] and most_direct_asteroid_shot_tuple[0]['position'][1] != a['position'][1]:
                    # As long as the second one isn't the same as the first asteroid lmao
                    # TODO: Clean up the if statements below and perhaps combine redundant cases
                    current_target_priority = target_priority(game_state, ship_state, a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y, imminent_collision_time_s, next_imminent_collision_time_stationary, close_proximity_asteroid_count)
                    if second_most_direct_asteroid_shot_tuple is None:
                        second_most_direct_asteroid_shot_tuple = feasible
                        best_target_priority = current_target_priority
                    elif current_target_priority < best_target_priority:
                        #print(f"{shooting_angle_error_deg} is better than {min_abs_shooting_angle_error_deg}")
                        # New best target that requires the least movement to hit
                        second_most_direct_asteroid_shot_tuple = feasible
                        best_target_priority = current_target_priority

        if most_direct_asteroid_shot_tuple is not None:
            # We have a #1 target
            min_abs_shooting_angle_error_deg = most_direct_asteroid_shot_tuple[1]
            #print(f"Our #1 target is angularlly away in degrees: {min_abs_shooting_angle_error_deg} and in num timesteps required for aiming: {most_direct_asteroid_shot_tuple[4]}")
            fired_on_this_timestep = False
            if abs(min_abs_shooting_angle_error_deg) < eps and ship_state['can_fire']:
                debug_print("Locked and loaded. We're firing on this timestep")
                #print(most_direct_asteroid_shot_tuple[0])
                #time.sleep(0.4)
                fired_on_this_timestep = True
                self.enqueue_action(self.current_timestep, None, None, True)
                #print(f"Time it takes to intercept asteroid with bullet: {most_direct_asteroid_shot_tuple[2]}")
                self.track_asteroid_we_shot_at(game_state, calculate_timesteps_until_bullet_hits_asteroid(most_direct_asteroid_shot_tuple[2], most_direct_asteroid_shot_tuple[0]['radius']), most_direct_asteroid_shot_tuple[0])
                self.forecasted_asteroid_splits.extend(forecast_asteroid_splits(game_state, ship_state, most_direct_asteroid_shot_tuple[0], calculate_timesteps_until_bullet_hits_asteroid(most_direct_asteroid_shot_tuple[2], most_direct_asteroid_shot_tuple[0]['radius']), ship_state['heading'] + min_abs_shooting_angle_error_deg))
                # The way the game works is at this timestep, I can fire a bullet and begin turning toward my second target
                # Aim at the second best target right now, since we're already shooting our best target and want something to do
                if second_most_direct_asteroid_shot_tuple is not None:
                    #print("Aiming at second best target on the same timestep, so we might be able to shoot it at the next timestep")
                    min_abs_shooting_angle_error_deg = second_most_direct_asteroid_shot_tuple[1]
            elif abs(min_abs_shooting_angle_error_deg) < eps:
                #print("DANG IT we're locked in, but we can't fire right now!")
                pass
            if not fired_on_this_timestep or second_most_direct_asteroid_shot_tuple is not None:
                # As long as we don't have the situation where we already shot at the first one but there's no second one to move to, move!
                # Alright so we have to aim as much toward this asteroid as possible. If we can aim at it in one timestep, then do that. If not, get as far as we can.
                if abs(min_abs_shooting_angle_error_deg) > ship_max_turn_rate*delta_time:
                    # Error is larger than the amount we can turn in one timestep, so use max turn
                    turn_rate = ship_max_turn_rate*np.sign(min_abs_shooting_angle_error_deg)
                else:
                    turn_rate = min_abs_shooting_angle_error_deg/delta_time
                #print(f"Aiming to prep to shoot! We just enqueued the turn rate {turn_rate}")
                self.enqueue_action(self.current_timestep, None, turn_rate)

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
        next_imminent_collision_time_stationary = math.inf
        next_imminent_collision_time = math.inf
        safe_time_threshold = 5

        relevant_asteroids = []
        for real_asteroid in game_state['asteroids']:
            duplicated_asteroids = duplicate_asteroids_for_wraparound(real_asteroid, game_state['map_size'][0], game_state['map_size'][1], pattern='half_surround', directional_culling=False)
            relevant_asteroids.extend(duplicated_asteroids)

        very_dangerous_time = 0.8
        moving_multiplies_safe_time_by_at_least = 1.5
        debug_print("Look for a respawn maneuver")
        # Run a simulation and find a course of action to put me to safety
        safe_maneuver_found = False
        best_imminent_collision_time_found = -math.inf
        best_maneuver_move_sequence = []
        best_maneuver_state_sequence = []
        best_safe_time_after_maneuver = -math.inf
        search_iterations_count = 0
        
        while search_iterations_count < min_search_iterations or (not safe_maneuver_found and search_iterations_count < max_search_iterations):
            search_iterations_count += 1
            if search_iterations_count % 5 == 0:
                debug_print(f"Search iteration {search_iterations_count}")
                pass
            random_ship_heading_angle = random.uniform(-20.0, 20.0)
            random_ship_accel_turn_rate = random.uniform(-ship_max_turn_rate, ship_max_turn_rate)
            random_ship_cruise_speed = ship_max_speed*np.sign(random.random()-0.5)
            random_ship_cruise_turn_rate = 0
            random_ship_cruise_timesteps = random.randint(round(max_cruise_seconds/delta_time), round(max_cruise_seconds/delta_time))
            maneuver_length, next_imminent_collision_time, safe_time_after_maneuver, move_sequence, state_sequence = self.simulate_maneuver(ship_state, game_state, relevant_asteroids, safe_time_threshold, random_ship_heading_angle, random_ship_accel_turn_rate, random_ship_cruise_speed, random_ship_cruise_turn_rate, random_ship_cruise_timesteps, math.inf)
            
            if next_imminent_collision_time > best_imminent_collision_time_found:
                debug_print(f"Alright we found a better one with time {next_imminent_collision_time}")
                best_imminent_collision_time_found = next_imminent_collision_time
                best_maneuver_move_sequence = move_sequence
                best_maneuver_state_sequence = state_sequence
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
        debug_print(f"Did {search_iterations_count} search iterations to find a respawn maneuver where we're safe for {best_safe_time_after_maneuver}s afterwards. Moving to coordinates {best_maneuver_state_sequence[-1]['position'][0]} {best_maneuver_state_sequence[-1]['position'][1]} at timestep {best_maneuver_state_sequence[-1]['timestep']}")

        # Enqueue the respawn maneuver
        #print(best_maneuver_move_sequence)
        for move in best_maneuver_move_sequence:
            self.enqueue_action(move['timestep'], move['thrust'], move['turn_rate'])
        
        self.last_respawn_maneuver_timestep_range = (self.current_timestep, self.current_timestep + len(best_maneuver_move_sequence)) # TODO: CHECK FOR OFF BY ONE ERROR
        self.last_respawn_invincible_timestep_range = (self.current_timestep, self.current_timestep + 2 + round(respawn_invincibility_time/delta_time))
        #print(self.last_respawn_maneuver_timestep_range, self.last_respawn_invincible_timestep_range)
        return True

    def check_mine_opportunity(self, ship_state, game_state):
        average_asteroid_density = len(game_state['asteroids'])/(game_state['map_size'][0]*game_state['map_size'][1])
        average_asteroids_inside_blast_radius = average_asteroid_density*math.pi*mine_blast_radius**2
        mine_ast_count = count_asteroids_in_mine_blast_radius(game_state, game_state['asteroids'], ship_state['position'][0], ship_state['position'][1], round(mine_fuse_time/delta_time))
        debug_print(f"Mine count inside: {mine_ast_count} compared to average density amount inside: {average_asteroids_inside_blast_radius}")
        if (len(game_state['asteroids']) > 50 and mine_ast_count > 1.5*average_asteroids_inside_blast_radius or mine_ast_count > 25 or mine_ast_count > 2*average_asteroids_inside_blast_radius) and len(game_state['mines']) == 0:
            self.enqueue_action(self.current_timestep, None, None, None, True)

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        thrust_default, turn_rate_default, fire_default, drop_mine_default = 0, 0, False, False
        # Method processed each time step by this controller.
        self.current_timestep += 1
        debug_print(f"Timestep {self.current_timestep}, ship is at {ship_state['position'][0]} {ship_state['position'][1]}")
        if not self.init_done:
            self.finish_init(game_state, ship_state)
            self.init_done = True
        

        #if game_state['bullets']:
        #    debug_print(game_state['bullets'])
        #if game_state['mines']:
        #    debug_print(game_state['mines'])
        #print("thrust is " + str(thrust) + "\n" + "turn rate is " + str(turn_rate) + "\n" + "fire is " + str(fire) + "\n")
        #print(ship_state["velocity"])
        #print(ship_state['position'])
        if ship_state['is_respawning']:
            #print("WE ARE RESPAWNING. Timestep range is:")
            #print(self.last_respawn_invincible_timestep_range)
            pass
        if ship_state['is_respawning'] and not (self.last_respawn_invincible_timestep_range[0] <= self.current_timestep <= self.last_respawn_invincible_timestep_range[1]):
            debug_print("OUCH WE LOST A LIFE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #print(ship_state)
            # Clear the move queue, since previous moves have been invalidated by us taking damage
            self.action_queue = []
            self.plan_respawn_maneuver_to_safety(ship_state, game_state)
        else:
            pass
            #print(ship_state)
        if not (self.last_respawn_maneuver_timestep_range[0] <= self.current_timestep <= self.last_respawn_maneuver_timestep_range[1]):
            # We're not in the process of doing our respawn maneuver
            # Nothing's in the action queue. Evaluate the current situation and figure out the best course of action
            if not self.action_queue:
                #print("Nothing's in the queue. Plan more actions.")
                planned_maneuvers = self.plan_maneuver(ship_state, game_state)
            else:
                planned_maneuvers = True
            #print(len(self.action_queue))
            if planned_maneuvers:
                #print("We've gotta dodge asteroids. We can still shoot them while we're dodging, if a shot happens to line up.")
                #self.check_convenient_shots(ship_state, game_state)
                pass
            elif not self.action_queue:
                # We didn't plan any maneuvering and nothing's in the action queue. We can use this time to do targetting
                #print("We can chill at this spot for a bit. Entering targetting mode.")
                self.plan_targetting(ship_state, game_state)
            if ship_state['mines_remaining'] > 0:
                pass
                #self.check_mine_opportunity(ship_state, game_state)
        
        #print("All asteroids")
        #print(game_state['asteroids'])
        #print("Asteroids with midair bullets toward them")
        #print(self.asteroids_pending_death)
        #print("Forecasted asteroids")
        #print(self.forecasted_asteroid_splits)

        # Prune out the list of asteroids we shot at if the timestep (key) is in the past
        self.asteroids_pending_death = {timestep: asteroids for timestep, asteroids in self.asteroids_pending_death.items() if timestep > self.current_timestep}

        # Maintain the list of projected split asteroids by advancing the position, decreasing the timestep, and facilitate removal
        new_forecasted_asteroids = []
        for forecasted_asteroid in self.forecasted_asteroid_splits:
            if forecasted_asteroid['timesteps_until_appearance'] > 1:
                new_a = {
                    'position': (forecasted_asteroid['position'][0] + forecasted_asteroid['velocity'][0]*delta_time, forecasted_asteroid['position'][1] + forecasted_asteroid['velocity'][1]*delta_time),
                    'velocity': forecasted_asteroid['velocity'],
                    'size': forecasted_asteroid['size'],
                    'mass': forecasted_asteroid['mass'],
                    'radius': forecasted_asteroid['radius'],
                    'timesteps_until_appearance': forecasted_asteroid['timesteps_until_appearance'] - 1,
                }
                new_forecasted_asteroids.append(new_a)
        self.forecasted_asteroid_splits = new_forecasted_asteroids
        # Execute the actions already in the queue for this timestep
        #print("Forecasted splits:")
        #print(self.forecasted_asteroid_splits)
        #print("Actual asteroids:")
        #print(game_state['asteroids'])
        # Initialize defaults. If a component of the action is missing, then the default value will be returned
        thrust_combined, turn_rate_combined, fire_combined, drop_mine_combined = thrust_default, turn_rate_default, fire_default, drop_mine_default

        while self.action_queue and self.action_queue[0][0] == self.current_timestep:
            #print("Stuff is in the queue!")
            _, _, thrust, turn_rate, fire, drop_mine = heapq.heappop(self.action_queue)
            thrust_combined = thrust if thrust is not None else thrust_combined
            turn_rate_combined = turn_rate if turn_rate is not None else turn_rate_combined
            fire_combined = fire if fire is not None else fire_combined
            drop_mine_combined = drop_mine if drop_mine is not None else drop_mine_combined
        if fire_combined == True:
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
        #time.sleep(0.2)
        #print(game_state, ship_state)
        #drop_mine_combined = random.random() < 0.01
        #print(game_state)

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
        #time.sleep(0.1)
        #print('Asteroids killed this timestep:')
        for a in self.previous_asteroids_list:
            a['position'] = (a['position'][0] + a['velocity'][0]*delta_time, a['position'][1] + a['velocity'][1]*delta_time)
        
        #print(missing_elements(self.previous_asteroids_list, game_state['asteroids']))
        self.previous_asteroids_list = game_state['asteroids']
        state_dump_dict = {
            'timestep': self.current_timestep,
            'ship_state': ship_state,
            'asteroids': game_state['asteroids'],
            'bullets': game_state['bullets'],
        }
        if reality_state_dump:
            append_dict_to_file(state_dump_dict, 'Reality State Dump.txt')
        #print(mine_ast_count)
        if len(game_state['mines']) > 0:
            pass
            #print(game_state['mines'])
            #print(ship_state)
        return thrust_combined, turn_rate_combined, fire_combined, drop_mine_combined

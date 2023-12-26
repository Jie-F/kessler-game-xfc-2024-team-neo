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
import timeit

debug_mode = False
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

width = 1920
height = 1080
dummy_game_state = {'asteroids': [], 'map_size': (width, height)}

def check_coordinate_bounds(game_state, x, y):
    if 0 <= x < game_state['map_size'][0] and 0 <= y < game_state['map_size'][1]:
        return True
    else:
        return False

def angle_difference_rad(angle1, angle2):
    # Calculate the raw difference
    raw_diff = angle1 - angle2

    # Adjust for wraparound using modulo
    adjusted_diff = raw_diff % (2*math.pi)

    # If the difference is greater than pi, adjust to keep within -pi to pi
    if adjusted_diff > math.pi:
        adjusted_diff -= 2*math.pi

    return adjusted_diff

def ast_to_string(a):
    return f"Pos: ({a['position'][0]:0.2f}, {a['position'][1]:0.2f}), Vel: ({a['velocity'][0]:0.2f}, {a['velocity'][1]:0.2f}), Size: {a['size']}"


def alternative_intercept_calc(a, ship_state, game_state, timesteps_until_can_fire=0):
    t_0 = 1 - ship_radius/bullet_speed + bullet_length/2/bullet_speed # The bullet's head originates from the edge of the ship's radius. We want to set the position of the bullet to the center of the bullet, so we have to do some fanciness here
    ax = a['position'][0]
    ay = a['position'][1]
    avx = a['velocity'][0]
    avy = a['velocity'][1]
    vb = bullet_speed
    theta_0 = math.radians(ship_state['heading'])
    def intercept_theta_are_zeros_of_this_function(theta):
        # Domain of this function is theta_0 - pi to theta_0 + pi
        assert theta_0 - math.pi <= theta <= theta_0 + math.pi
        abs_delta_theta = abs(theta - theta_0)
        return (t_0*avx - avx*(1 - abs_delta_theta/math.pi) + ax)*(math.sin(theta) - avy/vb) - (t_0*avy - avy*(1 - abs_delta_theta/math.pi) + ay)*(math.cos(theta) - avx/vb)

    def derivative_of_intercept_theta_are_zeros_of_this_function(theta):
        # Domain of this function is theta_0 - pi to theta_0 + pi
        assert theta_0 - math.pi <= theta <= theta_0 + math.pi
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        abs_delta_theta = abs(theta - theta_0)
        if theta != theta_0:
            term1 = (-avx*sin_theta + avy*cos_theta)*(theta - theta_0)
            term2 = ((avx*(abs_delta_theta - math.pi) + math.pi*(avx*t_0 + ax))*cos_theta)
            term3 = ((avy*(abs_delta_theta - math.pi) + math.pi*(avy*t_0 + ay))*sin_theta)
            derivative = (-term1 + (term2 + term3)*abs_delta_theta) / (math.pi*abs_delta_theta)
        else:
            # The derivative at theta = theta_0 is just the average of the limits of the derivative on both sides, averaged
            derivative = (avx*(t_0 - 1) + ax)*math.cos(theta_0) + (avy*(t_0 - 1) + ay)*math.sin(theta_0)
        return derivative

    def interception_time(intercept_theta):
        sin_intercept_theta = math.sin(intercept_theta)
        cos_intercept_theta = math.cos(intercept_theta)
        return (ax*sin_intercept_theta - ay*cos_intercept_theta)/(avy*cos_intercept_theta - avx*sin_intercept_theta)

    def interception_distance(intercept_theta, intercept_time):
        return vb*((math.pi - abs(intercept_theta - theta_0))/math.pi + intercept_time - t_0)

    def plot_function(a, ship_state, game_state, timesteps_until_can_fire):
        theta_0 = math.radians(ship_state['heading'])
        theta_range = np.linspace(theta_0 - math.pi, theta_0 + math.pi, 400)
        
        # Vectorize the functions for numpy compatibility
        vectorized_function = np.vectorize(intercept_theta_are_zeros_of_this_function)
        vectorized_derivative = np.vectorize(derivative_of_intercept_theta_are_zeros_of_this_function)

        # Calculate function values
        function_values = vectorized_function(theta_range)
        derivative_values = vectorized_derivative(theta_range)

        plt.figure(figsize=(12, 6))

        # Plot the function
        plt.subplot(1, 2, 1)
        plt.plot(theta_range, function_values, label="Function")
        plt.xlabel("Theta")
        plt.ylabel("Function Value")
        plt.title("Function Plot")
        plt.grid(True)
        plt.legend()

        # Plot the derivative
        plt.subplot(1, 2, 2)
        plt.plot(theta_range, derivative_values, label="Derivative", color="orange")
        plt.xlabel("Theta")
        plt.ylabel("Derivative Value")
        plt.title("Derivative Plot")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

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
    if spam: print(f"\nGETTING FEASIBLE INTERCEPT FOR ASTEROID {ast_to_string(a)}, THIS IS A BIG HONKER OF A FUNCTION")
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
                if spam: print(f"Converged. Aiming timesteps required is {turning_timesteps} which was the same as the previous iteration")
            elif turning_timesteps > new_aiming_timesteps_required:
                # Wacky oscillation case
                if spam: print(f"WACKY OSCILLATION CASE WHERE aiming_timesteps_required ({turning_timesteps}) > new_aiming_timesteps_required ({new_aiming_timesteps_required})")
            # Found an answer
            if not aiming_timesteps_required:
                # Only set the first answer
                aiming_timesteps_required = turning_timesteps
                if not debug:
                    break
        else:
            if spam: print(f"Not converged yet. aiming_timesteps_required ({turning_timesteps}) != new_aiming_timesteps_required ({new_aiming_timesteps_required})")
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
    if spam: print("Final answer is:")
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



def generate_random_asteroid(pos_range_x, pos_range_y, vel_range, radius_values):
    position = (random.uniform(*pos_range_x), random.uniform(*pos_range_y))
    velocity = (random.uniform(*vel_range), random.uniform(*vel_range))
    radius = random.choice(radius_values)
    return {'position': position, 'velocity': velocity, 'radius': radius}

def generate_random_ship_state(pos_range_x, pos_range_y, vel_range, speed_range, heading_range):
    position = (random.uniform(*pos_range_x), random.uniform(*pos_range_y))
    velocity = (random.uniform(*vel_range), random.uniform(*vel_range))
    speed = random.uniform(*speed_range)
    heading = random.uniform(*heading_range)
    return {'position': position, 'velocity': velocity, 'speed': speed, 'heading': heading}

def generate_test_data(num_sets, pos_range_x, pos_range_y, vel_range, radius_values, speed_range, heading_range, timestep_range):
    test_data = []
    for _ in range(num_sets):
        asteroid = generate_random_asteroid(pos_range_x, pos_range_y, vel_range, radius_values)
        ship_state = generate_random_ship_state(pos_range_x, pos_range_y, (0, 0), speed_range, heading_range)
        timesteps_until_can_fire = random.randint(*timestep_range)
        test_data.append((asteroid, ship_state, dummy_game_state, timesteps_until_can_fire))
    return test_data

def benchmark_function(function, test_data, number=1):
    total_time = 0
    for data in test_data:
        stmt = lambda: function(*data)
        time = timeit.timeit(stmt, number=number)
        total_time += time
    return total_time

def main():
    num_sets = 10000  # Adjust this based on your needs
    pos_range_x = (-width, 2*width)  # Example range for position
    pos_range_y = (-height, 2*height)
    vel_range = (-400, 400)  # Example range for velocity
    radius_values = [8, 16, 24, 32]
    speed_range = (0, 0)  # Example range for ship speed
    heading_range = (0, 360)  # Example range for ship heading
    timestep_range = (0, 5)  # Example range for timesteps

    test_data = generate_test_data(num_sets, pos_range_x, pos_range_y, vel_range, radius_values, speed_range, heading_range, timestep_range)

    function_time = benchmark_function(get_feasible_intercept_angle_and_turn_time, test_data, number=10)

    print(f"Function Execution Time: {function_time}")

if __name__ == "__main__":
    main()

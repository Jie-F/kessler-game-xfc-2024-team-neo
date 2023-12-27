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


def solve_interception(asteroid, ship_state, game_state, timesteps_until_can_fire: int=0):
    # The bullet's head originates from the edge of the ship's radius.
    # We want to set the position of the bullet to the center of the bullet, so we have to do some fanciness here so that at t=0, the bullet's center is where it should be
    t_0 = (ship_radius - bullet_length/2)/bullet_speed
    # Positions are relative to the ship. We set the origin to the ship's position. Remember to translate back!
    origin_x = ship_state['position'][0]
    origin_y = ship_state['position'][1]
    ax = asteroid['position'][0] - origin_x
    ay = asteroid['position'][1] - origin_y
    avx = asteroid['velocity'][0]
    avy = asteroid['velocity'][1]
    vb = bullet_speed
    tr = math.radians(ship_max_turn_rate) # rad/s
    vb_sq = vb*vb
    theta_0 = math.radians(ship_state['heading'])

    def naive_desired_heading_calc(timesteps_until_can_fire: int=0):
        time_until_can_fire_s = timesteps_until_can_fire*delta_time
        ax_delayed = ax + time_until_can_fire_s*avx # We add a delay to account for the timesteps until we can fire delay
        ay_delayed = ay + time_until_can_fire_s*avy
        A = avx*avx + avy*avy - vb_sq
        B = 2*(ax_delayed*avx + ay_delayed*avy - vb_sq*t_0)
        C = ax_delayed*ax_delayed + ay_delayed*ay_delayed - vb_sq*t_0*t_0
        D = B*B - 4*A*C

        positive_interception_times = []
        t1, t2 = None, None
        if D > 0:
            t1 = (-B - math.sqrt(D))/(2*A)
            t2 = (-B + math.sqrt(D))/(2*A)
        elif D == 0:
            t1 = -B/(2*A)
        if t1 and t1 >= 0:
            positive_interception_times.append(t1)
        if t2 and t2 >= 0:
            positive_interception_times.append(t2)
        solutions = []
        for t in positive_interception_times:
            x = ax_delayed + t*avx
            y = ay_delayed + t*avy
            theta = math.atan2(y, x)
            # If the asteroid is out of bounds, then it will wrap around and this shot isn't feasible
            # However, if an unwrapped asteroid was passed into this function and the interception is inbounds, then it's a feasible shot
            intercept_x = x + origin_x
            intercept_y = y + origin_y
            solutions.append((t + time_until_can_fire_s, angle_difference_rad(theta, theta_0), timesteps_until_can_fire, None, intercept_x, intercept_y, None))
        return solutions

    def intercept_theta_are_zeros_of_this_function(theta):
        # Domain of this function is theta_0 - pi to theta_0 + pi
        # Make this function periodic by wrapping inputs outside this range, to within the range
        if not (theta_0 - math.pi <= theta <= theta_0 + math.pi):
            theta = (theta - theta_0 + math.pi)%(2*math.pi) - math.pi + theta_0
        #assert theta_0 - math.pi <= theta <= theta_0 + math.pi
        abs_delta_theta = abs(theta - theta_0)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        #return (t_0*avx - avx*(1 - abs_delta_theta/math.pi) + ax)*(math.sin(theta) - avy/vb) - (t_0*avy - avy*(1 - abs_delta_theta/math.pi) + ay)*(math.cos(theta) - avx/vb)
        return (avx - vb*cos_theta)*(vb*t_0*sin_theta - ay - avy*abs_delta_theta/tr) - (avy - vb*sin_theta)*(vb*t_0*cos_theta - ax - avx*abs_delta_theta/tr)
        # (avx - vb*math.cos(theta))*(vb*t_0*math.sin(theta) - ay - avy*abs(theta - theta_0)/tr) - (avy - vb*math.sin(theta))*(vb*t_0*math.cos(theta) - ax - avx*abs(theta - theta_0)/tr)
    
    def derivative_of_intercept_theta_are_zeros_of_this_function(theta):
        # Domain of this function is theta_0 - pi to theta_0 + pi
        # Make this function periodic by wrapping inputs outside this range, to within the range
        if not (theta_0 - math.pi <= theta <= theta_0 + math.pi):
            theta = (theta - theta_0 + math.pi)%(2*math.pi) - math.pi + theta_0
        #assert theta_0 - math.pi <= theta <= theta_0 + math.pi
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        abs_delta_theta = abs(theta - theta_0)
        
        term1 = vb*(-avx*abs_delta_theta/tr - ax + t_0*vb*cos_theta)*cos_theta
        term2 = vb*(-avy*abs_delta_theta/tr - ay + t_0*vb*sin_theta)*sin_theta
        term3 = (avx - vb*cos_theta)*(-avy*(theta - theta_0)*np.sign(theta - theta_0)/(tr*(theta - theta_0)) + t_0*vb*cos_theta)
        term4 = (avy - vb*sin_theta)*(-avx*(theta - theta_0)*np.sign(theta - theta_0)/(tr*(theta - theta_0)) - t_0*vb*sin_theta)
        derivative = term1 + term2 + term3 - term4
        return derivative

    def alt_derivative_of_intercept_theta_are_zeros_of_this_function(theta):
        # Domain of this function is theta_0 - pi to theta_0 + pi
        # Make this function periodic by wrapping inputs outside this range, to within the range
        if not (theta_0 - math.pi <= theta <= theta_0 + math.pi):
            theta = (theta - theta_0 + math.pi)%(2*math.pi) - math.pi + theta_0
        #assert theta_0 - math.pi <= theta <= theta_0 + math.pi
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        abs_delta_theta = abs(theta - theta_0)
        
        if theta == theta_0:
            derivative = vb*(avx*t_0*math.cos(theta_0) + avy*t_0*math.sin(theta_0) - ax*math.cos(theta_0) - ay*math.sin(theta_0))
        else:
            term1 = -vb*(avx*cos_theta*abs_delta_theta + avy*sin_theta*abs_delta_theta + ax*tr*cos_theta + ay*tr*sin_theta - t_0*tr*vb)*abs_delta_theta
            term2 = (avx - vb*cos_theta)*(avy*(theta - theta_0) - t_0*tr*vb*cos_theta*abs_delta_theta)
            term3 = (avy - vb*sin_theta)*(avx*(theta - theta_0) + t_0*tr*vb*sin_theta*abs_delta_theta)
            derivative = (term1 - term2 + term3)/(tr*abs_delta_theta)
        return derivative

    def interception_time(intercept_theta):
        pass
        return

    def turbo_rootinator(initial_guess, function, derivative_function, tolerance=eps, max_iterations=5):
        # theta_new = theta_old - f(theta_old)/f'(theta_old)
        theta_old = initial_guess
        #print(f"Our initial guess is {initial_guess}")
        for iteration in range(max_iterations):
            f_value = function(theta_old)
            derivative_value = derivative_function(theta_old)
            
            # Avoid division by zero
            if derivative_value == 0:
                raise ValueError("Derivative is zero. Rootinator fails :(")

            # Update the estimate
            theta_new = theta_old - f_value/derivative_value
            #print(f"After iteration {iteration + 1}, our new theta value is {theta_new}")
            # Check for convergence
            if abs(theta_new - theta_old) < tolerance:
                return theta_new, (iteration + 1)

            theta_old = theta_new
        return None, 0

    def plot_function():
        naive_theta_ans_list = naive_desired_heading_calc(timesteps_until_can_fire)  # Assuming this function returns a list of angles
        theta_0 = math.radians(ship_state['heading'])
        theta_range = np.linspace(theta_0 - math.pi, theta_0 + math.pi, 400)
        theta_delta_range = np.linspace(-math.pi, math.pi, 400)

        # Vectorize the functions for numpy compatibility
        vectorized_function = np.vectorize(intercept_theta_are_zeros_of_this_function)
        vectorized_derivative = np.vectorize(derivative_of_intercept_theta_are_zeros_of_this_function)
        vectorized_alt_derivative = np.vectorize(alt_derivative_of_intercept_theta_are_zeros_of_this_function)

        # Calculate function values
        function_values = vectorized_function(theta_range)
        derivative_values = vectorized_derivative(theta_range)
        alt_derivative_values = vectorized_alt_derivative(theta_range)

        plt.figure(figsize=(12, 6))

        # Plot the function and its derivatives
        plt.plot(theta_delta_range, function_values, label="Function")
        plt.plot(theta_delta_range, derivative_values, label="Derivative", color="orange")
        plt.plot(theta_delta_range, alt_derivative_values, label="Alt Derivative", color="blue", linestyle=':')

        # Add vertical lines for each naive_theta_ans
        for theta_ans in naive_theta_ans_list:
            plt.axvline(x=theta_ans[1], color='yellow', linestyle='--', label=f"Naive Theta Ans at {theta_ans[1]:.2f}")

            zero, iterations = turbo_rootinator(theta_ans[1] + theta_0, intercept_theta_are_zeros_of_this_function, derivative_of_intercept_theta_are_zeros_of_this_function, 0.1)
            if zero:
                delta_theta_solution = zero - theta_0
                if not (-math.pi <= delta_theta_solution <= math.pi):
                    #print(f"SOLUTION WAS OUT OUT BOUNDS AT {delta_theta_solution} AND WRAPPED TO -pi, pi")
                    delta_theta_solution = (delta_theta_solution + math.pi)%(2*math.pi) - math.pi
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
    naive_solutions = naive_desired_heading_calc(timesteps_until_can_fire)
    amount_we_can_turn_before_we_can_shoot_rad = math.radians(timesteps_until_can_fire*delta_time*ship_max_turn_rate)
    for naive_solution in naive_solutions:
        if abs(naive_solution[1]) <= amount_we_can_turn_before_we_can_shoot_rad + eps:
            # The naive solution works because there's no turning delay
            #print('Naive solution works!')
            valid_solutions.append((True, math.degrees(naive_solution[1]), timesteps_until_can_fire, naive_solution[0], naive_solution[4], naive_solution[5], None))
        else:
            # Use more advanced solution
            #print('Using more advanced root finder')
            sol, _ = turbo_rootinator(naive_solution[1] + theta_0, intercept_theta_are_zeros_of_this_function, derivative_of_intercept_theta_are_zeros_of_this_function, 0.01)
            if not sol:
                continue
            delta_theta_solution = sol - theta_0
            if not (-math.pi <= delta_theta_solution <= math.pi):
                #print(f"SOLUTION WAS OUT OUT BOUNDS AT {delta_theta_solution} AND WRAPPED TO -pi, pi")
                delta_theta_solution = (delta_theta_solution + math.pi)%(2*math.pi) - math.pi
            # Check validity of solution to make sure time is positive and stuff
            delta_theta_solution_deg = math.degrees(delta_theta_solution)
            t_rot = abs(delta_theta_solution_deg)/ship_max_turn_rate
            # Intuitively, the following equation divides the distance between the bullet's initial position and the asteroid's interception point, by the velocity of the bullet relative to the asteroid, and this gives the interception time
            # It only does this for either x or y component, but it's really the same.

            t_bullet_1 = (ax + avx*t_rot - vb*t_0*math.cos(sol))/(vb*math.cos(sol) - avx)
            t_bullet_2 = (ay + avy*t_rot - vb*t_0*math.sin(sol))/(vb*math.sin(sol) - avy)
            if not math.isclose(t_bullet_1, t_bullet_2, abs_tol=0.1):
                print('\nBAD:')
                print(f"t_bullet_1: {t_bullet_1} with denom {(avx - vb*math.cos(sol))}, t_bullet_2: {t_bullet_2} with denom {(avy - vb*math.sin(sol))}")
                print(f"({ax} + {avx} * {t_rot} - {vb} * {t_0} * cos({sol}))/({vb} * cos({sol}) - {avx})")
                print(f"({ay} + {avy} * {t_rot} - {vb} * {t_0} * sin({sol}))/({vb} * sin({sol}) - {avy})")

            

            #assert math.isclose(t_bullet_1, t_bullet_2, abs_tol=0.5)
            t_bullet = t_bullet_2
            t_total = t_rot + t_bullet
            intercept_x = origin_x + vb*math.cos(sol)*(t_bullet + t_0)
            intercept_y = origin_y + vb*math.sin(sol)*(t_bullet + t_0)

            feasible = check_coordinate_bounds(game_state, intercept_x, intercept_y)
            if feasible:
                # Since half timesteps don't exist, we need to discretize this solution by rounding up the amount of timesteps, and now we can use the naive method to confirm and get the exact angle
                t_rot_ts = math.ceil(t_rot/delta_time)
                #valid_solutions.append((True, delta_theta_solution_deg, t_rot_ts, None, intercept_x, intercept_y, None))
                proper_discretized_solutions = naive_desired_heading_calc(t_rot_ts)
                for disc_sol in proper_discretized_solutions:
                    # Only expecting there to be one
                    if not abs(math.degrees(disc_sol[1])) - eps <= t_rot_ts*delta_time*ship_max_turn_rate:
                        print(f"About to fail assertion! degrees required to turn: {abs(math.degrees(disc_sol[1]))} isn't at most the amount we can rotate: {t_rot_ts*delta_time*ship_max_turn_rate}")
                        print('New sol:')
                        print((True, math.degrees(disc_sol[1]), t_rot_ts, disc_sol[0], disc_sol[4], disc_sol[5], None))
                        print('Old continuous sol:')
                        print((feasible, delta_theta_solution_deg, t_rot/delta_time, t_total, intercept_x, intercept_y, None))
                    if not t_rot_ts == disc_sol[2]:
                        print(f"About to fail assertion! TS we need to rotate ceiled: {t_rot_ts} isn't equal to our discretized solution of {disc_sol[2]}")
                        print((True, math.degrees(disc_sol[1]), t_rot_ts, disc_sol[0], disc_sol[4], disc_sol[5], None))
                    #assert abs(math.degrees(disc_sol[1])) - eps <= t_rot_ts*delta_time*ship_max_turn_rate
                    
                    #assert t_rot_ts == disc_sol[2]
                    valid_solutions.append((True, math.degrees(disc_sol[1]), t_rot_ts, disc_sol[0], disc_sol[4], disc_sol[5], None))
            else:
                pass
                #print(f"The coordinates of {intercept_x}, {intercept_y} are outside the bounds! Invalid solution. We'd have to turn this many ts: {t_rot/delta_time}")

    sorted_solutions = sorted(valid_solutions, key=lambda x: x[2])
    if sorted_solutions:
        return sorted_solutions[0]
    else:
        return False, None, None, None, None, None, None
    # return feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception

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
    # Lastly, find the difference between firing angle and the ship's current orientation.
    shot_heading = math.atan2(intercept_y - ship_pos_y, intercept_x - ship_pos_x)
    shot_heading_error_rad = angle_difference_rad(shot_heading, math.radians(ship_heading))
    
    #if not feasible:
    #    print(f"For future timesteps {future_shooting_timesteps}, the coordinates aren't feasible: {intercept_x}, {intercept_y}, aiming ts {abs(math.degrees(shot_heading_error_rad))/6}")
    
    


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
    num_sets = 1000  # Adjust this based on your needs
    pos_range_x = (-width, 2*width)  # Example range for position
    pos_range_y = (-height, 2*height)
    vel_range = (-400, 400)  # Example range for velocity
    radius_values = [8, 16, 24, 32]
    speed_range = (0, 0)  # Example range for ship speed
    heading_range = (0, 360)  # Example range for ship heading
    timestep_range = (0, 5)  # Example range for timesteps

    randseed = random.randint(1, 1000)
    print(f'Using seed {randseed}')
    random.seed(randseed)#315 diff ans!!! TODO
    nonzero_ans = False
    while not nonzero_ans:
    #for i in range(2):
        test_data = generate_test_data(num_sets, pos_range_x, pos_range_y, vel_range, radius_values, speed_range, heading_range, timestep_range)
        
        asteroid = test_data[0][0]
        ship_state = test_data[0][1]
        game_state = test_data[0][2]
        timesteps_until_can_fire=0
        #print("\nfeasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception")
        #print('Old solution:')
        #print(calculate_interception(ship_pos_x, ship_pos_y, asteroid_pos_x, asteroid_pos_y, asteroid_vel_x, asteroid_vel_y, asteroid_r, ship_heading, game_state, future_shooting_timesteps=0))
        #print(f"feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception")
        #old_ans = get_feasible_intercept_angle_and_turn_time(asteroid, ship_state, game_state, timesteps_until_can_fire)
        #print(old_ans)
        #print('New solution:')
        #new_ans = alternative_intercept_calc(asteroid, ship_state, game_state, timesteps_until_can_fire)
        #print(new_ans)
        #if old_ans[0] != False:
        #nonzero_ans = old_ans[0]
        function_time = benchmark_function(get_feasible_intercept_angle_and_turn_time, test_data, number=10)
        print(f"Old Function Execution Time: {function_time}")
        function_time = benchmark_function(solve_interception, test_data, number=10)
        print(f"New Function Execution Time: {function_time}")

if __name__ == "__main__":
    main()

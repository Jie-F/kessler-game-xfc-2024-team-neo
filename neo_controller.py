import random
from src.kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
from collections import deque
import heapq


delta_time = 1/30 # s/ts
ship_max_turn_rate = 180.0 # deg/s
ship_max_thrust = 480.0 # px/s^2
ship_drag = 80.0 # px/s^2
ship_max_speed = 240.0 # px/s
eps = 0.00000001
ship_radius = 20.0 # px
timesteps_until_terminal_velocity = math.ceil(ship_max_speed/(ship_max_thrust - ship_drag)/delta_time)

def angle_difference_rad(angle1, angle2):
    # Calculate the raw difference
    raw_diff = angle1 - angle2

    # Adjust for wraparound using modulo
    adjusted_diff = raw_diff % (2 * math.pi)

    # If the difference is greater than pi, adjust to keep within -pi to pi
    if adjusted_diff > math.pi:
        adjusted_diff -= 2 * math.pi

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
    # Ensure current and target headings are within 0 to 360 degrees
    current_heading %= 360
    target_heading %= 360

    # Calculate the differences for both direct and reverse alignment
    direct_diff = angle_difference_deg(current_heading, target_heading)
    reverse_diff = angle_difference_deg(current_heading, (target_heading + 180) % 360)

    # Choose the one with the minimum absolute rotation
    if abs(direct_diff) < abs(reverse_diff):
        return direct_diff, 1
    else:
        return reverse_diff, -1

def ship_sim_update(ship_distance, ship_speed, thrust):
    # This was totally just taken from the Kessler game code lul
    drag_amount = ship_drag * delta_time
    if drag_amount > abs(ship_speed):
        ship_speed = 0
    else:
        ship_speed -= drag_amount * np.sign(ship_speed)
    ship_speed += thrust * delta_time
    if ship_speed > ship_max_speed:
        ship_speed = ship_max_speed
    elif ship_speed < -ship_max_speed:
        ship_speed = -ship_max_speed
    ship_distance += ship_speed * delta_time
    return ship_distance, ship_speed

def build_thrust_lookup_table(max_forward_thrust_time_limit=round(5/delta_time)):
    lookup = []

    for max_forward_thrust_time in range(1, max_forward_thrust_time_limit + 1):
        # For each amount of time to hold max thrust for, re-do the simulation
        ship_distance = 0
        ship_speed = 0
        for _ in range(max_forward_thrust_time):
            # Thrust forward for max_forward_thrust_time timesteps
            ship_distance, ship_speed = ship_sim_update(ship_distance, ship_speed, ship_max_thrust)
        # Now slow down the ship until it is stationary again
        # Keep slowing down the ship until the point where in the next timestep, we can get down to 0
        deceleration_timestep_count = 0
        while ship_speed > (ship_max_thrust + ship_drag) * delta_time:
            ship_distance, ship_speed = ship_sim_update(ship_distance, ship_speed, -ship_max_thrust)
            deceleration_timestep_count += 1
        # One timestep left until we get down to 0 speed! The magnitude of thrust required is up to -ship_max_thrust but is pretty much always less
        #print(f"Before final thrust, the ship speed is {ship_speed}")
        if ship_speed > eps:
            final_thrust = max(ship_speed / delta_time - ship_drag, 0)
            ship_distance, ship_speed = ship_sim_update(ship_distance, ship_speed, -final_thrust)
            deceleration_timestep_count += 1
        assert(abs(ship_speed) < eps)
        final_thrust = round(final_thrust, 4)
        #print(f"Dist: {ship_distance}, accel timesteps: {max_forward_thrust_time}, decel timesteps: {deceleration_timestep_count}, final decel: {final_thrust}")
        # Lookup table has tuples (distance, accel timesteps, total decel timesteps, final decel rate)
        lookup.append((ship_distance, max_forward_thrust_time, deceleration_timestep_count, final_thrust))
    return lookup

def check_collision(a_x, a_y, a_r, b_x, b_y, b_r):
    collision_fudge_factor = 1.01
    if (a_x - b_x)**2 + (a_y - b_y)**2 <= collision_fudge_factor*(a_r + b_r)**2:
        return True
    else:
        return False

def collision_prediction(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_radius, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius):
    # https://stackoverflow.com/questions/11369616/circle-circle-collision-prediction/
    # Note that if an asteroid is inside the ship, the predicted collision time will correspond to the time the asteroid LEAVES the ship, not hits
    Oax, Oay = ship_pos_x, ship_pos_y
    Dax, Day = ship_vel_x, ship_vel_y
    Obx, Oby = ast_pos_x, ast_pos_y
    Dbx, Dby = ast_vel_x, ast_vel_y
    ra = ship_radius
    rb = ast_radius
    a = Dax**2 + Dbx**2 + Day**2 + Dby**2 - (2*Dax*Dbx) - (2*Day*Dby)
    b = (2*Oax*Dax) - (2*Oax*Dbx) - (2*Obx*Dax) + (2*Obx*Dbx) + (2*Oay*Day) - (2*Oay*Dby) - (2*Oby*Day) + (2*Oby*Dby)
    c = Oax**2 + Obx**2 + Oay**2 + Oby**2 - (2*Oax*Obx) - (2*Oay*Oby) - (ra + rb)**2
    d = b**2 - 4*a*c
    if (a != 0) and (d >= 0):
        t1 = (-b + math.sqrt(d)) / (2*a)
        t2 = (-b - math.sqrt(d)) / (2*a)
    else:
        t1 = math.nan
        t2 = math.nan
    #print(t1, t2)
    return t1, t2

def predict_next_imminent_collision_time_with_asteroid(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius):
    next_imminent_collision_time = math.inf
    t1, t2 = collision_prediction(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius)
    if not math.isnan(t1) and t1 >= 0:
        next_imminent_collision_time = min(t1, next_imminent_collision_time)
    if not math.isnan(t2) and t2 >= 0:
        next_imminent_collision_time = min(t2, next_imminent_collision_time)
    return next_imminent_collision_time

def duplicate_asteroids_for_wraparound(asteroid, max_x, max_y, pattern='half_surround'):
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
                if (col, row) != (0, 0):
                    continue
            #print(f"It: col{col} row{row}")
            duplicate = asteroid.copy()
            duplicate['dx'] = dx
            duplicate['dy'] = dy
            duplicate["position"] = (orig_x + dx, orig_y + dy)
            
            duplicates.append(duplicate)
    return duplicates

def find_best_asteroid(game_state, ship_state, shot_at_asteroids, time_to_simulate = 4.0):
    game_state['map_size']
    asteroids = game_state['asteroids']
    closest_asteroid = None
    ship_x = ship_state['position'][0]
    ship_y = ship_state['position'][1]
    min_x = 0
    min_y = 0
    max_x = game_state['map_size'][0]
    max_y = game_state['map_size'][1]

    def check_coordinate_bounds(x, y, dx=0, dy=0):
        if min_x <= x - dx*0 <= max_x and min_y <= y - dy*0 <= max_y:
            return True
        else:
            return False

    def check_intercept_bounds(candidate_asteroid, new_ship_heading, additional_future_timesteps = 0):
        bullet_t, _, intercept_x, intercept_y = calculate_interception(ship_x + (1 + additional_future_timesteps) * ship_state['velocity'][0], ship_y + (1 + additional_future_timesteps) * ship_state['velocity'][1], candidate_asteroid["position"][0] + (2 + additional_future_timesteps) * delta_time * candidate_asteroid["velocity"][0], candidate_asteroid["position"][1] + (2 + additional_future_timesteps) * delta_time * candidate_asteroid["velocity"][1], candidate_asteroid["velocity"][0], candidate_asteroid["velocity"][1], new_ship_heading, additional_future_timesteps * delta_time)
        if 'dx' in candidate_asteroid and 'dy' in candidate_asteroid:
            bounds_check = check_coordinate_bounds(intercept_x, intercept_y, candidate_asteroid['dx'], candidate_asteroid['dy'])
        else:
            bounds_check = check_coordinate_bounds(intercept_x, intercept_y)
        return bounds_check and bullet_t + additional_future_timesteps * delta_time >= 0
        

    def check_intercept_feasibility(candidate_asteroid):
        if not check_intercept_bounds(candidate_asteroid, ship_state['heading']):
            # If we shoot at this instant and it'll still be out of bounds, don't bother
            return False, 10000000
        # Intercept is within bounds, but only guaranteed if we shoot at this instant!
        # Check whether we have enough time to turn our camera to aim at it, and still intercept it within bounds.
        # If we can't turn in one second far enough to shoot it before it goes off screen, don't even bother dude
        timesteps_from_now = 0.0 # We allow fractional timesteps
        iterations = 1
        max_bullet_time = 200
        ship_heading = ship_state["heading"]
        ship_vel_x = ship_state['velocity'][0]
        ship_vel_y = ship_state['velocity'][1]
        for it in range(iterations):
            if timesteps_from_now > max_bullet_time:
                # Fuggetaboutit, we ain't hunting down this asteroid until it loops around at least
                return False, 10000000
            _, shooting_theta, _, _ = calculate_interception(ship_x + timesteps_from_now*ship_vel_x, ship_y + timesteps_from_now*ship_vel_y, candidate_asteroid["position"][0] + (2 + timesteps_from_now) * delta_time * candidate_asteroid["velocity"][0], candidate_asteroid["position"][1] + (2 + timesteps_from_now) * delta_time * candidate_asteroid["velocity"][1], candidate_asteroid["velocity"][0], candidate_asteroid["velocity"][1], ship_heading, timesteps_from_now * delta_time)
            
            shooting_theta_deg = shooting_theta * 180.0 / math.pi
            shooting_theta_deg = abs(shooting_theta_deg)
            
            # Update the ship heading
            ship_heading += shooting_theta_deg
            # Keep the angle within (-180, 180)
            while ship_heading > 360:
                ship_heading -= 360.0
            while ship_heading < 0:
                ship_heading += 360.0
            #print(f"Ship heading: {ship_heading}")
            number_of_timesteps_itll_take_to_turn = shooting_theta_deg / (ship_max_turn_rate * delta_time)
            timesteps_from_now += number_of_timesteps_itll_take_to_turn
            if number_of_timesteps_itll_take_to_turn < 1:
                # Turning is trivial, no need to test further, it should be fine
                break
        #print(f"Checking incpt bounds with timestps from now: ceil({timesteps_from_now})")
        if check_intercept_bounds(candidate_asteroid, ship_heading, math.ceil(timesteps_from_now)):
            # If at that future time, we can shoot and still intercept the bullet...
            # Iterate once more to see whether we can hit it after we turn
            return True, timesteps_from_now
        else:
            return False, 10000000
            
    
    # We're gonna simulate the game to detect imminent collisions
    asteroids = game_state['asteroids']
    ship_radius = ship_state['radius']
    #simulated_asteroids = copy.deepcopy(asteroids)
    simulated_asteroids = []
    # Make our own copy of the asteroids list (no duplicates needed since we compute wraparound during simulation), and also add in a variable to keep track of how many timesteps we still have yet to simulate
    for a in asteroids:
        simulated_asteroids.append({
            'position': a['position'],
            'velocity': a['velocity'],
            'radius': a['radius'],
            'num_timesteps_left_to_simulate': math.ceil(time_to_simulate / delta_time)
        })
    closest_asteroid = None
    breakout_flag = False
    total_asteroids_to_simulate = len(simulated_asteroids)
    index_scale_factor = total_asteroids_to_simulate // len(asteroids)
    num_asteroids_done_simulation = 0
    while True:
        if breakout_flag:
            break
        for ind, a in enumerate(simulated_asteroids):
            # Check how far these are. If they're farther than like 200 units away from me, just advance the simulation by a LOT of timesteps, reducing accuracy where it doesn't really matter
            if not check_collision(a['position'][0], a['position'][1], a['radius'], ship_x, ship_y, ship_radius + 300):
                speedup_factor = 30 # Jump by 30 steps, which is 1 second
            else:
                speedup_factor = 1 # Only simulate a single frame for accuracy, since the asteroid is close enough to the ship that it might hit it
            
            # Check for simulation complete condition
            if simulated_asteroids[ind]['num_timesteps_left_to_simulate'] is not None:
                if simulated_asteroids[ind]['num_timesteps_left_to_simulate'] <= 0:
                    # Done simulating this asteroid
                    num_asteroids_done_simulation += 1
                    simulated_asteroids[ind]['num_timesteps_left_to_simulate'] = None
                    continue
                else:
                    simulated_asteroids[ind]['num_timesteps_left_to_simulate'] -= speedup_factor
            if num_asteroids_done_simulation == total_asteroids_to_simulate:
                breakout_flag = True
                break
            # Get the next position
            a['position'] = [pos + speedup_factor*v*delta_time for pos, v in zip(a['position'], a['velocity'])]
            
            # Wraparound the bounds
            while a['position'][0] > max_x:
                a['position'][0] -= max_x
            while a['position'][0] < 0:
                a['position'][0] += max_x
            while a['position'][1] > max_y:
                a['position'][1] -= max_y
            while a['position'][1] < 0:
                a['position'][1] += max_y
            if check_collision(a['position'][0], a['position'][1], a['radius'], ship_x, ship_y, ship_radius):
                closest_asteroid = asteroids[ind//index_scale_factor] # Asteroids list only has 1/9 the elements before duplication
                breakout_flag = True
                break
    if closest_asteroid is not None:
        return closest_asteroid

    # Target physically closest asteroid

    # Find closest angular distance so aiming is faster
    closest_asteroid = None
    closest_asteroid_angular_dist = 10000000
    closest_asteroid_total_waiting = 10000000
    closest_asteroid_regardless_of_feasibility = None
    closest_asteroid_regardless_of_feasibility_angular_dist = 10000000
    ship_heading = ship_state['heading']
    #print(f"Num asteroids: {len(asteroids)}, num shot at already: {len(shot_at_asteroids)}")
    for a_no_wraparound in asteroids:
        duplicated_asteroids = duplicate_asteroids_for_wraparound(a_no_wraparound, max_x, max_y, 'surround')
        for a in duplicated_asteroids:
            _, shooting_theta, _, _ = calculate_interception(ship_x + ship_state['velocity'][0], ship_y + ship_state['velocity'][1], a['position'][0] + 2 * delta_time * a["velocity"][0], a["position"][1] + 2 * delta_time * a["velocity"][1], a["velocity"][0], a["velocity"][1], ship_heading)
            curr_angular_dist = abs(shooting_theta)
            #print(f"Checking feasibility for hitting asteroid at ({a['position'][0]}, {a['position'][1]})")
            intercept_feasible, additional_waiting_timesteps = check_intercept_feasibility(a)
            if (a["velocity"][0], a["velocity"][1], a["radius"], a['dx'], a['dy']) not in shot_at_asteroids and intercept_feasible:
                # Prioritize based on not waiting too much time, and then by min angular distance
                if additional_waiting_timesteps + curr_angular_dist / (ship_max_turn_rate * delta_time) <= closest_asteroid_total_waiting:
                    #print(f"New best asteroid: ({a['position'][0]}, {a['position'][1]})additional wait: {additional_waiting_timesteps}, ang dist: {curr_angular_dist}")
                    closest_asteroid = a
                    closest_asteroid_angular_dist = curr_angular_dist
                    closest_asteroid_total_waiting = additional_waiting_timesteps + curr_angular_dist / (ship_max_turn_rate * delta_time)
            elif intercept_feasible:
                #print('Cant shoot at this again')
                #print(shot_at_asteroids)
                pass
            if curr_angular_dist < closest_asteroid_regardless_of_feasibility_angular_dist:
                closest_asteroid_regardless_of_feasibility = a
                closest_asteroid_regardless_of_feasibility_angular_dist = curr_angular_dist
    if closest_asteroid is None:
        #print("Yup we're aiming at a crappy one!")
        #closest_asteroid = closest_asteroid_regardless_of_feasibility
        #closest_asteroid = 
        pass
    return closest_asteroid

def calculate_interception(ship_pos_x, ship_pos_y, asteroid_pos_x, asteroid_pos_y, asteroid_vel_x, asteroid_vel_y, ship_heading, lookahead_time=0):
    # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
    # Based on Law of Cosines calculation, see notes.
    
    # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
    #    and the angle of the asteroid's current movement.
    # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
    
    asteroid_ship_x = ship_pos_x - asteroid_pos_x
    asteroid_ship_y = ship_pos_y - asteroid_pos_y
    
    asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)
    
    asteroid_direction = math.atan2(asteroid_vel_y, asteroid_vel_x) # Velocity is a 2-element array [vx,vy].
    my_theta2 = asteroid_ship_theta - asteroid_direction
    cos_my_theta2 = math.cos(my_theta2)
    # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
    asteroid_vel = math.sqrt(asteroid_vel_x**2 + asteroid_vel_y**2)
    bullet_speed = 800 # Hard-coded bullet speed from bullet.py
    
    # Discriminant of the quadratic formula b^2-4ac
    asteroid_dist = math.sqrt((ship_pos_x - asteroid_pos_x)**2 + (ship_pos_y - asteroid_pos_y)**2)
    targ_det = (-2 * asteroid_dist * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * asteroid_dist**2)
    if targ_det < 0:
        # There is no intercept. Return a fake intercept
        return 100000, 100000, -100, -100
    # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
    intercept1 = ((2 * asteroid_dist * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 - bullet_speed**2))
    intercept2 = ((2 * asteroid_dist * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2 - bullet_speed**2))
    #print(f"intercept 1: {intercept1}, intercept2: {intercept2}")
    # Take the smaller intercept time, as long as it is positive AFTER ADDING LOOKAHEAD TIME; if not, take the larger one.
    if intercept1 > intercept2:
        if intercept2 + lookahead_time >= 0:
            bullet_t = intercept2
        else:
            bullet_t = intercept1
    else:
        if intercept1 + lookahead_time >= 0:
            bullet_t = intercept1
        else:
            bullet_t = intercept2
            
    # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
    intercept_x = asteroid_pos_x + asteroid_vel_x * bullet_t
    intercept_y = asteroid_pos_y + asteroid_vel_y * bullet_t
    
    my_theta1 = math.atan2(intercept_y - ship_pos_y, intercept_x - ship_pos_x)
    
    # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
    shooting_theta = my_theta1 - ((math.pi/180)*ship_heading)

    # Wrap all angles to (-pi, pi)
    shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

    return bullet_t, shooting_theta, intercept_x, intercept_y



class NeoController(KesslerController):
    @property
    def name(self) -> str:
        return "Neo"

    def __init__(self):
        self.init_done = False
        self.current_timestep = 0
        self.previously_targetted_asteroid = None
        self.fire_on_frames = set()
        self.shot_at_asteroids = {} # Dict of tuples, with the values corresponding to the timesteps we need to wait until they can be shot at again
        self.last_time_fired = -1
        self.thrust_lookup_table = build_thrust_lookup_table()
        print(self.thrust_lookup_table)
        self.action_queue = []  # This will be our heap
        heapq.heapify(self.action_queue)  # Transform list into a heap

    def finish_init(self, game_state):
        pass

    def enqueue_action(self, timestep, thrust=None, turn_rate=None, fire=None, drop_mine=None):
        heapq.heappush(self.action_queue, (timestep, thrust, turn_rate, fire, drop_mine))

    def plan_actions(self, ship_state: Dict, game_state: Dict):
        # Simulate and look for a good move
        print("Checking for imminent danger")
        # Check for danger
        next_imminent_collision_time = math.inf
        for real_asteroid in game_state['asteroids']:
            duplicated_asteroids = duplicate_asteroids_for_wraparound(real_asteroid, game_state['map_size'][0], game_state['map_size'][1], 'half_surround')
            for a in duplicated_asteroids:
                next_imminent_collision_time = predict_next_imminent_collision_time_with_asteroid(ship_state['position'][0], ship_state['position'][1], ship_state['radius'], ship_state['velocity'][0], ship_state['velocity'][1], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'])
        #print(f"Next imminent collision is in {next_imminent_collision_time}s")
        safe_time_threshold = 10
        do_nothing_time = 5
        if next_imminent_collision_time > safe_time_threshold:
            # Nothing's gonna hit us for a long time
            print("We're safe for the next 10 seconds, so do nothing for 5 seconds")
            for timestep in range(self.current_timestep, self.current_timestep + round(do_nothing_time/delta_time)):
                self.enqueue_action(timestep)
        else:
            # Avoid!
            # Run a simulation and find a course of action to put me to safety
            current_ship_heading = ship_state['heading']
            safe_maneuver_found = False
            while not safe_maneuver_found:
                random_ship_heading_angle = random.uniform(-180.0, 180.0)
                unit_vec_x = math.cos(math.radians(random_ship_heading_angle))
                unit_vec_y = math.sin(math.radians(random_ship_heading_angle))
                heading_difference_deg, alignment = min_rotation_to_align_deg(current_ship_heading, random_ship_heading_angle)
                timesteps_it_takes_to_steer_ship_by_this_angle = math.ceil(abs(heading_difference_deg) / (ship_max_turn_rate*delta_time))
                seconds_it_takes_to_steer_ship_by_this_angle = timesteps_it_takes_to_steer_ship_by_this_angle * delta_time
                # Now we know the extra time it takes for the ship to first steer to the correct alignment, given that front and back alignment are acceptable
                # Simulate moving in this direction in different speeds/distances until we hopefully find one that gets us to safety
                for thrust_amount in self.thrust_lookup_table:
                    ship_sim_pos_x, ship_sim_pos_y = ship_state['position']
                    ship_sim_dist = 0
                    ship_sim_speed = 0
                    initial_thrust = ship_max_thrust*alignment
                    nope_this_aint_gonna_work_rip = False
                    for _ in range(thrust_amount[1]):
                        ship_sim_update(ship_sim_dist, ship_sim_speed, initial_thrust)
                        ship_sim_pos_x, ship_sim_pos_y = ship_state['position'][0] + unit_vec_x*ship_sim_dist, ship_state['position'][1] + unit_vec_y*ship_sim_dist

                        for real_asteroid in game_state['asteroids']:
                            # Check feasibility
                            if check_collision(ship_sim_pos_x, ship_sim_pos_y, ship_radius, real_asteroid['position'][0], real_asteroid['position'][1], real_asteroid['radius']):
                                nope_this_aint_gonna_work_rip = True
                                break
                        if nope_this_aint_gonna_work_rip:
                            break
                    if nope_this_aint_gonna_work_rip:
                        if thrust_amount[1] <= timesteps_until_terminal_velocity:
                            # Try the next thrust amount
                            continue
                        else:
                            # Early break condition. If we're already past terminal velocity and this isn't feasible, then any further thrust values will also be infeasible so no need to try them
                            break
                    # We're approaching the goal. Time to start decelerating
                    decel_thrust = -ship_max_thrust*alignment
                    nope_this_aint_gonna_work_rip = False
                    for _ in range(thrust_amount[2] - 1):
                        ship_sim_update(ship_sim_dist, ship_sim_speed, decel_thrust)
                        ship_sim_pos_x, ship_sim_pos_y = ship_state['position'][0] + unit_vec_x*ship_sim_dist, ship_state['position'][1] + unit_vec_y*ship_sim_dist
                        # Check feasibility
                        if check_collision(ship_sim_pos_x, ship_sim_pos_y, ship_radius, real_asteroid['position'][0], real_asteroid['position'][1], real_asteroid['radius']):
                            nope_this_aint_gonna_work_rip = True
                            break
                    if nope_this_aint_gonna_work_rip:
                        # Try the next thrust amount just in case we can dodge it by going faster
                        continue
                    final_decel_thrust = thrust_amount[3]*alignment
                    ship_sim_update(ship_sim_dist, ship_sim_speed, final_decel_thrust)
                    # Ship's final, hopefully safe, resting position:
                    ship_sim_pos_x, ship_sim_pos_y = ship_state['position'][0] + unit_vec_x*ship_sim_dist, ship_state['position'][1] + unit_vec_y*ship_sim_dist
                    assert(abs(ship_sim_pos_x - (ship_state['position'][0] + unit_vec_x*thrust_amount[0])) < eps)
                    assert(abs(ship_sim_pos_y - (ship_state['position'][1] + unit_vec_y*thrust_amount[1])) < eps)
                    next_imminent_collision_time = predict_next_imminent_collision_time_with_asteroid(ship_sim_pos_x, ship_sim_pos_y, 0, 0, )
                    if next_imminent_collision_time < safe_time_threshold:
                        # Try the next thrust amount just in case we can dodge it by going more
                        continue
                    else:
                        safe_maneuver_found = True
                        safe_maneuver_turn = heading_difference_deg
                        safe_maneuver_thrust = thrust_amount
                    # DOP NOT FORGET TO INCREASE THE TIMESTEP BY TIME IT TAKES TO TURN SHP!
                    # sdf789y89234yhr90aserhF*(N YSR()E*F GYHSEU(DFHNM(SDYNUF( USDHU(FGVsdhu80F HSD*(UFYH*()SD))))))
            # Enqueue the safe maneuver
            for timestep in range(self.current_timestep, self.current_timestep + round(do_nothing_time/delta_time)):
                self.enqueue_action(timestep)
    '''
    #print('Self:')
            #print(self)
            #print('Game state:')
            #print(game_state)
            #print('Ship state:')
            #print(ship_state)
            #print(f"Frame {self.eval_frames}")

            # Find the closest asteroid (disregards asteroid velocity)
            ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
            ship_pos_y = ship_state["position"][1]       
            ship_vel_x = ship_state["velocity"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
            ship_vel_y = ship_state["velocity"][1]
            best_asteroid = None
            
            map_size_x = game_state['map_size'][0]
            map_size_y = game_state['map_size'][1]
            #print(map_size_x, map_size_y)
            
            #if self.eval_frames % 30 == 0 or self.previously_targetted_asteroid is None:
            best_asteroid = find_closest_asteroid(game_state, ship_state, self.shot_at_asteroids)
            #    self.previously_targetted_asteroid = closest_asteroid
            # closest_asteroid now contains the nearest asteroid considering wraparound
            #print("We're targetting:")
            #print(closest_asteroid)
            if best_asteroid is not None:
                bullet_t, shooting_theta, _, _ = calculate_interception(ship_pos_x + ship_vel_x, ship_pos_y + ship_vel_y, best_asteroid["position"][0] + 2 * delta_time * best_asteroid["velocity"][0], best_asteroid["position"][1] + 2 * delta_time * best_asteroid["velocity"][1], best_asteroid["velocity"][0], best_asteroid["velocity"][1], ship_state["heading"])
            else:
                #print('WACKO CASE')
                bullet_t = 10000
                shooting_theta = 0

            shooting_theta_deg = shooting_theta * 180.0 / math.pi
            #print(f"shooting theta deg {shooting_theta_deg}")
            if shooting_theta_deg < 0:
                sign = -1
            else:
                sign = +1
            shooting_theta_deg = abs(shooting_theta_deg)
            if len(self.fire_on_frames) > 0:
                # Scheduled to fire. Wait until we're done firing before we turn.
                turn_rate = 0
            if shooting_theta_deg > ship_max_turn_rate * delta_time:
                turn_rate = sign * ship_max_turn_rate
            elif bullet_t == 10000:
                turn_rate = 0
            else:
                #print(f"Locked in. Scheduled to fire. shooting_theta_deg: {shooting_theta_deg}")
                #print(f'Scheduled to fire at asteroid at:')
                #print(closest_asteroid)
                if self.eval_frames - self.last_time_fired >= 5:
                    self.fire_on_frames.add(self.eval_frames + 1)
                else:
                    self.fire_on_frames.add(self.eval_frames + 1 + 5 - (self.eval_frames - self.last_time_fired))
                turn_rate = sign * shooting_theta_deg / delta_time
                #print(f'SNAP! Turn rate: {turn_rate}')
            # 0 1 2 3 4 5 6
            if self.eval_frames in self.fire_on_frames and not ship_state['is_respawning']:
                # We are scheduled to fire on this frame, and we aren't invincible so we can fire without losing that
                if self.eval_frames - self.last_time_fired >= 5 and best_asteroid is not None:
                    # Our firing cooldown has ran out, and we can fire. We can only shoot once every 5 frames.
                    fire = True
                    self.fire_on_frames.remove(self.eval_frames)
                    if 'dx' in best_asteroid and 'dy' in best_asteroid:
                        dx = best_asteroid['dx']
                        dy = best_asteroid['dy']
                    else:
                        dx = 0
                        dy = 0
                    if (best_asteroid["velocity"][0], best_asteroid["velocity"][1], best_asteroid["radius"], dx, dy) not in self.shot_at_asteroids:
                        self.shot_at_asteroids[(best_asteroid["velocity"][0], best_asteroid["velocity"][1], best_asteroid["radius"], dy, dx)] = math.ceil(bullet_t / delta_time)
                elif best_asteroid is not None:
                    fire = False
                    # Try to fire on the next frame
                    self.fire_on_frames.add(self.eval_frames + 1)
                    #print('WAITING UNTIL I CAN FIRE')
                else:
                    fire = False
            elif self.eval_frames in self.fire_on_frames and ship_state['is_respawning']:
                self.fire_on_frames.add(self.eval_frames + 1)
                fire = False
            else:
                fire = False
            
            # List to hold keys to be removed
            keys_to_remove = []
            # Iterate through the dictionary items
            for key, value in self.shot_at_asteroids.items():
                # Decrease each value by 1
                self.shot_at_asteroids[key] = value - 1
                # If the value hits 0, add the key to the list of keys to be removed
                if self.shot_at_asteroids[key] <= 0:
                    keys_to_remove.append(key)
            # Remove the keys from the dictionary
            for key in keys_to_remove:
                del self.shot_at_asteroids[key]
            
            # Iterate through all asteroids, and calculate the minimum angular distance to any intercept with them
            asteroids = game_state['asteroids']
            min_aim_delta = 1000
            for a in asteroids:
                ship_heading = ship_state["heading"]
                ship_x = ship_state['position'][0]
                ship_y = ship_state['position'][1]
                ship_vel_x = ship_state['velocity'][0]
                ship_vel_y = ship_state['velocity'][1]
                _, shooting_theta, _, _ = calculate_interception(ship_x + ship_vel_x, ship_y + ship_vel_y, a['position'][0] + 2 * delta_time * a["velocity"][0], a["position"][1] + 2 * delta_time * a["velocity"][1], a["velocity"][0], a["velocity"][1], ship_heading)
                shooting_theta_deg = shooting_theta * 180.0 / math.pi
                shooting_theta_deg = abs(shooting_theta_deg)
                if shooting_theta_deg < min_aim_delta:
                    min_aim_delta = shooting_theta_deg

            # Find physically closest asteroid
            closest_asteroid = None
            closest_gap_between_ship_and_closest_asteroid = 1000000000
            for a in game_state["asteroids"]:
                curr_dist_square = (a["position"][0] - ship_x)**2 + (a["position"][1] - ship_y)**2
                curr_dist = math.sqrt(curr_dist_square)
                if max(curr_dist - a['radius'] - ship_state['radius'], 0) < closest_gap_between_ship_and_closest_asteroid:
                    closest_gap_between_ship_and_closest_asteroid = max(curr_dist - a['radius'] - ship_state['radius'], 0)
                    closest_asteroid = a

            _, shooting_theta, _, _ = calculate_interception(ship_x + ship_vel_x, ship_y + ship_vel_y, closest_asteroid['position'][0] + 2 * delta_time * closest_asteroid["velocity"][0], closest_asteroid["position"][1] + 2 * delta_time * closest_asteroid["velocity"][1], closest_asteroid["velocity"][0], closest_asteroid["velocity"][1], ship_heading)
            
            #thrust = 0
            
            #DEBUG
            #print(thrust, bullet_t, shooting_theta, turn_rate, fire)
            
            if fire == True:
                self.last_time_fired = self.eval_frames

            turn_rate = 0
    '''

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # Method processed each time step by this controller.
        self.current_timestep += 1
        if not self.init_done:
            self.finish_init(game_state)
            self.init_done = True
        
        #print("thrust is " + str(thrust) + "\n" + "turn rate is " + str(turn_rate) + "\n" + "fire is " + str(fire) + "\n")
        #print(ship_state["velocity"])
        #print(ship_state['position'])
        
        thrust_default, turn_rate_default, fire_default, drop_mine_default = 0, 0, False, False

        # Nothing's in the action queue. Evaluate the current situation and figure out the best course of action
        if not self.action_queue:
            print("Nothing's in the queue. Plan more actions.")
            self.plan_actions(ship_state, game_state)
        print(len(self.action_queue))
        # Execute the actions already in the queue for this timestep
        # Initialize defaults. If a component of the action is missing, then the default value will be returned
        thrust_combined, turn_rate_combined, fire_combined, drop_mine_combined = thrust_default, turn_rate_default, fire_default, drop_mine_default

        while self.action_queue and self.action_queue[0][0] == self.current_timestep:
            print("Stuff is in the queue!")
            _, thrust, turn_rate, fire, drop_mine = heapq.heappop(self.action_queue)
            thrust_combined = thrust if thrust is not None else thrust_combined
            turn_rate_combined = turn_rate if turn_rate is not None else turn_rate_combined
            fire_combined = fire if fire is not None else fire_combined
            drop_mine_combined = drop_mine if drop_mine is not None else drop_mine_combined
        # The next action in the queue is for a future timestep. All actions for this timestep are processed.
        print(timesteps_until_terminal_velocity)
        return thrust_combined, turn_rate_combined, fire_combined, drop_mine_combined

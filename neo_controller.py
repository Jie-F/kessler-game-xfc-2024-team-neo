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

delta_time = 1/30 # s/ts
bullet_speed = 800.0 # px/s
ship_max_turn_rate = 180.0 # deg/s
ship_max_thrust = 480.0 # px/s^2
ship_drag = 80.0 # px/s^2
ship_max_speed = 240.0 # px/s
eps = 0.00000001
ship_radius = 20.0 # px
timesteps_until_terminal_velocity = math.ceil(ship_max_speed/(ship_max_thrust - ship_drag)/delta_time) # 18 timesteps
collision_check_pad = 1 # px
asteroid_aim_buffer_pixels = 7 # px

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
    # Note that if an asteroid is inside the ship, the predicted collision time will correspond to the time the asteroid LEAVES the ship, not hits
    #print(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_radius, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius)
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
        if a == 0:
            print("Uhh yeah nuh uh")
            sys.exit()
        t1 = math.nan
        t2 = math.nan
    return t1, t2

def predict_next_imminent_collision_time_with_asteroid(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius):
    next_imminent_collision_time = math.inf
    t1, t2 = collision_prediction(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius)
    #print(f"Quadratic equation gave t1 {t1} t2: {t2} and chatgpt got me {t3} and {t4}")
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
                if (dx, dy) != (0, 0):
                    continue
            # Check to make sure the asteroids are headed in the direction of the main game bounds
            # This optimization multiplies the amount of duplicates asteroids by 0.375 approximately, for the surround/half_surround pattern
            if (dx, dy) != (0, 0):
                if (dx != 0 and np.sign(dx) == np.sign(asteroid['velocity'][0])) or (dy != 0 and np.sign(dy) == np.sign(asteroid['velocity'][1])):
                    # This imaginary asteroid's heading out into space
                    continue
            #print(f"It: col{col} row{row}")
            duplicate = asteroid.copy()
            duplicate['dx'] = dx
            duplicate['dy'] = dy
            duplicate['position'] = (orig_x + dx, orig_y + dy)
            
            duplicates.append(duplicate)
    return duplicates

def check_coordinate_bounds(game_state, x, y):
    if 0 <= x <= game_state['map_size'][0] and 0 <= y <= game_state['map_size'][1]:
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



class NeoShipSim():
    # This was totally just taken from the Kessler game code lul
    def __init__(self, game_state, ship_state, initial_timestep, asteroids=[]):
        self.speed = ship_state['speed']
        self.position = ship_state['position']
        self.velocity = ship_state['velocity']
        self.heading = ship_state['heading']
        self.initial_timestep = initial_timestep
        self.future_timesteps = 0
        self.move_sequence = []
        self.state_sequence = [(self.initial_timestep, self.position, self.velocity, self.speed, self.heading)]
        self.asteroids = asteroids
        self.game_state = game_state

    def get_instantaneous_asteroid_collision(self):
        collision = False
        for a in self.asteroids:
            if check_collision(self.position[0], self.position[1], ship_radius, a['position'][0] + self.future_timesteps*delta_time*a['velocity'][0], a['position'][1] + self.future_timesteps*delta_time*a['velocity'][1], a['radius']):
                collision = True
                break
        return collision

    def get_next_extrapolated_collision_time(self):
        # Assume constant velocity from here
        next_imminent_collision_time = math.inf
        #print('Extrapolating stuff at rest in end')
        for a in self.asteroids:
            next_imminent_collision_time = min(predict_next_imminent_collision_time_with_asteroid(self.position[0], self.position[1], self.velocity[0], self.velocity[1], ship_radius, a['position'][0] + self.future_timesteps*delta_time*a['velocity'][0], a['position'][1] + self.future_timesteps*delta_time*a['velocity'][1], a['velocity'][0], a['velocity'][1], a['radius']), next_imminent_collision_time)
        return next_imminent_collision_time

    def update(self, thrust=0, turn_rate=0):
        self.future_timesteps += 1
        # Apply drag. Fully stop the ship if it would cross zero speed in this time (prevents oscillation)
        drag_amount = ship_drag * delta_time
        if drag_amount > abs(self.speed):
            self.speed = 0
        else:
            self.speed -= drag_amount * np.sign(self.speed)
        # Bounds check the thrust
        thrust = min(max(-ship_max_thrust, thrust), ship_max_thrust)
        # Apply thrust
        self.speed += thrust * delta_time
        # Bounds check the speed
        if self.speed > ship_max_speed:
            self.speed = ship_max_speed
        elif self.speed < -ship_max_speed:
            self.speed = -ship_max_speed
        # Bounds check the turn rate
        turn_rate = min(max(-ship_max_turn_rate, turn_rate), ship_max_turn_rate)
        # Update the angle based on turning rate
        self.heading += turn_rate * delta_time
        # Keep the angle within (-180, 180)
        while self.heading > 360:
            self.heading -= 360.0
        while self.heading < 0:
            self.heading += 360.0
        # Use speed magnitude to get velocity vector
        self.velocity = [math.cos(math.radians(self.heading))*self.speed, math.sin(math.radians(self.heading))*self.speed]
        # Update the position based off the velocities
        self.position = [pos + vel*delta_time for pos, vel in zip(self.position, self.velocity)]
        # Wrap position
        for idx, pos in enumerate(self.position):
            bound = self.game_state['map_size'][idx]
            offset = bound - pos
            if offset < 0 or offset > bound:
                self.position[idx] += bound * np.sign(offset)
        if self.get_instantaneous_asteroid_collision():
            return False
        self.move_sequence.append((self.initial_timestep + self.future_timesteps, thrust, turn_rate))
        self.state_sequence.append((self.initial_timestep + self.future_timesteps - 1, self.position, self.velocity, self.speed, self.heading))
        return True

    def rotate_heading(self, heading_difference_deg):
        target_heading = (self.heading + heading_difference_deg) % 360
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
        return self.move_sequence
    
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
        self.current_timestep = 0
        self.previously_targetted_asteroid = None
        self.fire_on_frames = set()
        self.shot_at_asteroids = {} # Dict of tuples, with the values corresponding to the timesteps we need to wait until they can be shot at again
        self.last_time_fired = -4 # Set it to -4, so that we think the cooldown is gone at the start of the game lul
        self.action_queue = []  # This will become our heap
        self.ship_id = None
        heapq.heapify(self.action_queue) # Transform list into a heap

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

    def plan_maneuver(self, ship_state: Dict, game_state: Dict):
        # Simulate and look for a good move
        print(f"Checking for imminent danger. We're currently at position {ship_state['position'][0]} {ship_state['position'][1]}")
        #print(f"Current ship location: {ship_state['position'][0]}, {ship_state['position'][1]}, ship heading: {ship_state['heading']}")
        # Check for danger
        next_imminent_collision_time_stationary = math.inf
        next_imminent_collision_time = None
        for real_asteroid in game_state['asteroids']:
            duplicated_asteroids = duplicate_asteroids_for_wraparound(real_asteroid, game_state['map_size'][0], game_state['map_size'][1], 'half_surround')
            for a in duplicated_asteroids:
                next_imminent_collision_time_stationary = min(predict_next_imminent_collision_time_with_asteroid(ship_state['position'][0], ship_state['position'][1], ship_state['velocity'][0], ship_state['velocity'][1], ship_state['radius'], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius']), next_imminent_collision_time_stationary)
        
        print(f"Next imminent collision is in {next_imminent_collision_time_stationary}s if we don't move")
        safe_time_threshold = 2
        moving_multiplies_safe_time_by_at_least = 1.5
        if next_imminent_collision_time_stationary > safe_time_threshold:
            # Nothing's gonna hit us for a long time
            print(f"We're safe for the next {safe_time_threshold} seconds")
            #self.enqueue_action(self.current_timestep)
            return False
        else:
            print("Look for a course of action to potentially move")
            # Run a simulation and find a course of action to put me to safety
            # Ghetto genetic algorithm
            safe_maneuver_found = False
            best_imminent_collision_time_found = 0
            best_maneuver_move_sequence = []
            max_search_iterations = 100
            min_search_iterations = 10
            search_iterations_count = 0
            duplicated_asteroids = []
            # TO DO: CULL ASTEROIDS THAT DONT COME ANYWHERE CLOSE TO ALL POSSIBLE PLACES THE SHIP CAN MOVE TO, AND THIS INCLUDES CULLING REAL ASTEROIDS, NOT JUST DUPES FOR WRAPPING
            for real_asteroid in game_state['asteroids']:
                duplicated_asteroids.extend(duplicate_asteroids_for_wraparound(real_asteroid, game_state['map_size'][0], game_state['map_size'][1], 'half_surround'))
            while search_iterations_count < min_search_iterations or (not safe_maneuver_found and search_iterations_count < max_search_iterations):
                search_iterations_count += 1
                if search_iterations_count % 10 == 0:
                    print(f"Search iteration {search_iterations_count}")
                #random_ship_heading_angle = random.uniform(-90.0, 90.0)
                random_ship_accel_turn_rate = random.uniform(-ship_max_turn_rate, ship_max_turn_rate)
                random_ship_cruise_speed = random.uniform(-ship_max_speed, ship_max_speed)
                random_ship_cruise_turn_rate = random.uniform(-ship_max_turn_rate, ship_max_turn_rate)
                max_cruise_seconds = 1
                random_ship_cruise_timesteps = random.randint(1, round(max_cruise_seconds/delta_time))

                sim_ship = NeoShipSim(game_state, ship_state, self.current_timestep, duplicated_asteroids)
                #if not sim_ship.rotate_heading(random_ship_heading_angle):
                #    continue
                if not sim_ship.accelerate(random_ship_cruise_speed, random_ship_accel_turn_rate):
                    continue
                if not sim_ship.cruise(random_ship_cruise_timesteps, random_ship_cruise_turn_rate):
                    continue
                if not sim_ship.accelerate(0):
                    continue
                next_imminent_collision_time = sim_ship.get_next_extrapolated_collision_time()
                if next_imminent_collision_time > best_imminent_collision_time_found:
                    print(f"Alright we found a better one with time {next_imminent_collision_time}")
                    best_imminent_collision_time_found = next_imminent_collision_time
                    best_maneuver_move_sequence = sim_ship.get_move_sequence()
                if next_imminent_collision_time >= safe_time_threshold:
                    safe_maneuver_found = True
                    print(f"Found safe maneuver! Next imminent collision time is {best_imminent_collision_time_found}")
                    print(f"Maneuver takes this many seconds: {len(best_maneuver_move_sequence)*delta_time}")
                    #print(f"Projected location to move will be to: {ship_sim_pos_x}, {ship_sim_pos_y} with heading {current_ship_heading + heading_difference_deg}")
                    if search_iterations_count >= min_search_iterations:
                        break
                    else:
                        continue
                else:
                    # Try the next thrust amount just in case we can dodge it by going more
                    continue
                #print("Went through all possible thrusts for this direction! Trying new random one.")
            if search_iterations_count == max_search_iterations:
                print("Hit the max iteration count")
            else:
                print(f"Did {search_iterations_count} search iterations to find something good")
            #end_state = sim_ship.get_state_sequence()[-1]
            if next_imminent_collision_time is not None and next_imminent_collision_time > next_imminent_collision_time_stationary*moving_multiplies_safe_time_by_at_least:
                # K yeah it's worth moving
                print("K yeah it's worth moving")
                #print(f"We're gonna end up at {end_state[1][0]}, {end_state[1][1]}")
                # Enqueue the safe maneuver
                for move in best_maneuver_move_sequence:
                    self.enqueue_action(move[0], move[1], move[2])
                if not best_maneuver_move_sequence:
                    # Null action, just so I can take damage and the sim doesn't crash
                    #self.enqueue_action(self.current_timestep)
                    return False
                else:
                    return True
            else:
                # Stay put, not worth moving
                print("Stay put, not worth moving")
                return False
                #self.enqueue_action(self.current_timestep)
            #print(self.action_queue)

    def check_convenient_shots(self, ship_state, game_state):
        # Iterate over all asteroids including duplicated ones, and check whether we're perfectly in line to hit any
        feasible_to_hit = []
        for real_asteroid in game_state['asteroids']:
            duplicated_asteroids = duplicate_asteroids_for_wraparound(real_asteroid, game_state['map_size'][0], game_state['map_size'][1], 'surround')
            for a in duplicated_asteroids:
                feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception = calculate_interception(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], a['velocity'][0], a['velocity'][1], a['radius'], ship_state['heading'], game_state)
                if feasible:
                    feasible_to_hit.append((a, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, asteroid_dist_during_interception))
        #print(feasible_to_hit)
        # Iterate through the feasible ones and figure out which one we're actually going to hit if we shoot
        if ship_state['can_fire']:
            most_direct_asteroid_shot_tuple = None
            for feasible in feasible_to_hit:
                a, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, asteroid_dist_during_interception = feasible
                #print(f"Shot tolerance is and the ship's heading is {math.radians(ship_state['heading'])}")
                if abs(shot_heading_error_rad) < shot_heading_tolerance_rad:
                    if most_direct_asteroid_shot_tuple == None or asteroid_dist_during_interception < most_direct_asteroid_shot_tuple[4]:
                        most_direct_asteroid_shot_tuple = feasible
            if most_direct_asteroid_shot_tuple is not None:
                print("Shooting the convenient shot!")
                self.enqueue_action(self.current_timestep, None, None, True)
                self.shot_at_asteroids[(most_direct_asteroid_shot_tuple[0]['velocity'][0], most_direct_asteroid_shot_tuple[0]['velocity'][1], most_direct_asteroid_shot_tuple[0]['radius'], most_direct_asteroid_shot_tuple[0]['dy'], most_direct_asteroid_shot_tuple[0]['dx'])] = math.ceil(most_direct_asteroid_shot_tuple[3] / delta_time)

    def get_feasible_intercept_angle_and_turn_time(self, a, ship_state, game_state, timesteps_until_can_fire=0):
        # a could be a virtual asteroid, and we'll bounds check it for feasibility
        #timesteps_until_can_fire += 2
        # This function will check whether it's feasible to intercept an asteroid, whether real or virtual, within the bounds of the game
        # It will find exactly how long it'll take to turn toward the right heading before shooting, and consider turning both directions and taking the min turn amount/time
        # There are possible cases where you counterintuitively want to turn the opposite way of the asteroid, especially in crazy cases where the asteroid is moving faster than the bullet, so the ship doesn't chase down the asteroid and instead preempts it

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
                shooting_angle_error_rad = 0
            else:
                if shot_heading_error_rad > 0:
                    shooting_angle_error_rad = shot_heading_error_rad - shot_heading_tolerance_rad
                else:
                    shooting_angle_error_rad = shot_heading_error_rad + shot_heading_tolerance_rad
            # shooting_angle_error is the amount we need to move our heading by, in radians
            # If there's some timesteps until we can fire, then we can get a head start on the aiming
            new_aiming_timesteps_required = max(0, math.ceil(math.degrees(abs(shooting_angle_error_rad))/(ship_max_turn_rate*delta_time) - eps) - timesteps_until_can_fire)
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
            print(f'shooting_angle_error_deg: {shooting_angle_error_deg} asdy8f9y7asdf89asdy789fads789y9asdf78y7asdfy79asdfy789asdfy789asdfy789asdfy978asdfy789asdfy78adfsy78asdfy78y78asdfyf87asdyf78dsay9f78dsayf78sy78fsya78fysa78fy78sdayf78ysa78yf78asdy78fas')
        if not feasible:
            return False, None, None, None, None, None, None
        return feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception

    def plan_targetting(self, ship_state, game_state):
        # Iterate over all asteroids including duplicated ones to get the ones that can be feasibly aimed for on the current as well as next timestep, not considering line of sight
        feasible_to_hit_current_timestep = []
        feasible_to_hit_next_timestep = []
        #print(self.last_time_fired, self.current_timestep)
        timesteps_until_can_fire = max(0, 5 - (self.current_timestep - self.last_time_fired))
        #print(f"Timesteps until we can fire: {timesteps_until_can_fire}")
        for real_asteroid in game_state['asteroids']:
            duplicated_asteroids = duplicate_asteroids_for_wraparound(real_asteroid, game_state['map_size'][0], game_state['map_size'][1], 'surround')
            for a in duplicated_asteroids:
                if (a["velocity"][0], a["velocity"][1], a["radius"], a['dx'], a['dy']) not in self.shot_at_asteroids:
                    # Alright so what the heck is this convergence stuff all about?
                    # So let's say at this timestep we calculate that if we want to shoot an asteroid, we have to shoot at X degrees from our heading.
                    # But the thing is, we can only hit the asteroid if we were already looking there and we shoot at this exact instant.
                    # By the time we can turn our ship to the correct heading, the future correct heading will have moved even farther! We have Zeno's paradox.
                    # So what we need to find, is a future timestep at which between now and then, we have enough time to turn our ship, to achieve the future correct heading to shoot at the bullet
                    # There may be a way to solve this algebraically, but since we're working with discrete numbers, we can hone in on this value within a few iterations.
                    feasible_current_timestep, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = self.get_feasible_intercept_angle_and_turn_time(a, ship_state, game_state, timesteps_until_can_fire)
                    #print(f"Converged. Need extra timesteps: {aiming_timesteps_required}")
                    if feasible_current_timestep:
                        feasible_to_hit_current_timestep.append((a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y))
                    if timesteps_until_can_fire == 0:
                        # Also calculate for the next timestep in case we can fire on this timestep and begin turning toward our next target for the future
                        feasible_next_timestep, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = self.get_feasible_intercept_angle_and_turn_time(a, ship_state, game_state, timesteps_until_can_fire + 1)
                        if feasible_next_timestep:
                            feasible_to_hit_next_timestep.append((a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, aiming_timesteps_required, intercept_x, intercept_y))
        #print("Feasible to hit list:")
        #print(feasible_to_hit_current_timestep)
        #print(feasible_to_hit_next_timestep)

        # Iterate through the feasible asteroids and figure out which one we're actually probably going to hit if we shoot (since some asteroids can cover up other asteroids and you may not hit your intended target)
        min_abs_shooting_angle_error_deg = math.inf
        min_turn_waiting_timesteps = math.inf
        most_direct_asteroid_shot_tuple = None
        # TODO: min_abs_shooting_angle_error is kind of a useless temp variable, but it might be good to keep anyway
        for feasible in feasible_to_hit_current_timestep:
            #print(shot_heading_error_rad)
            # Shooting_angle_deg already considers that we don't have to aim at the center of the asteroid. It shoots the side of the asteroid that requires us to aim less
            a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, _, _, _ = feasible
            # TODO: Clean up the if statements below and perhaps combine redundant cases
            if most_direct_asteroid_shot_tuple is None:
                most_direct_asteroid_shot_tuple = feasible
                min_abs_shooting_angle_error_deg = shooting_angle_error_deg
                min_turn_waiting_timesteps = feasible[4]
            elif feasible[4] < min_turn_waiting_timesteps or (feasible[4] == min_turn_waiting_timesteps and abs(shooting_angle_error_deg) < abs(min_abs_shooting_angle_error_deg)):
                #print(f"{shooting_angle_error_deg} is better than {min_abs_shooting_angle_error_deg}")
                # New best target that requires the least movement to hit
                most_direct_asteroid_shot_tuple = feasible
                min_abs_shooting_angle_error_deg = shooting_angle_error_deg
            elif abs(shooting_angle_error_deg) == abs(min_abs_shooting_angle_error_deg):
                # These are both likely direct perfect shots. As a tiebreaker, use whichever asteroid is closer when considering its radius, since it should mean it's more in front
                if most_direct_asteroid_shot_tuple == None or (asteroid_dist_during_interception - a['radius'] < most_direct_asteroid_shot_tuple[3] - most_direct_asteroid_shot_tuple[0]['radius']):
                    most_direct_asteroid_shot_tuple = feasible
                    min_abs_shooting_angle_error_deg = shooting_angle_error_deg
        #print(timesteps_until_can_fire)
        assert((timesteps_until_can_fire == 0) == ship_state['can_fire'])

        if most_direct_asteroid_shot_tuple is not None:
            min_abs_shooting_angle_error_deg = math.inf
            second_most_direct_asteroid_shot_tuple = None
            for feasible in feasible_to_hit_next_timestep:
                #print(shot_heading_error_rad)
                # Shooting_angle_deg already considers that we don't have to aim at the center of the asteroid. It shoots the side of the asteroid that requires us to aim less
                a, shooting_angle_error_deg, interception_time_s, asteroid_dist_during_interception, _, _, _ = feasible
                if most_direct_asteroid_shot_tuple[0]['velocity'][0] != feasible[0]['velocity'][0] and most_direct_asteroid_shot_tuple[0]['velocity'][1] != feasible[0]['velocity'][1] and most_direct_asteroid_shot_tuple[0]['dx'] != feasible[0]['dx'] and most_direct_asteroid_shot_tuple[0]['dy'] != feasible[0]['dy']:
                    # As long as the second one isn't the same as the first asteroid lmao
                    # TODO: Clean up the if statements below and perhaps combine redundant cases
                    if second_most_direct_asteroid_shot_tuple is None:
                        second_most_direct_asteroid_shot_tuple = feasible
                        min_abs_shooting_angle_error_deg = shooting_angle_error_deg
                        min_turn_waiting_timesteps = feasible[4]
                    elif feasible[4] < min_turn_waiting_timesteps or (feasible[4] == min_turn_waiting_timesteps and abs(shooting_angle_error_deg) < abs(min_abs_shooting_angle_error_deg)):
                        #print(f"{shooting_angle_error_deg} is better than {min_abs_shooting_angle_error_deg}")
                        # New best target that requires the least movement to hit
                        second_most_direct_asteroid_shot_tuple = feasible
                        min_abs_shooting_angle_error_deg = shooting_angle_error_deg
                    elif abs(shooting_angle_error_deg) == abs(min_abs_shooting_angle_error_deg):
                        # These are both likely direct perfect shots. As a tiebreaker, use whichever asteroid is closer when considering its radius, since it should mean it's more in front
                        if second_most_direct_asteroid_shot_tuple == None or (asteroid_dist_during_interception - a['radius'] < second_most_direct_asteroid_shot_tuple[3] - second_most_direct_asteroid_shot_tuple[0]['radius']):
                            second_most_direct_asteroid_shot_tuple = feasible
                            min_abs_shooting_angle_error_deg = shooting_angle_error_deg

        if most_direct_asteroid_shot_tuple is not None:
            # We have a #1 target
            min_abs_shooting_angle_error_deg = most_direct_asteroid_shot_tuple[1]
            #print(f"Our #1 target is angularlly away in degrees: {min_abs_shooting_angle_error_deg} and in num timesteps required for aiming: {most_direct_asteroid_shot_tuple[4]}")
            fired_on_this_timestep = False
            if abs(min_abs_shooting_angle_error_deg) < eps and ship_state['can_fire']:
                print("Locked and loaded. We're firing on this timestep, and probably gonna move too")
                fired_on_this_timestep = True
                self.enqueue_action(self.current_timestep, None, None, True)
                #print(f"Time it takes to intercept asteroid with bullet: {most_direct_asteroid_shot_tuple[2]}")
                self.shot_at_asteroids[(most_direct_asteroid_shot_tuple[0]['velocity'][0], most_direct_asteroid_shot_tuple[0]['velocity'][1], most_direct_asteroid_shot_tuple[0]['radius'], most_direct_asteroid_shot_tuple[0]['dy'], most_direct_asteroid_shot_tuple[0]['dx'])] = math.ceil(most_direct_asteroid_shot_tuple[2]/delta_time)
                # The way the game works is at this timestep, I can fire a bullet and begin turning toward my second target
                # Aim at the second best target right now, since we're already shooting our best target and want something to do
                if second_most_direct_asteroid_shot_tuple is not None:
                    #print("Aiming at second best target on the same timestep, so we might be able to shoot it at the next timestep")
                    min_abs_shooting_angle_error_deg = second_most_direct_asteroid_shot_tuple[1]
            elif abs(min_abs_shooting_angle_error_deg) < eps:
                print("DANG IT we're locked and loaded, but we can't fire right now!")
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
        #self.shot_at_asteroids[(best_asteroid["velocity"][0], best_asteroid["velocity"][1], best_asteroid["radius"], dy, dx)] = math.ceil(bullet_t / delta_time)

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # Method processed each time step by this controller.
        self.current_timestep += 1
        print(f"Timestep {self.current_timestep}")
        if not self.init_done:
            self.finish_init(game_state, ship_state)
            self.init_done = True
        
        #print("thrust is " + str(thrust) + "\n" + "turn rate is " + str(turn_rate) + "\n" + "fire is " + str(fire) + "\n")
        #print(ship_state["velocity"])
        #print(ship_state['position'])
        if ship_state['is_respawning']:
            print("OUCH WE LOST A LIFE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        thrust_default, turn_rate_default, fire_default, drop_mine_default = 0, 0, False, False

        # Nothing's in the action queue. Evaluate the current situation and figure out the best course of action
        if not self.action_queue:
            #print("Nothing's in the queue. Plan more actions.")
            planned_maneuvers = self.plan_maneuver(ship_state, game_state)
        else:
            planned_maneuvers = True
        #print(len(self.action_queue))
        if planned_maneuvers:
            print("We've gotta dodge asteroids. We can still shoot them while we're dodging, if a shot happens to line up.")
            self.check_convenient_shots(ship_state, game_state)
        else:
            # We didn't plan any maneuvering. We can use this time to do targetting
            print("We can chill at this spot for a bit. Entering targetting mode.")
            self.plan_targetting(ship_state, game_state)
        
        # Maintain the list of already shot at asteroids
        keys_to_remove = []
        # Iterate through the dictionary items
        for key, value in self.shot_at_asteroids.items():
            # Decrease each value by 1 (the outstanding number of frames the bullet has until it would hit or miss its target decreases by 1 each timestep)
            self.shot_at_asteroids[key] = value - 1
            # If the value hits 0, add the key to the list of keys to be removed
            if self.shot_at_asteroids[key] <= 0:
                keys_to_remove.append(key)
        # Remove the keys from the dictionary
        for key in keys_to_remove:
            del self.shot_at_asteroids[key]

        # Execute the actions already in the queue for this timestep

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
            self.last_time_fired = self.current_timestep
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
        if drop_mine_combined and not ship_state['can_deploy_mine']:
            print("You can't deploy mines dude!")
        print(f"Inputs on timestep {self.current_timestep} - thrust: {thrust_combined}, turn_rate: {turn_rate_combined}, fire: {fire_combined}, drop_mine: {drop_mine_combined}")
        #time.sleep(0.2)
        #print(game_state, ship_state)
        return thrust_combined, turn_rate_combined, fire_combined, drop_mine_combined

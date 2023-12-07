import random
from src.kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
from collections import deque
import heapq
import sys

delta_time = 1/30 # s/ts
ship_max_turn_rate = 180.0 # deg/s
ship_max_thrust = 480.0 # px/s^2
ship_drag = 80.0 # px/s^2
ship_max_speed = 240.0 # px/s
eps = 0.00000001
ship_radius = 20.0 # px
timesteps_until_terminal_velocity = math.ceil(ship_max_speed/(ship_max_thrust - ship_drag)/delta_time)
collision_check_pad = 1

class NeoShipSim():
    # This was totally just taken from the Kessler game code lul
    def __init__(self, position, velocity, angle, initial_timestep, asteroids=[]):
        self.speed = 0
        self.position = position
        self.velocity = velocity
        self.heading = angle
        self.initial_timestep = initial_timestep
        self.future_timesteps = 0
        self.move_sequence = []
        self.state_sequence = [(self.initial_timestep, self.position, self.velocity, self.speed, self.heading)]
        self.asteroids = asteroids

    def get_instantaneous_asteroid_collision(self):
        return check_collision()

    def update(self, thrust=0, turn_rate=0):
        self.future_timesteps += 1
        # Apply drag. Fully stop the ship if it would cross zero speed in this time (prevents oscillation)
        drag_amount = ship_drag * delta_time
        if drag_amount > abs(self.speed):
            self.speed = 0
        else:
            self.speed -= drag_amount * np.sign(self.speed)
        # Apply thrust
        self.speed += thrust * delta_time
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
        self.move_sequence.append((self.initial_timestep + self.future_timesteps, thrust, turn_rate))
        self.state_sequence.append((self.initial_timestep + self.future_timesteps, self.position, self.velocity, self.speed, self.heading))

    def rotate_heading(self, heading_difference_deg):
        target_heading = (self.heading + heading_difference_deg) % 360
        still_need_to_turn = heading_difference_deg
        while abs(still_need_to_turn) > ship_max_turn_rate*delta_time:
            self.update(0, ship_max_turn_rate*np.sign(heading_difference_deg))
            still_need_to_turn -= ship_max_turn_rate*np.sign(heading_difference_deg)*delta_time
        self.update(0, still_need_to_turn/delta_time)
        assert(abs(target_heading - self.heading) < eps)

    def accelerate(self, target_speed):
        # Keep in mind speed can be negative
        # Drag will always slow down the ship
        while abs(self.speed - target_speed) > eps:
            drag = -ship_drag*np.sign(self.speed)
            delta_speed_to_target = target_speed - self.speed
            thrust_amount = delta_speed_to_target/delta_time - drag
            thrust_amount = min(max(-ship_max_thrust, thrust_amount), ship_max_thrust)
            self.update(thrust_amount, 0)

    def cruise(self, cruise_time, cruise_turn_rate=0):
        # Maintain current speed
        for _ in range(cruise_time):
            self.update(np.sign(self.speed)*ship_drag, cruise_turn_rate)

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

s = NeoShipSim((0, 0), (0, 0), 0, 0)
s.rotate_heading(90)
s.accelerate(4)
s.cruise(10)
s.accelerate(-9)
s.accelerate(0)
print(s.get_move_sequence())
print(s.get_state_sequence())

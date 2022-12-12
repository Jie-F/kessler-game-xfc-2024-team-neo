# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import math
import numpy as np
from typing import Dict, Any, List

from .bullet import Bullet


class Ship:
    def __init__(self, ship_id,
                 position: List[float],
                 angle: float = 90,
                 lives: int = 3,
                 team: int = 1,
                 bullets_remaining: int = -1):
        """
        Instantiate a ship with default parameters and infinite bullets if not specified
        """

        # Control information
        self.thrust = 0.0     # speed defaults to minimum
        self.turn_rate = 0.0

        # State info
        self.id = ship_id
        self.speed = 0
        self.position = position
        self.velocity = [0, 0]
        self.heading = angle
        self.lives = lives
        self.deaths = 0
        self.team = team

        # Controller inputs
        self.fire = False
        self.thrust = 0
        self.turn_rate = 0

        # Physical model constants/params
        self.thrust_range = (-480.0, 480.0)  # m/s^2
        self.turn_rate_range = (-180.0, 180.0)  # Degrees per second
        self.max_speed = 240  # Meters per second
        self.drag = 80.0  # m/s^2
        self.radius = 20  # meters TODO verify radius size
        self.mass = 300  # kg - reasonable? max asteroid mass currently is ~490 kg

        # Manage respawns/firing via timers
        self._respawning = 0
        self._respawn_time = 3  # seconds
        self._fire_limiter = 0
        self._fire_time = 1 / 10  # seconds

        # Track bullet statistics
        self.bullets_remaining = bullets_remaining
        self.bullets_shot = 0
        self.bullets_hit = 0    # Number of bullets that hit an asteroid
        self.asteroids_hit = 0  # Number of asteroids hit (including ship collision)

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "is_respawning": True if self.is_respawning else False,
            "position": tuple(self.position),
            "velocity": tuple([float(v) for v in self.velocity]),
            "speed": float(self.speed),
            "heading": float(self.heading),
            "mass": float(self.mass),
            "radius": float(self.radius),
            "id": int(self.id),
            "team": int(self.team),
            "lives_remaining": int(self.lives),
        }

    @property
    def ownstate(self) -> Dict[str, Any]:
        return {**self.state,
                "bullets_remaining": self.bullets_remaining,
                "can_fire": True if self.can_fire else False,
                "fire_rate": self.fire_rate,
                "thrust_range": self.thrust_range,
                "turn_rate_range": self.turn_rate_range,
                "max_speed": self.max_speed,
                "drag": self.drag,
        }

    @property
    def alive(self):
        return True if self.lives > 0 else False

    @property
    def is_respawning(self) -> bool:
        return True if self._respawning else False

    @property
    def respawn_time_left(self) -> float:
        return self._respawning

    @property
    def respawn_time(self) -> float:
        return self._respawn_time

    @property
    def can_fire(self) -> bool:
        return (not self._fire_limiter) and self.bullets_remaining != 0

    @property
    def fire_rate(self) -> float:
        return 1 / self._fire_time

    @property
    def fire_wait_time(self) -> float:
        return self._fire_limiter

    def shoot(self):
        self.fire = True

    def update(self, delta_time: float = 1 / 30) -> Bullet:
        """
        Update our position and other particulars.
        """

        # Fire a bullet if instructed to
        if self.fire:
            new_bullet = self.fire_bullet()
        else:
            new_bullet = None

        # Decrement respawn timer (if necessary)
        if self._respawning <= 0:
            self._respawning = 0
        else:
            self._respawning -= delta_time

        # Decrement fire limit timer (if necessary)
        if self._fire_limiter <= 0.0:
            self._fire_limiter = 0.0
        else:
            self._fire_limiter -= delta_time

        # Apply drag. Fully stop the ship if it would cross zero speed in this time (prevents oscillation)
        drag_amount = self.drag * delta_time
        if drag_amount > abs(self.speed):
            self.speed = 0
        else:
            self.speed -= drag_amount * np.sign(self.speed)

        # Apply thrust
        self.speed += self.thrust * delta_time

        # Bounds check the speed
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        elif self.speed < -self.max_speed:
            self.speed = -self.max_speed

        # Update the angle based on turning rate
        self.heading += self.turn_rate * delta_time

        # Keep the angle within (-180, 180)
        while self.heading > 360:
            self.heading -= 360.0
        while self.heading < 0:
            self.heading += 360.0

        # Use speed magnitude to get velocity vector
        self.velocity = [math.cos(math.radians(self.heading)) * self.speed,
                         math.sin(math.radians(self.heading)) * self.speed]

        # Update the position based off the velocities
        self.position = [pos + v * delta_time for pos, v in zip(self.position, self.velocity)]

        return new_bullet

    def destruct(self, map_size):
        """
        Called by the game when a ship collides with something and dies. Handles life decrementing and triggers respawn
        """
        self.lives -= 1
        self.deaths +=1
        if self.lives > 0:
            # spawn_position = [map_size[0] / 2,
            #                   map_size[1] / 2]
            spawn_position = self.position
            spawn_heading = self.heading
            self.respawn(spawn_position, spawn_heading)

    def respawn(self, position: List[float], heading: float = 90.0) -> None:
        """
        Called when we die and need to make a new ship.
        'respawning' is an invulnerability timer.
        """
        # If we are in the middle of respawning, this is non-zero.
        self._respawning = self._respawn_time

        # Set location and physical parameters
        self.position = position
        self.speed = 0
        self.heading = heading

    def fire_bullet(self):
        if self.bullets_remaining != 0 and not self._fire_limiter:

            # Remove respawn invincibility. Trigger fire limiter
            self._respawning = 0
            self._fire_limiter = self._fire_time

            # Bullet counters
            if self.bullets_remaining > 0:
                self.bullets_remaining -= 1
            self.bullets_shot += 1

            # Return the bullet object that was fired
            bullet_x = self.position[0] + self.radius * np.cos(np.radians(self.heading))
            bullet_y = self.position[1] + self.radius * np.sin(np.radians(self.heading))
            return Bullet([bullet_x, bullet_y], self.heading, owner=self)

        # Return nothing if we can't fire a bullet right now
        return None
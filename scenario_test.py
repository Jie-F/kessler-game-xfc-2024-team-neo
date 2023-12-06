# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time
import random
from neo_controller import NeoController

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

random.seed(8)

from src.kesslergame import Scenario, KesslerGame, GraphicsType
from src.kesslergame.controller_gamepad import GamepadController
from examples.test_controller import TestController
#from examples.graphics_both import GraphicsBoth

def generate_asteroids(num_asteroids, position_range_x, position_range_y, speed_range, angle_range, size_range):
    asteroids = []
    for _ in range(num_asteroids):
        position = (random.randint(*position_range_x), random.randint(*position_range_y))
        speed = random.randint(*speed_range)
        angle = random.randint(*angle_range)
        size = random.randint(*size_range)
        asteroids.append({'position': position, 'speed': speed, 'angle': angle, 'size': size})
    return asteroids

width, height = (2560, 1440)

asteroids_random = generate_asteroids(
                                num_asteroids=1,
                                position_range_x=(0, width),
                                position_range_y=(0, height),
                                speed_range=(1, 150),
                                angle_range=(-180, 180),
                                size_range=(1, 4)
                            )

# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=5,
                            #asteroid_states=asteroids_random,
                            #asteroid_states=[{'position': (500, 920), 'speed': 60, 'angle': -15, 'size': 4}],
                            ship_states=[
                                {'position': (width//2, height//2), 'angle': 90, 'lives': 30, 'team': 1, "mines_remaining": 30},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(width, height),
                            seed=14,
                            time_limit=120,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,#UnrealEngine,Tkinter
                 'realtime_multiplier': 1,
                 'graphics_obj': None,
                 'frequency': 30}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# Evaluate the game
pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[NeoController()])#, TestController()])GamepadController NeoController

# Print out some general info about the result
print('Scenario eval time: '+str(time.perf_counter()-pre))
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: ' + str([team.deaths for team in score.teams]))
print('Accuracy: ' + str([team.accuracy for team in score.teams]))
print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))

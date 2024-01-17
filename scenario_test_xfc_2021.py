# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time
import random
from neo_controller import Neo
from baby_neo_controller import NeoController

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1) # Fixes blurriness when a scale factor is used in Windows

from scenarios import *

portfolio = [
    threat_test_1,
    threat_test_2,
    threat_test_3,
    threat_test_4,
    accuracy_test_1,
    accuracy_test_2,
    accuracy_test_3,
    accuracy_test_4,
    accuracy_test_5,
    accuracy_test_6,
    accuracy_test_7,
    accuracy_test_8,
    accuracy_test_9,
    accuracy_test_10,
    wall_left_easy,
    wall_right_easy,
    wall_top_easy,
    wall_bottom_easy,
    ring_closing,
    ring_static_left,
    ring_static_right,
    ring_static_top,
    ring_static_bottom,

    wall_right_wrap_1,
    wall_right_wrap_2,
    wall_right_wrap_3,
    wall_right_wrap_4,
    wall_left_wrap_1,
    wall_left_wrap_2,
    wall_left_wrap_3,
    wall_left_wrap_4,
    wall_top_wrap_1,
    wall_top_wrap_2,
    wall_top_wrap_3,
    wall_top_wrap_4,
    wall_bottom_wrap_1,
    wall_bottom_wrap_2,
    wall_bottom_wrap_3,
    wall_bottom_wrap_4,
]

show_portfolio = [
    threat_test_1,
    threat_test_2,
    threat_test_3,
    threat_test_4,
    accuracy_test_5,
    accuracy_test_6,
    accuracy_test_7,
    accuracy_test_8,
    accuracy_test_9,
    accuracy_test_10,
    wall_left_easy,
    wall_right_easy,
    wall_top_easy,
    wall_bottom_easy,
    ring_closing,
    ring_static_left,
    ring_static_right,
    ring_static_top,
    ring_static_bottom,
    wall_right_wrap_3,
    wall_right_wrap_4,
    wall_left_wrap_3,
    wall_left_wrap_4,
    wall_top_wrap_3,
    wall_top_wrap_4,
    wall_bottom_wrap_3,
    wall_bottom_wrap_4,
]

alternate_scenarios = [
    #corridor_left,
    #corridor_right,
    #corridor_top,
    #corridor_bottom,

    # May have to cut these
    #moving_corridor_1,
    #moving_corridor_2,
    #moving_corridor_3,
    #moving_corridor_4,
    #moving_corridor_angled_1,
    #moving_corridor_angled_2,
    #moving_corridor_curve_1,
    #moving_corridor_curve_2,

    #scenario_small_box,
    #scenario_big_box,
    scenario_2_still_corridors,
]

portfolio_dict = {scenario.name: scenario for scenario in portfolio}
show_portfolio_dict = {scenario.name: scenario for scenario in show_portfolio}


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

width, height = (1000, 800)
#random.seed(22)
asteroids_random = generate_asteroids(
                                num_asteroids=100,
                                position_range_x=(0, width),
                                position_range_y=(0, height),
                                speed_range=(1, 300),
                                angle_range=(-180, 180),
                                size_range=(1, 3)
                            )

# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                            #num_asteroids=200,
                            asteroid_states=asteroids_random,
                            #asteroid_states=[{'position': (width*54//100, height*54//100), 'speed': 1000, 'angle': -180, 'size': 2}],
                            ship_states=[
                                {'position': (width//2, height//2), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                {'position': (width*2//3, height//2), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(width, height),
                            #seed=2,
                            time_limit=300,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,#UnrealEngine,Tkinter
                 'realtime_multiplier': 5,
                 'graphics_obj': None,
                 'frequency': 30}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

for scene in portfolio:
    # Evaluate the game
    pre = time.perf_counter()
    print(f"Running scenario: {scene.name}")
    score, perf_data = game.run(scenario=scene, controllers=[Neo(), NeoController()])#, TestController()])GamepadController NeoController

    # Print out some general info about the result
    print('Scenario eval time: '+str(time.perf_counter()-pre))
    print(score.stop_reason)
    print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    print('Deaths: ' + str([team.deaths for team in score.teams]))
    print('Accuracy: ' + str([team.accuracy for team in score.teams]))
    print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))

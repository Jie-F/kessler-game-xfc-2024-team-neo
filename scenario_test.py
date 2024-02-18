# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time
import random
from neo_controller import Neo
from smith_controller import Smith
from baby_neo_controller import NeoController
import numpy as np
import cProfile
import sys
from r_controller import RController
from null_controller import NullController
from scenarios import *
#from controller_0 import ReplayController0
#from controller_1 import ReplayController1

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1) # Fixes blurriness when a scale factor is used in Windows

from xfc_2023_replica_scenarios import *
from custom_scenarios import *

from src.kesslergame import Scenario, KesslerGame, GraphicsType
from src.kesslergame.controller_gamepad import GamepadController
from examples.test_controller import TestController
#from examples.graphics_both import GraphicsBoth

global color_text
color_text = True

def color_print(text='', color='white', style='normal', same=False, previous=False):
    global color_text
    global colors
    global styles

    if color_text and 'colorama' not in sys.modules:
        import colorama
        colorama.init()

        colors = {
            'black': colorama.Fore.BLACK,
            'red': colorama.Fore.RED,
            'green': colorama.Fore.GREEN,
            'yellow': colorama.Fore.YELLOW,
            'blue': colorama.Fore.BLUE,
            'magenta': colorama.Fore.MAGENTA,
            'cyan': colorama.Fore.CYAN,
            'white': colorama.Fore.WHITE,
        }

        styles = {
            'dim': colorama.Style.DIM,
            'normal': colorama.Style.NORMAL,
            'bright': colorama.Style.BRIGHT,
        }
    elif color_text:
        import colorama

    if same:
        end = ''
    else:
        end = '\n'
    if color_text:
        print(previous*'\033[A' + colors[color] + styles[style] + str(text) + colorama.Style.RESET_ALL, end=end)
    else:
        print(str(text), end=end)

def generate_asteroids(num_asteroids, position_range_x, position_range_y, speed_range, angle_range, size_range):
    asteroids = []
    for _ in range(num_asteroids):
        position = (random.uniform(*position_range_x), random.uniform(*position_range_y))
        speed = random.triangular(*speed_range) #random.randint(*speed_range) 
        angle = random.uniform(*angle_range)
        size = random.randint(*size_range)
        asteroids.append({'position': position, 'speed': speed, 'angle': angle, 'size': size})
    return asteroids

width, height = (1000, 800)



# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,#UnrealEngine,Tkinter,NoGraphics
                 'realtime_multiplier': 2,
                 'graphics_obj': None,
                 'frequency': 30.0,
                 'UI_settings': 'all'}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# Evaluate the game
missed = False
#for _ in range(1):
iterations = 0

xfc_2021_portfolio = [
    #threat_test_1,
    #threat_test_2,
    #threat_test_3,
    #threat_test_4,
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

    scenario_small_box,
    scenario_big_box,
    scenario_2_still_corridors,
]

xfc2023 = [
    ex_adv_four_corners_pt1,
    ex_adv_asteroids_down_up_pt1,
    ex_adv_asteroids_down_up_pt2,
    ex_adv_direct_facing,
    ex_adv_two_asteroids_pt1,
    ex_adv_two_asteroids_pt2,
    ex_adv_ring_pt1,
    adv_random_big_1,
    adv_random_big_3,
    adv_multi_wall_bottom_hard_1,
    adv_multi_wall_right_hard_1,
    adv_multi_ring_closing_left,
    adv_multi_ring_closing_right,
    adv_multi_two_rings_closing,
    avg_multi_ring_closing_both2,
    adv_multi_ring_closing_both_inside,
    adv_multi_ring_closing_both_inside_fast
]

custom = [
    #target_priority_optimization1,
    #closing_ring_scenario,
    #easy_closing_ring_scenario,
    #more_intense_closing_ring_scenario,
    #rotating_square_scenario,
    #rotating_square_2_overlap,
    #falling_leaves_scenario,
    #zigzag_motion_scenario,
    #shearing_pattern_scenario,
    #super_hard_wrap,
    #wonky_ring,
    #moving_ring_scenario,
    #shifting_square_scenario,
    #delayed_closing_ring_scenario,
    #spiral_assault_scenario,
    #dancing_ring,
    #dancing_ring_2,
    #intersecting_lines_scenario,
    #exploding_grid_scenario,
    #grid_formation_explosion_scenario,
    #aspect_ratio_grid_formation_scenario
]

#for i in range(0, len(seeds)):
#while True:
score = None
died = False
#for sc in xfc2023:
#while died or not missed:
for i in range(1):
#while True:
    iterations += 1
    randseed = random.randint(1, 1000000000) # 187709936 # Try XFC 2023 adv_multi_ring_closing_both_inside with seed 989425266, with [SomeController(), Neo()] and Neo will die because it doesn't use respawn cooldown properly!
    color_print(f'\nUsing seed {randseed}, running test iteration {iterations}', 'green')
    random.seed(randseed)
    asteroids_random = generate_asteroids(
                                    num_asteroids=10,
                                    position_range_x=(0, width),
                                    position_range_y=(0, height),
                                    speed_range=(-300, 600, 0),
                                    angle_range=(-1, 361),
                                    size_range=(1, 4)
                                )*random.choice([1])

    # Define game scenario
    rand_scenario = Scenario(name='Random Scenario',
                                #num_asteroids=200,
                                asteroid_states=asteroids_random,
                                #asteroid_states=[{'position': (width//2+10000, height*40//100), 'speed': 100, 'angle': -89, 'size': 4}],
                                #                {'position': (width*2//3, height*40//100), 'speed': 100, 'angle': -91, 'size': 4},
                                #                 {'position': (width*1//3, height*40//100), 'speed': 100, 'angle': -91, 'size': 4}],
                                ship_states=[
                                    {'position': (width//3, height//2), 'angle': 0, 'lives': 2, 'team': 1, "mines_remaining": 0},
                                    #{'position': (width*2//3, height//2), 'angle': 90, 'lives': 2, 'team': 2, "mines_remaining": 0},
                                ],
                                map_size=(width, height),
                                #seed=2,
                                time_limit=600,
                                ammo_limit_multiplier=random.choice([0]),
                                stop_if_no_ammo=False)

    pre = time.perf_counter()
    try:
        print(f"Evaluating scenario {sc.name}")
    except:
        pass
    #cProfile.run('game.run(scenario=zigzag_motion_scenario, controllers=[Neo(), Neo()])')
    # my_test_scenario
    # ex_adv_four_corners_pt1 ex_adv_asteroids_down_up_pt1 ex_adv_asteroids_down_up_pt2 adv_multi_wall_bottom_hard_1 
    # closing_ring_scenario more_intense_closing_ring_scenario rotating_square_scenario falling_leaves_scenario shearing_pattern_scenario zigzag_motion_scenario
    controllers_used = [Neo(), NullController()] # [ReplayController0(), ReplayController1()] GamepadController()])#, NeoController()])#, TestController()])GamepadController NeoController Neo
    try:
        score, perf_data = game.run(scenario=sc, controllers=controllers_used)
    except:
        score, perf_data = game.run(scenario=rand_scenario, controllers=controllers_used)
    # Print out some general info about the result
    if score:
        color_print('Scenario eval time: '+str(time.perf_counter()-pre), 'green')
        color_print(score.stop_reason, 'green')
        color_print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]), 'green')
        color_print('Deaths: ' + str([team.deaths for team in score.teams]), 'green')
        if [team.deaths for team in score.teams][0] >= 1:
            died = True
        else:
            died = False
        color_print('Accuracy: ' + str([team.accuracy for team in score.teams]), 'green')
        color_print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]), 'green')
        if score.teams[0].accuracy < 1:
            color_print('NEO MISSED SDIOFJSDI(FJSDIOJFIOSDJFIODSJFIOJSDIOFJSDIOFJOSDIJFISJFOSDJFOJSDIOFJOSDIJFDSJFI)SDFJHSUJFIOSJFIOSJIOFJSDIOFJIOSDFOSDF\n\n', 'red')
            missed = True
        else:
            missed = False
if missed:
    color_print(f"Ran {iterations} simulations to get one where Neo missed!", 'green')

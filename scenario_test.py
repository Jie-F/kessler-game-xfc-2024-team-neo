import time
import random
from hardcoded_test_controller import Smith
from src.baby_neo_controller import BabyNeoController
from benchmark_controller import BenchmarkController
import numpy as np
import cProfile
import sys
from r_controller import RController
from null_controller import NullController
from test_controller import TestController
from scenarios import *
from adversarial_scenarios_for_jie import *
import argparse
#from controller_0 import ReplayController0
#from controller_1 import ReplayController1
from src.xfc2024_neo_controller import NeoController as XFC2024NeoController

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1) # Fixes blurriness when a scale factor is used in Windows

ASTEROID_COUNT_LOOKUP = (0, 1, 4, 13, 40)

#from xfc_2023_replica_scenarios import *
from custom_scenarios import *

from src.kesslergame import Scenario, KesslerGame, GraphicsType
from src.kesslergame.controller_gamepad import GamepadController
#from examples.test_controller import TestController

parser = argparse.ArgumentParser(description='Run Kessler Game with optional CLI flags.')
parser.add_argument('-invisible', action='store_true', help='Use NoGraphics for the game visualization.')
parser.add_argument('-unreal', action='store_true', help='Use UnrealEngine for the game visualization.')
parser.add_argument('-profile', action='store_true', help='Enable profiling of the game run.')
parser.add_argument('-seed', type=int, help='Set the seed for random number generation.')
parser.add_argument('-interp', action='store_true', help='Import the interpreted Neo Controller.')
parser.add_argument('-scenario', type=str, help='Name of scenario to evaluate. Must be available in environment.')
parser.add_argument('-portfolio', type=str, help='Name of portfolio to run')
parser.add_argument('-index', type=int, help='Pick the starting index of the portfolio. Count from zero.')
parser.add_argument('-once', action='store_true', help='Only do one iteration.')

args = parser.parse_args()

if args.interp:
    from neo_controller import NeoController
else:
    from src.neo_controller import NeoController
    #from src.neo_controller_explanations import NeoController
    #from src.neo_controller_training import NeoController


global color_text
color_text = True

def color_print(text='', color='white', style='normal', same=False, previous=False) -> None:
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
                 'graphics_type': GraphicsType.NoGraphics if args.invisible else (GraphicsType.UnrealEngine if args.unreal else GraphicsType.Tkinter),#UnrealEngine,Tkinter,NoGraphics
                 'realtime_multiplier': 0,
                 'graphics_obj': None,
                 'frequency': 30.0,
                 'UI_settings': 'all'}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# Evaluate the game
missed = False
iterations = 0

xfc_2021_portfolio = [
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
    corridor_left,
    corridor_right,
    corridor_top,
    corridor_bottom,

    # May have to cut these
    moving_corridor_1,
    moving_corridor_2,
    moving_corridor_3,
    moving_corridor_4,
    moving_corridor_angled_1,
    moving_corridor_angled_2,
    moving_corridor_curve_1,
    moving_corridor_curve_2,

    scenario_small_box,
    scenario_big_box,
    scenario_2_still_corridors,
]

# xfc2023 = [
#     ex_adv_four_corners_pt1,
#     ex_adv_four_corners_pt2,
#     ex_adv_asteroids_down_up_pt1,
#     ex_adv_asteroids_down_up_pt2,
#     ex_adv_direct_facing,
#     ex_adv_two_asteroids_pt1,
#     ex_adv_two_asteroids_pt2,
#     ex_adv_ring_pt1,
#     adv_random_big_1,
#     adv_random_big_2,
#     adv_random_big_3,
#     adv_random_big_4,
#     adv_multi_wall_bottom_hard_1,
#     adv_multi_wall_right_hard_1,
#     adv_multi_ring_closing_left,
#     adv_multi_ring_closing_right,
#     adv_multi_two_rings_closing,
#     avg_multi_ring_closing_both2,
#     adv_multi_ring_closing_both_inside,
#     adv_multi_ring_closing_both_inside_fast
# ]

xfc2024 = [
    adv_random_small_1,
    adv_random_small_1_2,
    adv_multi_wall_left_easy,
    adv_multi_four_corners,
    adv_multi_wall_top_easy,
    adv_multi_2wall_closing,
    adv_wall_bottom_staggered,
    adv_multi_wall_right_hard,
    adv_moving_corridor_angled_1,
    adv_moving_corridor_angled_1_mines,
    adv_multi_ring_closing_left,
    adv_multi_ring_closing_left2,
    adv_multi_ring_closing_both2,
    adv_multi_ring_closing_both_inside_fast,
    adv_multi_two_rings_closing
]

custom_scenarios = [
    target_priority_optimization1,
    closing_ring_scenario,
    easy_closing_ring_scenario,
    more_intense_closing_ring_scenario,
    rotating_square_scenario,
    rotating_square_2_overlap,
    falling_leaves_scenario,
    zigzag_motion_scenario,
    shearing_pattern_scenario,
    super_hard_wrap,
    wonky_ring,
    moving_ring_scenario,
    shifting_square_scenario,
    delayed_closing_ring_scenario,
    spiral_assault_scenario,
    dancing_ring,
    dancing_ring_2,
    intersecting_lines_scenario,
    exploding_grid_scenario,
    grid_formation_explosion_scenario,
    aspect_ratio_grid_formation_scenario,
    adv_asteroid_stealing,
    wrapping_nightmare,
    wrapping_nightmare_fast,
    purgatory,
    cross,
    fight_for_asteroid,
    shot_pred_test,
    shredder,
    diagonal_shredder,
    out_of_bound_mine,
    explainability_1,
    explainability_2,
    split_forecasting,
    minefield_maze_scenario,
    wrap_collision_test
]

rand_scenarios = []
for ind, num_ast in enumerate(range(1, 50)):
    random.seed(ind)
    randomly_generated_asteroids = generate_asteroids(
                            num_asteroids=num_ast,
                            position_range_x=(0, width),
                            position_range_y=(0, height),
                            speed_range=(-300, 600, 0),
                            angle_range=(-1, 361),
                            size_range=(1, 4)
                        )
    total_asts = 0
    for a in randomly_generated_asteroids:
        total_asts += ASTEROID_COUNT_LOOKUP[a['size']]
    #print(total_asts)
    rand_scenario = Scenario(name=f'Random Scenario {num_ast}',
                                asteroid_states=randomly_generated_asteroids,
                                ship_states=[
                                    {'position': (width/3, height/2), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 5},
                                    {'position': (width*2/3, height/2), 'angle': 180, 'lives': 6, 'team': 2, "mines_remaining": 5},
                                ],
                                map_size=(width, height),
                                time_limit=total_asts/10*0.8,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)
    rand_scenarios.append(rand_scenario)

score = None
died = False
team_1_hits = 0
team_2_hits = 0
team_1_deaths = 0
team_2_deaths = 0
team_1_wins = 0
team_2_wins = 0
team_1_shot_efficiency = 0
team_2_shot_efficiency = 0
team_1_shot_efficiency_including_mines = 0
team_2_shot_efficiency_including_mines = 0
team_1_bullets_hit = 0
team_1_shots_fired = 0
team_2_bullets_hit = 0
team_2_shots_fired = 0

selected_portfolio = [None]
if args.portfolio is not None:
    match args.portfolio:
        case 'xfc2023':
            selected_portfolio = xfc2023
        case 'xfc2021':
            selected_portfolio = xfc_2021_portfolio
        case 'xfc2021alt':
            selected_portfolio = alternate_scenarios
        case 'custom':
            selected_portfolio = custom_scenarios
        case 'rand_scenarios':
            selected_portfolio = rand_scenarios
        case 'xfc2024':
            selected_portfolio = xfc2024

random.seed()

while True:
    for scenario in selected_portfolio[0 if not args.index else args.index:]:
        iterations += 1
        if args.seed is not None:
            randseed = args.seed
        else:
            pass
            randseed = random.randint(1, 1000000000)
        color_print(f'\nUsing seed {randseed}, running test iteration {iterations}', 'green')
        random.seed(randseed)
        #controllers_used = [NeoController(), XFC2024NeoController()]
        
        #controllers_used = [XFC2024NeoController(), NeoController()]
        #controllers_used = [NeoController(), BabyNeoController()]

        asteroids_random = generate_asteroids(
                                        num_asteroids=random.randint(20, 30),
                                        position_range_x=(0, width),
                                        position_range_y=(0, height),
                                        speed_range=(-300, 300, 0),
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
                                        {'position': (width//3, height//2), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                        {'position': (width*2//3, height//2), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                                    ],
                                    map_size=(width, height),
                                    time_limit=10.0,
                                    ammo_limit_multiplier=random.choice([0]),
                                    stop_if_no_ammo=False)
        random.seed(randseed)
        benchmark_scenario = Scenario(name="Benchmark Scenario",
                                        num_asteroids=100,
                                        ship_states=[
                                            {'position': (width/2, height/2), 'angle': 0.0, 'lives': 10000, 'team': 1, 'mines_remaining': 10000},
                                        ],
                                        map_size=(width, height),
                                        seed=0,
                                        time_limit=120)

        pre = time.perf_counter()
        if scenario is not None:
            print(f"Evaluating scenario {scenario.name}")
        else:
            print("Evaluating random scenario")
        profile = args.profile
        # my_test_scenario
        # ex_adv_four_corners_pt1 ex_adv_asteroids_down_up_pt1 ex_adv_asteroids_down_up_pt2 adv_multi_wall_bottom_hard_1 
        # closing_ring_scenario more_intense_closing_ring_scenario rotating_square_scenario falling_leaves_scenario shearing_pattern_scenario zigzag_motion_scenario
        #state = 
        #random.seed(randseed)
         # [ReplayController0(), ReplayController1()] GamepadController()])#, NeoController()])#, TestController()])GamepadController NeoController Neo
        #controllers_used = [NeoController(), NeoController()]
        #random.setstate(state)
        #print(f"RNG State: {random.getstate()}")
        #score, perf_data = game.run(scenario=ex_adv_four_corners_pt1, controllers=controllers_used)
        #score, perf_data = game.run(scenario=ex_adv_four_corners_pt2, controllers=controllers_used)
        if scenario is None:
            scenario_to_run = rand_scenario
        else:
            scenario_to_run = scenario
        if args.scenario:
            eval(f"game.run(scenario={args.scenario}, controllers=controllers_used)")
        else:
            if profile:
                cProfile.run(f'game.run(scenario=scenario_to_run, controllers=controllers_used)')
            else:
                random.seed(randseed)
                score, perf_data = game.run(scenario=scenario_to_run, controllers=controllers_used)

        # Print out some general info about the result
        num_teams = len(score.teams)
        if score:
            team1 = score.teams[0]
            if num_teams > 1:
                team2 = score.teams[1]
            asts_hit = [team.asteroids_hit for team in score.teams]
            color_print('Scenario eval time: '+str(time.perf_counter()-pre), 'green')
            color_print(score.stop_reason, 'green')
            color_print(f"Scenario in-game time: {score.sim_time:.02f} s", 'green')
            color_print('Asteroids hit: ' + str(asts_hit), 'green')
            team_1_hits += asts_hit[0]
            if num_teams > 1:
                team_2_hits += asts_hit[1]
                if asts_hit[0] > asts_hit[1]:
                    team_1_wins += 1
                elif asts_hit[0] < asts_hit[1]:
                    team_2_wins += 1
            else:
                team_1_wins += 1
            team_deaths = [team.deaths for team in score.teams]
            team_1_deaths += team_deaths[0]
            if num_teams > 1:
                team_2_deaths += team_deaths[1]
            color_print('Deaths: ' + str(team_deaths), 'green')
            if team_deaths[0] >= 1:
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
            team_1_shot_efficiency = (team1.bullets_hit/score.sim_time)/(1/(1/10))
            team_1_shot_efficiency_including_mines = (team1.asteroids_hit/score.sim_time)/(1/(1/10))
            if num_teams > 1:
                team_2_shot_efficiency = (team2.bullets_hit/score.sim_time)/(1/(1/10))
                team_2_shot_efficiency_including_mines = (team2.asteroids_hit/score.sim_time)/(1/(1/10))
                team_1_bullets_hit += team1.bullets_hit
                team_2_bullets_hit += team2.bullets_hit
                team_1_shots_fired += team1.shots_fired
                team_2_shots_fired += team2.shots_fired
        print(f"Team 1, 2 hits: ({team_1_hits}, {team_2_hits})")
        print(f"Team 1, 2 wins: ({team_1_wins}, {team_2_wins})")
        print(f"Team 1, 2 deaths: ({team_1_deaths}, {team_2_deaths})")
        print(f"Team 1, 2 accuracies: ({team_1_bullets_hit/(team_1_shots_fired + 0.000000000000001)}, {team_2_bullets_hit/(team_2_shots_fired + 0.000000000000001)})")
        print(f"Team 1, 2 shot efficiencies: ({team_1_shot_efficiency:.02%}, {team_2_shot_efficiency:.02%})")
        print(f"Team 1, 2 shot efficiencies inc. mines/ship hits: ({team_1_shot_efficiency_including_mines:.02%}, {team_2_shot_efficiency_including_mines:.02%})")
        #if args.once:
        #    break
    if missed and len(team_deaths) == 1:
        color_print(f"Ran {iterations} simulations to get one where Neo missed!", 'green')
        break

    if args.once:
        break
    #break
    if iterations == 0:
        print("No scenario to run!")
        break

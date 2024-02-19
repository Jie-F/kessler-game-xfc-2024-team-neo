import numpy as np
import json
import shutil  # For copying the file
from datetime import datetime
import random
import time

from neo_controller import Neo
from baby_neo_controller import NeoController
import numpy as n
import sys
from r_controller import RController
from null_controller import NullController
from scenarios import *
from xfc_2023_replica_scenarios import *
from src.kesslergame import Scenario, KesslerGame, GraphicsType

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1) # Fixes blurriness when a scale factor is used in Windows

def generate_asteroids(num_asteroids, position_range_x, position_range_y, speed_range, angle_range, size_range):
    asteroids = []
    for _ in range(num_asteroids):
        position = (random.uniform(*position_range_x), random.uniform(*position_range_y))
        speed = random.triangular(*speed_range)
        angle = random.uniform(*angle_range)
        size = random.randint(*size_range)
        asteroids.append({'position': position, 'speed': speed, 'angle': angle, 'size': size})
    return asteroids

def backup_existing_results(filename="ga_results.json"):
    try:
        # Generate backup filename with timestamp
        backup_filename = f"{filename[:-5]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        # Copy the existing JSON file to the backup file
        shutil.copyfile(filename, backup_filename)
        print(f"Backup created: {backup_filename}")
    except FileNotFoundError:
        print("No existing file to backup.")

def load_existing_results(filename="ga_results.json"):
    try:
        with open(filename, 'r', encoding='utf8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("No existing results file found. Starting fresh.")
        return []

def save_results_incrementally(result, filename="ga_results.json"):
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(result, f, indent=4)

def run_training(training_portfolio, filename="ga_results.json"):
    # Backup and load existing results
    backup_existing_results(filename)
    results = load_existing_results(filename)
    iteration = 0
    # Define Game Settings
    game_settings = {'perf_tracker': True,
                    'graphics_type': GraphicsType.NoGraphics,#UnrealEngine,Tkinter,NoGraphics
                    'realtime_multiplier': 0,
                    'graphics_obj': None,
                    'frequency': 30.0,
                    'UI_settings': 'all'}

    game = KesslerGame(settings=game_settings)
    while True:
        iteration += 1
        scenarios_info = []
        random_chromosome = generate_normalized_chromosome()
        print(f"Using chromosome: {random_chromosome}")
        random.seed(1)
        team_1_hits = 0
        team_2_hits = 0
        team_1_deaths = 0
        team_2_deaths = 0
        team_1_wins = 0
        team_2_wins = 0
        for sc in training_portfolio:
            controllers_used = [Neo(random_chromosome), NeoController()]
            print(f"Evaluating scenario {sc.name}")
            pre = time.perf_counter()
            score, perf_data = game.run(scenario=sc, controllers=controllers_used)
            asts_hit = [team.asteroids_hit for team in score.teams]
            print('Scenario eval time: '+str(time.perf_counter()-pre))
            print(score.stop_reason)
            print('Asteroids hit: ' + str(asts_hit))
            team_1_hits += asts_hit[0]
            team_2_hits += asts_hit[1]
            if asts_hit[0] > asts_hit[1]:
                team_1_wins += 1
            elif asts_hit[0] < asts_hit[1]:
                team_2_wins += 1
            team_deaths = [team.deaths for team in score.teams]
            team_1_deaths += team_deaths[0]
            team_2_deaths += team_deaths[1]
            scenarios_info.append({'Name': sc.name,
                                   'team_1_hits': asts_hit[0],
                                   'team_2_hits': asts_hit[1],
                                   'team_1_wins': 1 if asts_hit[0] > asts_hit[1] else 0,
                                   'team_2_wins': 1 if asts_hit[0] < asts_hit[1] else 0,
                                   'team_1_deaths': team_deaths[0],
                                   'team_2_deaths': team_deaths[1]})
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'chromosome': random_chromosome,
            'scenarios_run': scenarios_info,
            'team_1_hits': team_1_hits,
            'team_2_hits': team_2_hits,
            'team_1_deaths': team_1_deaths,
            'team_2_deaths': team_2_deaths,
            'team_1_wins': team_1_wins,
            'team_2_wins': team_2_wins,
        }
        results.append(run_info)
        # Save incrementally
        save_results_incrementally(results, filename)

def generate_normalized_chromosome(chromosome_length=7, target_sum=10):
    random.seed()
    #return [4, 4, 1, 1, 3, 5, 1]
    #return [2.219970248198923, 1.8586284535496254, 0.2721467961735549, 0.21273937276022523, 2.5051100814512584, 1.2159755904145138, 1.7154294574518991]
    random_numbers = np.random.rand(chromosome_length)
    normalization_factor = target_sum / np.sum(random_numbers)
    normalized_chromosome = random_numbers * normalization_factor
    return list(normalized_chromosome)


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

training_portfolio = xfc2023

random.seed(1)

width, height = (1000, 800)

for num_ast in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
    rand_scenario = Scenario(name=f'Random Scenario {num_ast}',
                                asteroid_states=generate_asteroids(
                                        num_asteroids=num_ast,
                                        position_range_x=(0, width),
                                        position_range_y=(0, height),
                                        speed_range=(-300, 600, 0),
                                        angle_range=(-1, 361),
                                        size_range=(1, 4)
                                    ),
                                ship_states=[
                                    {'position': (width//3, height//2), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                    {'position': (width*2//3, height//2), 'angle': 180, 'lives': 3, 'team': 2, "mines_remaining": 3},
                                ],
                                map_size=(width, height),
                                time_limit=600,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)
    training_portfolio.append(rand_scenario)

run_training(training_portfolio, f"training\\{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Training Results.json")

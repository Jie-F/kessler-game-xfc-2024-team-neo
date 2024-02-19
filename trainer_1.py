import numpy as np
import json
import shutil  # For copying the file
from datetime import datetime
import random
import time
import os

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

GA_RESULTS_FILE = "ga_results.json"
TRAINING_DIRECTORY = 'training_v1'

def generate_asteroids(num_asteroids, position_range_x, position_range_y, speed_range, angle_range, size_range):
    asteroids = []
    for _ in range(num_asteroids):
        position = (random.uniform(*position_range_x), random.uniform(*position_range_y))
        speed = random.triangular(*speed_range)
        angle = random.uniform(*angle_range)
        size = random.randint(*size_range)
        asteroids.append({'position': position, 'speed': speed, 'angle': angle, 'size': size})
    return asteroids

def backup_existing_results(filename=GA_RESULTS_FILE):
    try:
        # Generate backup filename with timestamp
        backup_filename = f"{filename[:-5]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        # Copy the existing JSON file to the backup file
        shutil.copyfile(filename, backup_filename)
        print(f"Backup created: {backup_filename}")
    except FileNotFoundError:
        print("No existing file to backup.")

def load_existing_results(filename=GA_RESULTS_FILE):
    try:
        with open(filename, 'r', encoding='utf8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("No existing results file found. Starting fresh.")
        return []

def save_results_incrementally(result, filename=GA_RESULTS_FILE):
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(result, f, indent=4)

def read_and_process_json_files(directory="."):
    all_data = []
    # Iterate through all files in the current directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Check if the file is a JSON file
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf8') as file:
                try:
                    data = json.load(file)
                    if isinstance(data, list):  # Ensure the JSON content is a list
                        all_data.extend(data)
                    else:
                        print(f"File {filename} does not contain a list.")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file {filename}.")
    return all_data

def get_top_chromosomes():
    # Call the function to start processing
    all_data = read_and_process_json_files(TRAINING_DIRECTORY)

    # Initialize an empty list to keep track of top 3 scores
    top_scores = []

    for run in all_data:
        current_score = run['team_1_hits']
        current_chromosome = run['chromosome']
        # Append the current run's score and chromosome to the list
        top_scores.append((current_score, current_chromosome))

    # Sort the list by score in descending order and keep only the top 5
    top_scores = sorted(top_scores, key=lambda x: x[0], reverse=True)[:5]
    top_chromosomes = [chromosome for _, chromosome in top_scores]
    return top_chromosomes

def run_training(training_portfolio, filename=GA_RESULTS_FILE):
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
        random.seed()
        iteration += 1
        scenarios_info = []
        top_chromosomes = get_top_chromosomes()
        rand_decision = random.random()
        if rand_decision < 0.35 or len(top_chromosomes) < 5:
            print('Completely random chromosome')
            # Generate a completely random chromosome
            new_chromosome = generate_random_chromosome()
        elif rand_decision < 0.7:
            print('Mutating top chromosome')
            # Take the top chromosome and apply a mutation
            new_chromosome = normalize(mutate_chromosome(top_chromosomes[0]), 7)
            print(f"Mutated {top_chromosomes[0]} into {new_chromosome}")
        elif rand_decision < 0.85:
            print('Crossovering chromosomes')
            # Take some top chromosomes and crossover them
            child_1, child_2 = crossover_chromosomes(top_chromosomes[0], top_chromosomes[1])
            new_chromosome = random.choice([child_1, child_2])
            print(f"Took parents {top_chromosomes[0]} and {top_chromosomes[1]} to get {new_chromosome}")
        else:
            print('Crossovering rand chromosomes')
            print(top_chromosomes)
            parent1, parent2 = random.sample(top_chromosomes, 2)
            child_1, child_2 = crossover_chromosomes(parent1, parent2)
            new_chromosome = random.choice([child_1, child_2])
            print(f"Took parents {parent1} and {parent2} to get {new_chromosome}")

        print(f"\nNew run using chromosome: {new_chromosome}")
        team_1_hits = 0
        team_2_hits = 0
        team_1_deaths = 0
        team_2_deaths = 0
        team_1_wins = 0
        team_2_wins = 0
        for i in range(2):
            random.seed(i)
            for sc in training_portfolio:
                controllers_used = [Neo(new_chromosome), NeoController()]
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
                                    'randseed': i,
                                    'team_1_hits': asts_hit[0],
                                    'team_2_hits': asts_hit[1],
                                    'team_1_wins': 1 if asts_hit[0] > asts_hit[1] else 0,
                                    'team_2_wins': 1 if asts_hit[0] < asts_hit[1] else 0,
                                    'team_1_deaths': team_deaths[0],
                                    'team_2_deaths': team_deaths[1]})
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'chromosome': new_chromosome,
            'team_1_hits': team_1_hits,
            'team_2_hits': team_2_hits,
            'team_1_deaths': team_1_deaths,
            'team_2_deaths': team_2_deaths,
            'team_1_wins': team_1_wins,
            'team_2_wins': team_2_wins,
            'scenarios_run': scenarios_info,
        }
        results.append(run_info)
        # Save incrementally
        save_results_incrementally(results, filename)

def generate_random_numbers(length, lower_bound=0, upper_bound=1):
    return [random.uniform(lower_bound, upper_bound) for _ in range(length)]

def normalize(numbers, target_sum):
    sum_numbers = sum(numbers)
    return [number / sum_numbers * target_sum for number in numbers]

def generate_random_chromosome(chromosome_length=7, target_sum=7):
    random.seed()
    random_numbers = generate_random_numbers(chromosome_length)
    normalized_chromosome = normalize(random_numbers, target_sum)
    return normalized_chromosome

def mutate_chromosome(chromosome, mutation_rate=0.2, mutation_strength=0.3):
    mutation_occurred = False

    while not mutation_occurred:
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:  # Apply mutation with a certain probability
                # Add a random offset within the range [-mutation_strength, mutation_strength]
                offset = random.uniform(-mutation_strength, mutation_strength)
                chromosome[i] += offset
                chromosome[i] = max(chromosome[i], 0)
                mutation_occurred = True
    return chromosome

def crossover_chromosomes(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)  # Choose a crossover point
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

xfc2023 = [
    ex_adv_four_corners_pt1,
    ex_adv_asteroids_down_up_pt1,
    ex_adv_asteroids_down_up_pt2,
    ex_adv_direct_facing,
    ex_adv_two_asteroids_pt1,
    ex_adv_two_asteroids_pt2,
    ex_adv_ring_pt1,
    adv_random_big_1,
    adv_random_big_2,
    adv_random_big_3,
    adv_random_big_4,
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

width, height = (1000, 800)

for ind, num_ast in enumerate([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]):
    random.seed(ind)
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

run_training(training_portfolio, f"{TRAINING_DIRECTORY}\\{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Training 1 Results.json")

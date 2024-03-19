import numpy as np
import json
import shutil
from datetime import datetime
import random
import time
import os
import argparse

from typing import Any
from src.neo_controller_training import NeoController
from src.baby_neo_controller import BabyNeoController
import sys
#from r_controller import RController
#from null_controller import NullController
from scenarios import *
from xfc_2023_replica_scenarios import *
from src.kesslergame import Scenario, KesslerGame, GraphicsType

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1) # Fixes blurriness when a scale factor is used in Windows

GA_RESULTS_FILE = "ga_results.json"
TRAINING_DIRECTORY = 'training_v3'
CHROMOSOME_TUPLE_SIZE = 9
ASTEROID_COUNT_LOOKUP = (0, 1, 4, 13, 40)

if not os.path.exists(TRAINING_DIRECTORY):
    os.makedirs(TRAINING_DIRECTORY, exist_ok=True)

# Command line argument parsing
parser = argparse.ArgumentParser(description='Run genetic algorithm training with an optional custom chromosome.')
parser.add_argument('--chromosome', type=str, help='A custom chromosome to test, formatted as a comma-separated list of values (e.g., "0.1,0.2,0.3,...").', default='')
args = parser.parse_args()

# Convert the custom chromosome string to a list of floats if provided
custom_chromosome = [float(x.strip()) for x in args.chromosome.strip('(){}').split(',')] if args.chromosome else None
if custom_chromosome is not None:
    assert len(custom_chromosome) == 9, "Custom chromosome not the required length of 9!"

def generate_asteroids(num_asteroids, position_range_x, position_range_y, speed_range, angle_range, size_range) -> list:
    asteroids = []
    for _ in range(num_asteroids):
        position = (random.uniform(*position_range_x), random.uniform(*position_range_y))
        speed = random.triangular(*speed_range)
        angle = random.uniform(*angle_range)
        size = random.randint(*size_range)
        asteroids.append({'position': position, 'speed': speed, 'angle': angle, 'size': size})
    return asteroids

# def backup_existing_results(filename=GA_RESULTS_FILE) -> None:
#     try:
#         # Generate backup filename with timestamp
#         backup_filename = f"{filename[:-5]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
#         # Copy the existing JSON file to the backup file
#         shutil.copyfile(filename, backup_filename)
#         print(f"Backup created: {backup_filename}")
#     except FileNotFoundError:
#         print("No existing file to backup.")

# def load_existing_results(filename=GA_RESULTS_FILE) -> Any | list:
#     try:
#         with open(filename, 'r', encoding='utf8') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         print("No existing results file found. Starting fresh.")
#         return []

# def save_results_incrementally(result, filename=GA_RESULTS_FILE) -> None:
#     with open(filename, 'w', encoding='utf8') as f:
#         json.dump(result, f, indent=4)

def read_and_process_json_files(directory=".") -> list:
    all_data = []
    # Iterate through all files in the current directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Check if the file is a JSON file
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf8') as file:
                try:
                    data = json.load(file)
                    if isinstance(data, dict):  # Ensure the JSON content is a dict
                        all_data.append(data)
                    else:
                        print(f"File {filename} does not contain a dict.")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file {filename}.")
    return all_data

def get_top_chromosomes() -> list:
    # Call the function to start processing
    all_data = read_and_process_json_files(TRAINING_DIRECTORY)
    print(f"Getting top chromosomes from {len(all_data)} training runs!")
    # Initialize an empty list to keep track of top 3 scores
    top_scores = []
    
    for run in all_data:
        current_score = run['team_1_total_asteroids_hit']
        current_chromosome = run['chromosome']
        # Append the current run's score and chromosome to the list
        top_scores.append((current_score, current_chromosome))

    # Sort the list by score in descending order and keep only the top 5
    top_scores = sorted(top_scores, key=lambda x: x[0], reverse=True)[:5]
    top_chromosomes = [chromosome for _, chromosome in top_scores]
    return top_chromosomes

def run_training(training_portfolio, directory=TRAINING_DIRECTORY) -> None:#filename=GA_RESULTS_FILE):
    # Backup and load existing results
    #backup_existing_results(filename)
    #results = load_existing_results(filename)
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
        print('\n\n\nNew Training Run!')
        random.seed()
        iteration += 1
        scenarios_info = []
        if custom_chromosome is not None:
            print('Using custom chromosome')
            new_chromosome = custom_chromosome
        else:
            top_chromosomes = get_top_chromosomes()
            rand_decision = random.random()
            if rand_decision < 0.35 or len(top_chromosomes) < 5:
                print('Completely random chromosome')
                # Generate a completely random chromosome
                new_chromosome = generate_random_chromosome()
            elif rand_decision < 0.7:
                print('Mutating top chromosome')
                # Take the top chromosome and apply a mutation
                new_chromosome = mutate_chromosome(top_chromosomes[0], 0.2, 0.1)
                print(f"Mutated {top_chromosomes[0]} into {new_chromosome}")
            elif rand_decision < 0.85:
                print('Crossovering chromosomes')
                # Take some top chromosomes and crossover them
                child_1, child_2 = crossover_chromosomes(top_chromosomes[0], top_chromosomes[1])
                new_chromosome = random.choice([child_1, child_2])
                new_chromosome = mutate_chromosome(new_chromosome, 0.2, 0.05)
                print(f"Took parents {top_chromosomes[0]} and {top_chromosomes[1]} to get {new_chromosome}")
            else:
                print('Crossovering rand chromosomes')
                print(top_chromosomes)
                parent1, parent2 = random.sample(top_chromosomes, 2)
                child_1, child_2 = crossover_chromosomes(parent1, parent2)
                new_chromosome = random.choice([child_1, child_2])
                new_chromosome = mutate_chromosome(new_chromosome, 0.2, 0.05)
                print(f"Took parents {parent1} and {parent2} to get {new_chromosome}")
        new_chromosome = normalize(new_chromosome, 1.0)
        print(f"\nNew run using chromosome: {new_chromosome}")
        team_1_total_hits = 0
        team_2_total_hits = 0
        team_1_deaths = 0
        team_2_deaths = 0
        team_1_wins = 0
        team_2_wins = 0
        total_eval_time_s = 0
        neo_total_sim_ts = 0
        total_sim_time_s = 0.0
        team_1_total_bullets_hit = 0
        team_2_total_bullets_hit = 0
        team_1_total_shots_fired = 0
        team_2_total_shots_fired = 0
        for i in range(3):
            # Run portfolio 3 times, to even out randomness
            for sc in training_portfolio:
                #random.seed(i)
                randseed = random.randint(1, 1000000000)
                random.seed(randseed)
                controllers_used = [NeoController(tuple(new_chromosome)), BabyNeoController()]
                assert controllers_used[0].get_total_sim_ts() == 0
                print(f"\nEvaluating scenario {sc.name} with rng seed {randseed} on total pass number {i + 1} using chromosome {new_chromosome}")
                #print(f"RNG State: {random.getstate()}")
                pre = time.perf_counter()
                score, perf_data = game.run(scenario=sc, controllers=controllers_used)
                post = time.perf_counter()
                neo_sim_ts = controllers_used[0].get_total_sim_ts()
                neo_total_sim_ts += neo_sim_ts
                team_1 = score.teams[0]
                team_2 = score.teams[1]
                asts_hit = [team.asteroids_hit for team in score.teams]
                total_eval_time_s += max(0, post - pre)
                print('Scenario eval time: '+str(post - pre))
                print(score.stop_reason)
                print('Asteroids hit: ' + str(asts_hit))
                team_1_total_hits += team_1.asteroids_hit
                team_2_total_hits += team_2.asteroids_hit
                if team_1.asteroids_hit > team_2.asteroids_hit:
                    team_1_wins += 1
                elif team_1.asteroids_hit < team_2.asteroids_hit:
                    team_2_wins += 1

                team_1_total_bullets_hit += team_1.bullets_hit
                team_2_total_bullets_hit += team_2.bullets_hit
                team_1_total_shots_fired += team_1.shots_fired
                team_2_total_shots_fired += team_2.shots_fired

                team_1_deaths += team_1.deaths
                team_2_deaths += team_2.deaths
                total_sim_time_s += float(score.sim_time)
                assert float(score.sim_time) >= 0.0
                scenarios_info.append({'scenario_name': sc.name,
                                       'timestamp': datetime.now().isoformat(),
                                       'randseed': randseed,
                                       'eval_time_s': max(0, post - pre),
                                       'sim_time_s': score.sim_time,
                                       'neo_sim_ts': neo_sim_ts,
                                       'team_1_total_bullets': team_1.total_bullets,
                                       'team_2_total_bullets': team_2.total_bullets,
                                       'team_1_total_asteroids': team_1.total_asteroids,
                                       'team_2_total_asteroids': team_2.total_asteroids,
                                       'team_1_asteroids_hit': team_1.asteroids_hit,
                                       'team_2_asteroids_hit': team_2.asteroids_hit,
                                       'team_1_bullets_hit': team_1.bullets_hit,
                                       'team_2_bullets_hit': team_2.bullets_hit,
                                       'team_1_shots_fired': team_1.shots_fired,
                                       'team_2_shots_fired': team_2.shots_fired,
                                       'team_1_bullets_remaining': team_1.bullets_remaining,
                                       'team_2_bullets_remaining': team_2.bullets_remaining,
                                       'team_1_deaths': team_1.deaths,
                                       'team_2_deaths': team_2.deaths,
                                       'team_1_lives_remaining': team_1.lives_remaining,
                                       'team_2_lives_remaining': team_2.lives_remaining,
                                       'team_1_accuracy': team_1.accuracy,
                                       'team_2_accuracy': team_2.accuracy,
                                       'team_1_fraction_total_asteroids_hit': team_1.fraction_total_asteroids_hit,
                                       'team_2_fraction_total_asteroids_hit': team_2.fraction_total_asteroids_hit,
                                       'team_1_fraction_bullets_used': team_1.fraction_bullets_used,
                                       'team_2_fraction_bullets_used': team_2.fraction_bullets_used,
                                       'team_1_ratio_bullets_needed': team_1.ratio_bullets_needed,
                                       'team_2_ratio_bullets_needed': team_2.ratio_bullets_needed,
                                       'team_1_mean_eval_time': team_1.mean_eval_time,
                                       'team_2_mean_eval_time': team_2.mean_eval_time,
                                       'team_1_median_eval_time': team_1.median_eval_time,
                                       'team_2_median_eval_time': team_2.median_eval_time,
                                       'team_1_min_eval_time': team_1.min_eval_time,
                                       'team_2_min_eval_time': team_2.min_eval_time,
                                       'team_1_max_eval_time': team_1.max_eval_time,
                                       'team_2_max_eval_time': team_2.max_eval_time,
                                       'team_1_win': 1 if team_1.asteroids_hit > team_2.asteroids_hit else 0,
                                       'team_2_win': 1 if team_1.asteroids_hit < team_2.asteroids_hit else 0
                                       })
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'total_eval_time': total_eval_time_s,
            'total_sim_time': total_sim_time_s,
            'chromosome': new_chromosome,
            'neo_total_sim_ts': neo_total_sim_ts,
            'team_1_name': controllers_used[0].name,
            'team_2_name': controllers_used[1].name,
            'team_1_total_asteroids_hit': team_1_total_hits,
            'team_2_total_asteroids_hit': team_2_total_hits,
            'team_1_total_bullets_hit': team_1_total_bullets_hit,
            'team_2_total_bullets_hit': team_2_total_bullets_hit,
            'team_1_total_shots_fired': team_1_total_shots_fired,
            'team_2_total_shots_fired': team_2_total_shots_fired,
            'team_1_overall_accuracy': team_1_total_bullets_hit/team_1_total_shots_fired,
            'team_2_overall_accuracy': team_2_total_bullets_hit/team_2_total_shots_fired,
            'team_1_deaths': team_1_deaths,
            'team_2_deaths': team_2_deaths,
            'team_1_wins': team_1_wins,
            'team_2_wins': team_2_wins,
            'scenarios_run': scenarios_info,
        }
        
        # Generate a unique filename for this run
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
        unique_filename = f"{directory}/{timestamp}_Training_Run.json"
        # Save this run's results to a separate file
        with open(unique_filename, 'w', encoding='utf8') as f:
            json.dump(run_info, f, indent=4)

        print(f"Results saved to {unique_filename}")
        # Save incrementally
        #save_results_incrementally(results, filename)

        if custom_chromosome:
            print('Finished training with custom chromosome. Exiting.')
            break

def generate_random_numbers(length, lower_bound=0, upper_bound=1) -> list[float]:
    return [random.uniform(lower_bound, upper_bound) for _ in range(length)]

def normalize(numbers, target_sum) -> list:
    sum_numbers = sum(numbers)
    scale_ratio = target_sum / sum_numbers
    return [number*scale_ratio  for number in numbers]

def generate_random_chromosome(chromosome_length=CHROMOSOME_TUPLE_SIZE, target_sum=1.0) -> list:
    random.seed()
    random_numbers = generate_random_numbers(chromosome_length)
    #print(random_numbers)
    normalized_chromosome = normalize(random_numbers, target_sum)
    #print(normalized_chromosome)
    return normalized_chromosome

def mutate_chromosome(chromosome, mutation_rate=0.2, mutation_strength=0.1) -> Any:
    mutation_occurred = False
    chromosome = chromosome.copy()
    while not mutation_occurred:
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:  # Apply mutation with a certain probability
                # Add a random offset within the range [-mutation_strength, mutation_strength]
                offset = random.uniform(-mutation_strength, mutation_strength)
                original_gene = chromosome[i]
                chromosome[i] = max(chromosome[i] + offset, 0)
                if original_gene != chromosome[i]:
                    mutation_occurred = True
                else:
                    mutation_occurred = False
    return chromosome

def crossover_chromosomes(parent1, parent2) -> tuple:
    crossover_point = np.random.randint(1, len(parent1) - 1)  # Choose a crossover point
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

training_portfolio = []

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

xfc_2021_show_portfolio = [
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

training_portfolio.extend(xfc2023)

width, height = (1000, 800)
easyrand = []
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
    #if ind < 3:
    #    easyrand.append(rand_scenario)
#training_portfolio = easyrand
training_portfolio.extend(rand_scenarios)

run_training(training_portfolio)#, f"{TRAINING_DIRECTORY}\\{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Training Results.json")

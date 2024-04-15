# _____   __           
# ___  | / /__________ 
# __   |/ /_  _ \  __ \
# _  /|  / /  __/ /_/ /
# /_/ |_/  \___/\____/ 

# XFC 2024 Kessler controller
# Jie Fan (jie.f@pm.me)
# Feel free to reach out if you have questions or want to discuss anything!

# This controller is meant to be compiled by mypyc into a compiled C library. During development this can be imported as a normal Python script, but mypyc gives 3-4X speed improvement.

# The reason I don't just hide these behind a boolean flag, is that the mypyc and C compilers aren't smart enough to remove the dead code, and in hot loops, it has to do the boolean check each time which slows down the code

# TODO: Show stats at the end
# TODO: Verify that frontrun protection's working, because it still feels like it's not totally working!
# DONE: Make it so during a respawn maneuver, if I'm no longer gonna hit anything, I can begin to shoot!
# TODO: Use the tolerance in the shot for the target selection so I don't just aim for the center all the time
# KINDA DONE: Add error handling as a catch-all
# TODO: Analyze each base state, and store analysis results. Like the heuristic FIS, except use more random search. Density affects the movement speed and cruise timesteps. Tune stuff much better.
# TODO: Add error checks, so that if Neo thinks its done but there's still asteroids left, it'll realize that and re-ingest the updated state and finish off its job. This should NEVER happen though, but in the 1/1000000 chance a new bug happens during the competition, this would catch it.
# DONE: Tune gc to maybe speed stuff up
# DONE: Match collision checks with Kessler, including <= vs <
# TODO: Add iteration boosting algorithm to do more iterations in critical moments
# DONE: If we're chilling, have mines and lives, go and do some damage!
# WON'T FIX: Optimally, the target selection will consider mines blowing up asteroids, and having forecasted asteroids there. But this is a super specific condition and it's very complex to implement correctly, so maybe let's just skip this lol. It's probably not worth spending 50 hours implementing something that will rarely come up, and there's plenty of other asteroids I can shoot, and not just ones coming off of a mine blast.
# WON'T FIX: Remove unnecessary class attributes such as ship thrust range, asteroid mass, to speed up class creation and copying
# DONE: Differentiate between my mine and adversary's mine, to know whether to shoot size 1's or not
# TODO: Mine FIS currently doesn't take into account if an asteroid will ALREADY get hit by a mine, and drop another one anyway
# DONE: When validating a sim is good when there's another ship, make sure the shots hit! The other ship might have shot those asteroids already.
# TODO: Try a wider random search for maneuvers
# TODO: GA to beat the random maneuver search, and narrow down the search space. In crowded areas, don't cruise for as long, for example!
# TRIED, not faster: Add per-timestep velocities to asteroids and bullets and stuff to save a multiplication
# TODO: Revisit the aimbot and improve things more
# TODO: If we're gonna die and we're the only ship left, don't shoot a bullet if it doesn't land before I die, because it'll count as a miss
# TODO: Use math to see how the bullet lines up with the asteroid, to predict whether it's gonna hit before doing the bullet sim


import bisect
import gc
import math
import random
import time
from collections import deque
from functools import lru_cache
from itertools import chain
from math import acos, asin, atan2, ceil, cos, exp, floor, inf, isinf, isnan, nan, pi, sin, sqrt
from typing import Any, Final, Iterable, Optional, Sequence, TypedDict, cast

import matplotlib.patches as patches  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
from numpy import arange, fmax, linspace, vectorize
from mypy_extensions import i64
from skfuzzy import control, trimf  # type: ignore[import-untyped]

#from src.kesslergame import KesslerController  # type: ignore[import-untyped]
#from kesslergame import KesslerController  # type: ignore[import-untyped]

#gc.set_debug(gc.DEBUG_STATS)
gc.set_threshold(50000)
#gc.disable()

# IMPORTANT: if multiple scenarios are run back-to-back, this controller doesn't get freshly initialized in the subsequent runs.
# If any global variables are changed during execution, make sure to reset them when the timestep is 0.

# Output config
DEBUG_MODE: Final[bool] = False
PRINT_EXPLANATIONS: Final[bool] = True
EXPLANATION_MESSAGE_SILENCE_INTERVAL_S: Final[float] = 6.0  # Repeated messages within this time window get silenced

# These can trade off to get better performance at the expense of safety
STATE_CONSISTENCY_CHECK_AND_RECOVERY = True  # Enable this if we want to be able to recover from controller exceptions
CLEAN_UP_STATE_FOR_SUBSEQUENT_SCENARIO_RUNS = True  # If NeoController is only instantiated once and run through multiple scenarios, this must be on!
ENABLE_SANITY_CHECKS: Final[bool] = False  # Miscellaneous sanity checks throughout the code
PRUNE_SIM_STATE_SEQUENCE: Final[bool] = True  # Good to have on, because we don't really need the full state
VALIDATE_SIMULATED_KEY_STATES: Final[bool] = False  # Check for desyncs between Kessler and Neo's internal simulation of the game
VALIDATE_ALL_SIMULATED_STATES: Final[bool] = False  # Super meticulous check for desyncs. This is very slow! Not recommended, since just verifying the key states will catch desyncs eventually. This is only good for if you need to know exactly when the desync occurred.
VERIFY_AST_TRACKING: Final[bool] = False  # I'm using a very error prone way to track asteroids, where I very easily get the time of the asteroid wrong. This will check to make sure the times aren't mismatched, by checking whether the asteroid we're looking for appears in the wrong timestep.

# Strategic variables
ADVERSARY_ROTATION_TIMESTEP_FUDGE: Final[i64] = 20  # Since we can't predict the adversary ship, in the targetting frontrun protection, fudge the adversary's ship to be more conservative. Since we predict they don't move, but they could be aiming toward the target.
# TODO: Actually wait, doesn't the rotation timestep fudge just need to be 5, because each stationary targetting is just 5 timesteps long? So using 20 may be overkill!
UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON: Final[float] = 8.0
UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON: Final[float] = 2.3 # The upper bound would be sqrt(1000.0**2 + 800.0**2)/800.0 + 1.0 = 2.600781059358212 s, which is 1 second to turn and the rest for bullet travel time. But this is the worst case scenario. In most cases, we don't need this much.
ASTEROID_SIZE_SHOT_PRIORITY: Final = (nan, 1, 2, 3, 4)  # Index i holds the priority of shooting an asteroid of size i (the first element is not important)
fitness_function_weights: Optional[tuple[float, float, float, float, float, float, float, float, float]] = None
MINE_DROP_COOLDOWN_FUDGE_TS: Final[i64] = 60  # We can drop a mine every 30 timesteps. But it's better to wait a bit longer between mines, so then if I drop two and the first one blows me up, I have time to get out of the radius of the second blast!
MINE_ASTEROID_COUNT_FUDGE_DISTANCE: Final[float] = 50.0
MINE_OPPORTUNITY_CHECK_INTERVAL_TS: Final[i64] = 10
MINE_OTHER_SHIP_RADIUS_FUDGE: Final[float] = 40.0
MINE_OTHER_SHIP_ASTEROID_COUNT_EQUIVALENT: Final[i64] = 10
TARGETING_AIMING_UNDERTURN_ALLOWANCE_DEG: Final[float] = 6.0
# (asteroid_safe_time_fitness, mine_safe_time_fitness, asteroids_fitness, sequence_length_fitness, other_ship_proximity_fitness, crash_fitness, asteroid_aiming_cone_fitness, placed_mine_fitness, overall_safe_time_fitness)
DEFAULT_FITNESS_WEIGHTS: Final = (0.0, 0.12522228730851412, 0.15550196392058346, 0.0, 0.013734028404994915, 0.24604614326339902, 0.16593186495503653, 0.06818149764794686, 0.22538221449952509) # Hand picked: (7.0, 10.0, 1.5, 0.5, 6.0, 12.0, 0.5, 2.0, 7.0)
MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF: Final[float] = 45.0  # I'd expect the smaller this is, the faster. But apparently 30 can be slower than 45 for some reason. So I'll leave it on 45 lol
MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF_COSINE: Final[float] = cos(math.radians(MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF))
MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF: Final[float] = 60.0
MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF_COSINE: Final[float] = cos(math.radians(MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF))
MAX_CRUISE_TIMESTEPS: Final[float] = 30.0
MANEUVER_TUPLE_LEARNING_ROLLING_AVERAGE_PERIOD: Final[i64] = 10
OVERALL_FITNESS_ROLLING_AVERAGE_PERIOD: Final[i64] = 5
AIMING_CONE_FITNESS_CONE_WIDTH_HALF: Final[float] = 18.0 # This has to do with how fast we turn vs how often we can shoot. This can be thought of as some percentage of DEGREES_BETWEEN_SHOTS defined later
AIMING_CONE_FITNESS_CONE_WIDTH_HALF_COSINE: Final[float] = cos(math.radians(AIMING_CONE_FITNESS_CONE_WIDTH_HALF))
MANEUVER_SIM_DISALLOW_TARGETING_FOR_START_TIMESTEPS_AMOUNT: Final[i64] = 10
ASTEROID_AIM_BUFFER_PIXELS: Final[float] = 1.0  # px
COORDINATE_BOUND_CHECK_PADDING: Final[float] = 1.0  # px
SHIP_AVOIDANCE_PADDING: Final = 25
SHIP_AVOIDANCE_SPEED_PADDING_RATIO: Final[float] = 1/100
PERFORMANCE_CONTROLLER_ROLLING_AVERAGE_FRAME_INTERVAL: Final[i64] = 10
# The reason we need this, is because without it, I'm checking whether I have the budget to do another iteration, but let's say I'm already taking 0 time. Kessler will pause for DELTA_TIME, taking up all the time, and I'd think I have no time to do anything!
# So this fudge allows us to "push things" a bit, and fill up any remaining time so that Kessler is no longer pausing, and that time is spent in this controller to do more computations.
# The side effect is that if we set this multiplier too low, then we're going to be operating at less than real time for sure.
# Going slightly slower than real time is expected, since we always need to push things to know where that limit is
# This isn't needed if the realtime multiplier is set to 0 within Kessler, so that Kessler is not unnecessarily pausing and running as fast as it can. But with realtime multiplier at 1, this is needed.
# If realtime multiplier is set to some wacky value above 1, the game won't speed up and Neo will just use that time to do more iterations! To fix that, we need to change this multiplier to be above 1 too
# You can kind of think of this as the percentage of real time we're going at. Kinda...
# Also my logic is that if I always make sure I have enough time, then I’ll actually be within budget. Because say I take 10 time to do something. Well if I have 10 time left, I do it, but anything from 9 to 0 time left, I don’t. So on average, I leave out 10/2 time on the table. So that’s why I set the fudge multiplier to 0.5, so things average out to me being exactly on budget.
PERFORMANCE_CONTROLLER_PUSHING_THE_ENVELOPE_FUDGE_MULTIPLIER: Final[float] = 0.55
MINIMUM_DELTA_TIME_FRACTION_BUDGET: Final[float] = 0.5
ENABLE_PERFORMANCE_CONTROLLER: Final[bool] = True  # The performance controller uses realtime, so it's nondeterministic. For debugging and using set random seeds, turn this off so the controller is determinstic again

# For the tuples below, the index is the number of lives Neo has left while going into the move
# Index 0 in the tuples is not used, but to be safe I put a sane number there

MAX_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS: Final[i64] = 100
MAX_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS: Final[i64] = 100
# For each row of the lookup table, the index in the row corresponds to the number of lives left, minus one
MIN_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS_LUT: Final = ((21, 17, 14), # Fitness from 0.0 to 0.1
                                                         (20, 16, 13), # Fitness from 0.1 to 0.2
                                                         (19, 15, 12), # Fitness from 0.2 to 0.3
                                                         (18, 14, 11), # Fitness from 0.3 to 0.4
                                                         (17, 13, 10), # Fitness from 0.4 to 0.5
                                                         (16, 12, 9), # Fitness from 0.5 to 0.6
                                                         (15, 11, 8), # Fitness from 0.6 to 0.7
                                                         (14, 10, 7), # Fitness from 0.7 to 0.8
                                                         (13, 9, 6), # Fitness from 0.8 to 0.9
                                                         (12, 8, 5)) # Fitness from 0.9 to 1.0
MIN_RESPAWN_PER_PERIOD_SEARCH_ITERATIONS_LUT: Final = ((960, 780, 440), # Fitness from 0.0 to 0.1
                                                       (930, 760, 430), # Fitness from 0.1 to 0.2
                                                       (900, 740, 420), # Fitness from 0.2 to 0.3
                                                       (870, 720, 410), # Fitness from 0.3 to 0.4
                                                       (840, 700, 400), # Fitness from 0.4 to 0.5
                                                       (810, 680, 390), # Fitness from 0.5 to 0.6
                                                       (790, 660, 380), # Fitness from 0.6 to 0.7
                                                       (760, 640, 370), # Fitness from 0.7 to 0.8
                                                       (730, 620, 360), # Fitness from 0.8 to 0.9
                                                       (700, 600, 350)) # Fitness from 0.9 to 1.0
MIN_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS_LUT: Final = ((13, 12, 11), # Fitness from 0.0 to 0.1
                                                          (12, 11, 10), # Fitness from 0.1 to 0.2
                                                          (11, 10, 9), # Fitness from 0.2 to 0.3
                                                          (10, 9, 8), # Fitness from 0.3 to 0.4
                                                          (9, 8, 7), # Fitness from 0.4 to 0.5
                                                          (8, 7, 6), # Fitness from 0.5 to 0.6
                                                          (7, 6, 5), # Fitness from 0.6 to 0.7
                                                          (6, 5, 4), # Fitness from 0.7 to 0.8
                                                          (4, 3, 3), # Fitness from 0.8 to 0.9
                                                          (3, 3, 2)) # Fitness from 0.9 to 1.0
MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_LUT = ((39, 36, 33), # Fitness from 0.0 to 0.1
                                                 (36, 33, 30), # Fitness from 0.1 to 0.2
                                                 (33, 30, 27), # Fitness from 0.2 to 0.3
                                                 (30, 27, 24), # Fitness from 0.3 to 0.4
                                                 (27, 24, 21), # Fitness from 0.4 to 0.5
                                                 (24, 21, 18), # Fitness from 0.5 to 0.6
                                                 (21, 18, 15), # Fitness from 0.6 to 0.7
                                                 (18, 15, 12), # Fitness from 0.7 to 0.8
                                                 (12, 9, 9), # Fitness from 0.8 to 0.9
                                                 (9, 9, 6)) # Fitness from 0.9 to 1.0
MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_IF_WILL_DIE_LUT = ((860, 680, 340), # Fitness from 0.0 to 0.1
                                                             (830, 660, 330), # Fitness from 0.1 to 0.2
                                                             (800, 640, 320), # Fitness from 0.2 to 0.3
                                                             (770, 620, 310), # Fitness from 0.3 to 0.4
                                                             (740, 600, 300), # Fitness from 0.4 to 0.5
                                                             (710, 580, 290), # Fitness from 0.5 to 0.6
                                                             (690, 560, 280), # Fitness from 0.6 to 0.7
                                                             (660, 540, 270), # Fitness from 0.7 to 0.8
                                                             (630, 520, 260), # Fitness from 0.8 to 0.9
                                                             (600, 500, 250)) # Fitness from 0.9 to 1.0

# State dumping for debug
REALITY_STATE_DUMP: Final[bool] = False  # Dump each game state to json
SIMULATION_STATE_DUMP: Final[bool] = False  # Dump each simulated game state to json
KEY_STATE_DUMP: Final[bool] = False  # Dump only key (base) states to json
GAMESTATE_PLOTTING = False  # For whatever reason having this set to final gives a mypyc compilation error
BULLET_SIM_PLOTTING: Final[bool] = False
NEXT_TARGET_PLOTTING = False  # For whatever reason having this set to final gives a mypyc compilation error
MANEUVER_SIM_PLOTTING: Final[bool] = False
START_GAMESTATE_PLOTTING_AT_SECOND: Final[float] = 0.0
NEW_TARGET_PLOT_PAUSE_TIME_S: Final[float] = 0.5
SLOW_DOWN_GAME_AFTER_SECOND: Final[float] = inf
SLOW_DOWN_GAME_PAUSE_TIME: Final[float] = 2.0

# Quantities
TAD: Final[float] = 0.1
GRAIN: Final[float] = 0.001
EPS: Final[float] = 0.0000000001
INT_NEG_INF: Final[i64] = -1000000
INT_INF: Final[i64] = 1000000

RAD_TO_DEG: Final[float] = 180.0/pi
DEG_TO_RAD: Final[float] = pi/180.0
TAU: Final[float] = 2.0*pi

# Kessler game constants
FIRE_COOLDOWN_TS: Final[i64] = 3
MINE_COOLDOWN_TS: Final[i64] = 30
FPS: Final[float] = 30.0
DELTA_TIME: Final[float] = 1/FPS  # s/ts
SHIP_FIRE_TIME: Final[float] = 1/10  # seconds
BULLET_SPEED: Final[float] = 800.0  # px/s
BULLET_MASS: Final[float] = 1.0  # kg
BULLET_LENGTH: Final[float] = 12.0  # px
BULLET_LENGTH_RECIPROCAL: Final[float] = 1.0/BULLET_LENGTH
TWICE_BULLET_LENGTH_RECIPROCAL: Final[float] = 2.0/BULLET_LENGTH
SHIP_MAX_TURN_RATE: Final[float] = 180.0  # deg/s
SHIP_MAX_TURN_RATE_RAD: Final[float] = math.radians(SHIP_MAX_TURN_RATE)
SHIP_MAX_TURN_RATE_RAD_RECIPROCAL: Final[float] = 1.0/SHIP_MAX_TURN_RATE_RAD
SHIP_MAX_TURN_RATE_DEG_TS: Final[float] = DELTA_TIME*SHIP_MAX_TURN_RATE
SHIP_MAX_TURN_RATE_RAD_TS: Final[float] = math.radians(SHIP_MAX_TURN_RATE_DEG_TS)
SHIP_MAX_THRUST: Final[float] = 480.0  # px/s^2
SHIP_DRAG: Final[float] = 80.0  # px/s^2
SHIP_MAX_SPEED: Final[float] = 240.0  # px/s
SHIP_RADIUS: Final[float] = 20.0  # px
SHIP_MASS: Final[float] = 300.0  # kg
TIMESTEPS_UNTIL_SHIP_ACHIEVES_MAX_SPEED: Final[i64] = ceil(SHIP_MAX_SPEED/(SHIP_MAX_THRUST - SHIP_DRAG)*FPS)  # Should be 18 timesteps
MINE_BLAST_RADIUS: Final[float] = 150.0  # px
MINE_RADIUS: Final[float] = 12.0  # px
MINE_BLAST_PRESSURE: Final[float] = 2000.0
MINE_FUSE_TIME: Final[float] = 3.0  # s
MINE_MASS: Final[float] = 25.0  # kg
ASTEROID_RADII_LOOKUP: Final = tuple(8.0*size for size in range(5))  # asteroid.py
ASTEROID_AREA_LOOKUP: Final = tuple(pi*r*r for r in ASTEROID_RADII_LOOKUP)
ASTEROID_MASS_LOOKUP: Final = tuple(0.25*pi*(8*size)**2 for size in range(5))  # asteroid.py
RESPAWN_INVINCIBILITY_TIME_S: Final[float] = 3.0  # s
ASTEROID_COUNT_LOOKUP: Final = (0, 1, 4, 13, 40)  # A size 2 asteroid is 4 asteroids, size 4 is 30, etc. Each asteroid splits into 3, and itself is counted as well. Explicit formula is count(n) = (3^n - 1)/2
DEGREES_BETWEEN_SHOTS: Final[float] = float(FIRE_COOLDOWN_TS)*SHIP_MAX_TURN_RATE*DELTA_TIME

# FIS Settings
ASTEROIDS_HIT_VERY_GOOD: Final[i64] = 65
ASTEROIDS_HIT_OKAY_CENTER: Final = 20

# Dirty globals
# Store messages and their last printed timestep
explanation_messages_with_timestamps: dict[str, i64] = {}  # Make sure to clear this when timestep is 0 so back-to-back runs work properly!
#total_abs_cruise_speed: float = SHIP_MAX_SPEED/2
#total_cruise_timesteps: i64 = round(MAX_CRUISE_TIMESTEPS/2)
#total_maneuvers_to_learn_from: i64 = 1
abs_cruise_speeds: list[float] = [SHIP_MAX_SPEED/2]
cruise_timesteps: list[i64] = [round(MAX_CRUISE_TIMESTEPS/2)]
overall_fitness_record: list[float] = []
#heuristic_fis_iterations = 0
#heuristic_fis_total_fitness = 0.0
#random_search_iterations = 0
#random_search_total_fitness = 0.0
total_sim_timesteps: i64 = 0
#total_bullet_sim_timesteps = 0
#total_bullet_sim_iterations = 0
#update_ts_zero_count = 0
#update_ts_multiple_count = 0
unwrap_cache: dict[i64, list['Asteroid']] = {} #tuple[float, float, float, float, float]
#unwrap_cache_hits: i64 = 0
#unwrap_cache_misses: i64 = 0
#bullet_sim_time: float = 0.0
#sim_update_total_time: float = 0.0
#sim_cull_total_time: float = 0.0
#unwrap_total_time: float = 0.0
#asteroids_pending_death_total_cull_time: float = 0.0
#asteroid_tracking_total_time: float = 0.0
#asteroid_new_track_total_time: float = 0.0


class Asteroid:
    __slots__ = ('position', 'velocity', 'size', 'mass', 'radius', 'timesteps_until_appearance')

    def __init__(self, position: tuple[float, float] = (0.0, 0.0), velocity: tuple[float, float] = (0.0, 0.0), size: i64 = 0, mass: float = 0.0, radius: float = 0.0, timesteps_until_appearance: i64 = 0) -> None:
        self.position = position
        self.velocity = velocity
        self.size = size
        self.mass = mass
        self.radius = radius
        self.timesteps_until_appearance = timesteps_until_appearance

    def __str__(self) -> str:
        return f'Asteroid(position={self.position}, velocity={self.velocity}, size={self.size}, mass={self.mass}, radius={self.radius}, timesteps_until_appearance={self.timesteps_until_appearance})'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Asteroid):
            return NotImplemented
        return (self.position == other.position and
                self.velocity == other.velocity and
                self.size == other.size and
                self.mass == other.mass and
                self.radius == other.radius and
                self.timesteps_until_appearance == other.timesteps_until_appearance)

    def __hash__(self) -> i64:
        # Combine position, velocity, and size into a single float
        combined: float = self.position[0] + 0.4266548291679171*self.position[1] + 0.8164926348982552*self.velocity[0] + 0.8397584399461026*self.velocity[1]
        # Scale to reduce the likelihood of collisions after conversion to integer
        scaled_combined: float = combined * 1_000_000_000
        # Convert to integer
        hash_val = i64(scaled_combined) + self.size
        #print(hash_val)
        return hash_val

    def float_hash(self) -> float:
        # The magic numbers used here were just randomly chosen, to reduce the chance of collisions
        return self.position[0] + 0.4266548291679171*self.position[1] + 0.8164926348982552*self.velocity[0] + 0.8397584399461026*self.velocity[1]

    def int_hash(self) -> i64:
        return i64(1_000_000_000*(self.position[0] + 0.4266548291679171*self.position[1] + 0.8164926348982552*self.velocity[0] + 0.8397584399461026*self.velocity[1]))

    def copy(self) -> 'Asteroid':
        return Asteroid(
            position=self.position,
            velocity=self.velocity,
            size=self.size,
            mass=self.mass,
            radius=self.radius,
            timesteps_until_appearance=self.timesteps_until_appearance
        )


class Ship:
    __slots__ = ('is_respawning', 'position', 'velocity', 'speed', 'heading', 'mass', 'radius', 'id', 'team', 'lives_remaining', 'bullets_remaining', 'mines_remaining', 'can_fire', 'fire_rate', 'can_deploy_mine', 'mine_deploy_rate', 'thrust_range', 'turn_rate_range', 'max_speed', 'drag')

    def __init__(self, is_respawning: bool = False, position: tuple[float, float] = (0.0, 0.0), velocity: tuple[float, float] = (0.0, 0.0), speed: float = 0.0, heading: float = 0.0, mass: float = 0.0, radius: float = 0.0, id: i64 = 0, team: str = '', lives_remaining: i64 = 0, bullets_remaining: i64 = 0, mines_remaining: i64 = 0, can_fire: bool = True, fire_rate: float = 0.0, can_deploy_mine: bool = True, mine_deploy_rate: float = 0.0, thrust_range: tuple[float, float] = (-SHIP_MAX_THRUST, SHIP_MAX_THRUST), turn_rate_range: tuple[float, float] = (-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE), max_speed: float = SHIP_MAX_SPEED, drag: float = SHIP_DRAG) -> None:
        self.is_respawning = is_respawning
        self.position = position
        self.velocity = velocity
        self.speed = speed
        self.heading = heading
        self.mass = mass
        self.radius = radius
        self.id = id
        self.team = team
        self.lives_remaining = lives_remaining
        self.bullets_remaining = bullets_remaining
        self.mines_remaining = mines_remaining
        self.can_fire = can_fire
        self.fire_rate = fire_rate
        self.can_deploy_mine = can_deploy_mine
        self.mine_deploy_rate = mine_deploy_rate
        self.thrust_range = thrust_range
        self.turn_rate_range = turn_rate_range
        self.max_speed = max_speed
        self.drag = drag

    def __str__(self) -> str:
        return f'Ship(is_respawning={self.is_respawning}, position={self.position}, velocity={self.velocity}, speed={self.speed}, heading={self.heading}, mass={self.mass}, radius={self.radius}, id={self.id}, team="{self.team}", lives_remaining={self.lives_remaining}, bullets_remaining={self.bullets_remaining}, mines_remaining={self.mines_remaining}, can_fire={self.can_fire}, fire_rate={self.fire_rate}, can_deploy_mine={self.can_deploy_mine}, mine_deploy_rate={self.mine_deploy_rate}, thrust_range={self.thrust_range}, turn_rate_range={self.turn_rate_range}, max_speed={self.max_speed}, drag={self.drag})'

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'Ship':
        return Ship(
            is_respawning=self.is_respawning,
            position=self.position,
            velocity=self.velocity,
            speed=self.speed,
            heading=self.heading,
            mass=self.mass,
            radius=self.radius,
            id=self.id,
            team=self.team,
            lives_remaining=self.lives_remaining,
            bullets_remaining=self.bullets_remaining,
            mines_remaining=self.mines_remaining,
            can_fire=self.can_fire,
            fire_rate=self.fire_rate,
            can_deploy_mine=self.can_deploy_mine,
            mine_deploy_rate=self.mine_deploy_rate,
            thrust_range=self.thrust_range,
            turn_rate_range=self.turn_rate_range,
            max_speed=self.max_speed,
            drag=self.drag
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ship):
            return NotImplemented
        return (self.is_respawning == other.is_respawning and
                self.position == other.position and
                self.velocity == other.velocity and
                self.speed == other.speed and
                self.heading == other.heading and
                self.mass == other.mass and
                self.radius == other.radius and
                self.id == other.id and
                self.team == other.team and
                self.lives_remaining == other.lives_remaining and
                self.bullets_remaining == other.bullets_remaining and
                self.mines_remaining == other.mines_remaining and
                #self.can_fire == other.can_fire and
                self.fire_rate == other.fire_rate and
                #self.can_deploy_mine == other.can_deploy_mine and
                self.mine_deploy_rate == other.mine_deploy_rate and
                self.thrust_range == other.thrust_range and
                self.turn_rate_range == other.turn_rate_range and
                self.max_speed == other.max_speed and
                self.drag == other.drag)


class Mine:
    __slots__ = ('position', 'mass', 'fuse_time', 'remaining_time')

    def __init__(self, position: tuple[float, float] = (0.0, 0.0), mass: float = 0.0, fuse_time: float = 0.0, remaining_time: float = 0.0) -> None:
        self.position = position
        self.mass = mass
        self.fuse_time = fuse_time
        self.remaining_time = remaining_time

    def __str__(self) -> str:
        return f'Mine(position={self.position}, mass={self.mass}, fuse_time={self.fuse_time}, remaining_time={self.remaining_time})'

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'Mine':
        return Mine(
            position=self.position,
            mass=self.mass,
            fuse_time=self.fuse_time,
            remaining_time=self.remaining_time
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mine):
            return NotImplemented
        return (self.position == other.position and
                self.mass == other.mass and
                self.fuse_time == other.fuse_time and
                self.remaining_time == other.remaining_time)


class Bullet:
    __slots__ = ('position', 'velocity', 'heading', 'mass', 'tail_delta')

    def __init__(self, position: tuple[float, float] = (0.0, 0.0), velocity: tuple[float, float] = (0.0, 0.0), heading: float = 0.0, mass: float = BULLET_MASS, tail_delta: Optional[tuple[float, float]] = None) -> None:
        self.position = position
        self.velocity = velocity
        self.heading = heading
        self.mass = mass
        if tail_delta is not None:
            self.tail_delta = tail_delta
        else:
            self.tail_delta = (-BULLET_LENGTH*cos(radians(heading)), -BULLET_LENGTH*sin(radians(heading)))

    def __str__(self) -> str:
        return f'Bullet(position={self.position}, velocity={self.velocity}, heading={self.heading}, mass={self.mass}, tail_delta={self.tail_delta})'

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'Bullet':
        return Bullet(
            position=self.position,
            velocity=self.velocity,
            heading=self.heading,
            mass=self.mass,
            tail_delta=self.tail_delta
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bullet):
            return NotImplemented
        return (self.position == other.position and
                self.velocity == other.velocity and
                self.heading == other.heading and
                self.mass == other.mass and
                self.tail_delta == other.tail_delta)


class GameState:
    __slots__ = ('asteroids', 'ships', 'bullets', 'mines', 'map_size', 'time', 'delta_time', 'sim_frame', 'time_limit')

    def __init__(self, asteroids: list[Asteroid], ships: list[Ship], bullets: list[Bullet], mines: list[Mine], map_size: tuple[float, float] = (0.0, 0.0), time: float = 0.0, delta_time: float = 0.0, sim_frame: i64 = 0, time_limit: float = 0.0) -> None:
        self.asteroids = asteroids
        self.ships = ships
        self.bullets = bullets
        self.mines = mines
        self.map_size = map_size
        self.time = time
        self.delta_time = delta_time
        self.sim_frame = sim_frame
        self.time_limit = time_limit

    def __str__(self) -> str:
        return f'GameState(asteroids={self.asteroids}, ships={self.ships}, bullets={self.bullets}, mines={self.mines}, map_size={self.map_size}, time={self.time}, delta_time={self.delta_time}, sim_frame={self.sim_frame}, time_limit={self.time_limit})'

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'GameState':
        return GameState(
            asteroids=[asteroid.copy() for asteroid in self.asteroids], 
            ships=[ship.copy() for ship in self.ships], 
            bullets=[bullet.copy() for bullet in self.bullets], 
            mines=[mine.copy() for mine in self.mines], 
            map_size=self.map_size, 
            time=self.time, 
            delta_time=self.delta_time, 
            sim_frame=self.sim_frame,
            time_limit=self.time_limit
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return NotImplemented
        
        # Compare asteroids
        if len(self.asteroids) != len(other.asteroids):
            print("Asteroids lists are different lengths!")
            return False
        for i, (ast_a, ast_b) in enumerate(zip(self.asteroids, other.asteroids)):
            if ast_a != ast_b:
                print(f"Asteroids don't match at index {i}: {ast_a} vs {ast_b}")
                return False
        
        # Compare bullets
        if len(self.bullets) != len(other.bullets):
            print("Bullet lists are different lengths!")
            return False
        for i, (bul_a, bul_b) in enumerate(zip(self.bullets, other.bullets)):
            if bul_a != bul_b:
                print(f"Bullets don't match at index {i}: {bul_a} vs {bul_b}")
                return False
        
        # Compare mines
        if len(self.mines) != len(other.mines):
            print("Mine lists are different lengths!")
            return False
        for i, (mine_a, mine_b) in enumerate(zip(self.mines, other.mines)):
            if mine_a != mine_b:
                print(f"Mines don't match at index {i}: {mine_a} vs {mine_b}")
                return False
        '''
        # Compare ships
        if len(self.ships) != len(other.ships):
            print("Ships lists are different lengths!")
            return False
        for i, (ship_a, ship_b) in enumerate(zip(self.ships, other.ships)):
            if ship_a != ship_b:
                print(f"Ships don't match at index {i}: {ship_a} vs {ship_b}")
                return False
        '''
        # Trivial properties like timesteps are not compared, assuming they don't affect game state equality
        return True


class Target:
    __slots__ = ('asteroid', 'feasible', 'shooting_angle_error_deg', 'aiming_timesteps_required', 'interception_time_s', 'intercept_x', 'intercept_y', 'asteroid_dist_during_interception', 'imminent_collision_time_s', 'asteroid_will_get_hit_by_my_mine', 'asteroid_will_get_hit_by_their_mine')

    def __init__(self, asteroid: Asteroid, feasible: bool = False, shooting_angle_error_deg: float = 0.0, aiming_timesteps_required: i64 = 0, interception_time_s: float = 0.0, intercept_x: float = 0.0, intercept_y: float = 0.0, asteroid_dist_during_interception: float = 0.0, imminent_collision_time_s: float = 0.0, asteroid_will_get_hit_by_my_mine: bool = False, asteroid_will_get_hit_by_their_mine: bool = False) -> None:
        self.asteroid = asteroid
        self.feasible = feasible
        self.shooting_angle_error_deg = shooting_angle_error_deg
        self.aiming_timesteps_required = aiming_timesteps_required
        self.interception_time_s = interception_time_s
        self.intercept_x = intercept_x
        self.intercept_y = intercept_y
        self.asteroid_dist_during_interception = asteroid_dist_during_interception
        self.imminent_collision_time_s = imminent_collision_time_s
        self.asteroid_will_get_hit_by_my_mine = asteroid_will_get_hit_by_my_mine
        self.asteroid_will_get_hit_by_their_mine = asteroid_will_get_hit_by_their_mine

    def __str__(self) -> str:
        return f'Target(asteroid={self.asteroid}, feasible={self.feasible}, shooting_angle_error_deg={self.shooting_angle_error_deg}, aiming_timesteps_required={self.aiming_timesteps_required}, interception_time_s={self.interception_time_s}, intercept_x={self.intercept_x}, intercept_y={self.intercept_y}, asteroid_dist_during_interception={self.asteroid_dist_during_interception}, imminent_collision_time_s={self.imminent_collision_time_s}, asteroid_will_get_hit_by_my_mine={self.asteroid_will_get_hit_by_my_mine}, asteroid_will_get_hit_by_their_mine={self.asteroid_will_get_hit_by_their_mine})'

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'Target':
        return Target(
            asteroid=self.asteroid.copy(), 
            feasible=self.feasible, 
            shooting_angle_error_deg=self.shooting_angle_error_deg, 
            aiming_timesteps_required=self.aiming_timesteps_required, 
            interception_time_s=self.interception_time_s, 
            intercept_x=self.intercept_x, 
            intercept_y=self.intercept_y, 
            asteroid_dist_during_interception=self.asteroid_dist_during_interception, 
            imminent_collision_time_s=self.imminent_collision_time_s, 
            asteroid_will_get_hit_by_my_mine=self.asteroid_will_get_hit_by_my_mine,
            asteroid_will_get_hit_by_their_mine=self.asteroid_will_get_hit_by_their_mine
        )


class Action:
    __slots__ = ('thrust', 'turn_rate', 'fire', 'drop_mine', 'timestep')

    def __init__(self, thrust: float = 0.0, turn_rate: float = 0.0, fire: bool = False, drop_mine: bool = False, timestep: i64 = 0) -> None:
        self.thrust = thrust
        self.turn_rate = turn_rate
        self.fire = fire
        self.drop_mine = drop_mine
        self.timestep = timestep

    def __str__(self) -> str:
        return f'Action(thrust={self.thrust}, turn_rate={self.turn_rate}, fire={self.fire}, drop_mine={self.drop_mine}, timestep={self.timestep})'

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'Action':
        return Action(
            thrust=self.thrust, 
            turn_rate=self.turn_rate, 
            fire=self.fire, 
            drop_mine=self.drop_mine, 
            timestep=self.timestep
        )


class SimState:
    __slots__ = ('timestep', 'ship_state', 'game_state', 'asteroids_pending_death', 'forecasted_asteroid_splits')

    def __init__(self, timestep: i64, ship_state: Ship, game_state: Optional[GameState] = None, asteroids_pending_death: Optional[dict[i64, list[Asteroid]]] = None, forecasted_asteroid_splits: Optional[list[Asteroid]] = None) -> None:
        self.timestep = timestep
        self.ship_state = ship_state
        self.game_state = game_state
        self.asteroids_pending_death = asteroids_pending_death
        self.forecasted_asteroid_splits = forecasted_asteroid_splits

    def __str__(self) -> str:
        return f'SimState(timestep={self.timestep}, ship_state={self.ship_state}, game_state={self.game_state}, asteroids_pending_death={self.asteroids_pending_death}, forecasted_asteroid_splits={self.forecasted_asteroid_splits})'

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> 'SimState':
        return SimState(
            timestep=self.timestep, 
            ship_state=self.ship_state.copy(), 
            game_state=self.game_state.copy() if self.game_state else None, 
            asteroids_pending_death=self.asteroids_pending_death.copy() if self.asteroids_pending_death else None, 
            forecasted_asteroid_splits=[a.copy() for a in self.forecasted_asteroid_splits] if self.forecasted_asteroid_splits else []
        )



class AsteroidDict(TypedDict):#, total=False):
    position: tuple[float, float]
    velocity: tuple[float, float]
    size: i64
    mass: float
    radius: float
    #timesteps_until_appearance: i64


class ShipDict(TypedDict, total=False):
    is_respawning: bool
    position: tuple[float, float]
    velocity: tuple[float, float]
    speed: float
    heading: float
    mass: float
    radius: float
    id: i64
    team: str
    lives_remaining: i64
    bullets_remaining: i64
    mines_remaining: i64
    can_fire: bool
    fire_rate: float
    thrust_range: tuple[float, float]
    turn_rate_range: tuple[float, float]
    max_speed: float
    drag: float


class MineDict(TypedDict):
    position: tuple[float, float]
    mass: float
    fuse_time: float
    remaining_time: float


class BulletDict(TypedDict):#, total=False):
    # Since I add in the tail delta myself, it may or may not exist.
    position: tuple[float, float]
    velocity: tuple[float, float]
    heading: float
    mass: float
    #tail_delta: tuple[float, float]


class GameStateDict(TypedDict):
    asteroids: list[AsteroidDict]
    ships: list[ShipDict]
    bullets: list[BulletDict]
    mines: list[MineDict]
    map_size: tuple[float, float]
    time: float
    delta_time: float
    sim_frame: i64
    time_limit: float


class BasePlanningGameState(TypedDict):
    timestep: i64
    respawning: bool
    ship_state: Ship
    game_state: GameState
    ship_respawn_timer: float
    asteroids_pending_death: dict[i64, list[Asteroid]]
    forecasted_asteroid_splits: list[Asteroid]
    last_timestep_fired: i64
    last_timestep_mined: i64
    mine_positions_placed: set[tuple[float, float]]
    fire_next_timestep_flag: bool


class CompletedSimulation(TypedDict, total=False):
    sim: 'Matrix'
    fitness: float
    fitness_breakdown: tuple[float, float, float, float, float, float, float, float, float]
    action_type: str
    state_type: str
    maneuver_tuple: Optional[tuple[float, float, float, i64, float]]


def create_game_state_from_dict(game_state_dict: GameStateDict) -> GameState:
    asteroids = [Asteroid(**asteroid) for asteroid in game_state_dict['asteroids']]
    ships = [Ship(**ship) for ship in game_state_dict['ships']]
    bullets = [Bullet(**bullet) for bullet in game_state_dict['bullets']]
    mines = [Mine(**mine) for mine in game_state_dict['mines']]
    map_size = game_state_dict['map_size']
    time = game_state_dict['time']
    delta_time = game_state_dict['delta_time']
    sim_frame = game_state_dict['sim_frame']
    time_limit = game_state_dict['time_limit']
    return GameState(asteroids=asteroids, ships=ships, bullets=bullets, mines=mines, map_size=map_size, time=time, delta_time=delta_time, sim_frame=sim_frame, time_limit=time_limit)


def create_ship_from_dict(ship_state_dict: ShipDict) -> Ship:
    #print(f"Creating ship from: {ship_state_dict} and the bullets remaining is {ship_state_dict.get('bullets_remaining', 0)}")
    return Ship(**ship_state_dict)


def degrees(x: float) -> float:
    '''Convert radians to degrees.'''
    return x*RAD_TO_DEG


def radians(x: float) -> float:
    '''Convert degrees to radians.'''
    return x*DEG_TO_RAD


def sign(x: float) -> float:
    if x >= 0.0:
        return 1.0
    else:
        return -1.0


def dist(point_1: tuple[float, float], point_2: tuple[float, float]) -> float:
    '''Calculate the Euclidean distance between two points in 2D space.'''
    dx = point_1[0] - point_2[0]
    dy = point_1[1] - point_2[1]
    return sqrt(dx*dx + dy*dy)


def is_close(x: float, y: float) -> bool:
    '''Check if two numbers are close based on absolute tolerance.'''
    return abs(x - y) <= EPS


def is_close_to_zero(x: float) -> bool:
    '''Check if a number is close to zero.'''
    return abs(x) <= EPS


def super_fast_acos(x: float) -> float:
    return (-0.69813170079773212*x*x - 0.87266462599716477)*x + 1.5707963267948966


def fast_acos(x: float) -> float:
    negate = float(x < 0)
    x = abs(x)
    ret = (((-0.0187293*x + 0.0742610)*x - 0.2121144)*x + 1.5707288)*sqrt(1.0 - x)
    return negate*pi + ret - 2.0*negate*ret


def super_fast_asin(x: float) -> float:
    # 29 ns per call when compiled, compared to the math.asin which takes 160 ns
    x_square = x*x
    return x*(0.9678828 + x_square*(0.8698691 - x_square*(2.166373 - x_square*1.848968)))


def fast_asin(x: float) -> float:
    # 48 ns per call when compiled
    negate = float(x < 0)
    x = abs(x)
    ret = (((-0.0187293*x + 0.0742610)*x - 0.2121144)*x + 1.5707288)
    ret = 0.5*pi - sqrt(1.0 - x)*ret
    return ret - 2.0*negate*ret


def super_fast_atan2(y: float, x: float) -> float:
    # Coefficients for the atan approximation
    
    # Handle edge cases for 0 inputs
    if x == 0.0:
        if y == 0.0:
            return 0.0  # atan2(0, 0) is undefined; returning 0 for simplicity
        else:
            return 0.5*pi if y > 0.0 else -0.5*pi
    if y == 0.0:
        if x > 0.0:
            return 0.0
        else:
            return pi
    
    # Determine the input for the approximation based on |y| and |x|
    if abs(x) < abs(y):
        swap = True
        atan_input = x/y
    else:
        swap = False
        atan_input = y/x
    
    # Calculate the atan approximation
    x_sq = atan_input*atan_input
    atan_result = atan_input*(0.995354 - x_sq*(0.288679 - 0.079331*x_sq))
    
    # Adjust the result based on the original input quadrant
    if swap:
        if atan_input >= 0.0:
            atan_result = 0.5*pi - atan_result
        else:
            atan_result = -0.5*pi - atan_result
    
    # Adjust for the correct quadrant
    if x < 0.0:
        if y >= 0.0:
            atan_result += pi
        else:
            atan_result += -pi
    
    return atan_result


def fast_atan2(y: float, x: float) -> float:
    # Coefficients for the atan approximation
    
    # Handle edge cases for 0 inputs
    if x == 0.0:
        if y == 0.0:
            return 0.0  # atan2(0, 0) is undefined; returning 0 for simplicity
        else:
            return 0.5*pi if y > 0.0 else -0.5*pi
    if y == 0.0:
        if x > 0.0:
            return 0.0
        else:
            return pi
    
    # Determine the input for the approximation based on |y| and |x|
    if abs(x) < abs(y):
        swap = True
        atan_input = x/y
    else:
        swap = False
        atan_input = y/x
    
    # Calculate the atan approximation
    x_sq = atan_input*atan_input
    atan_result = atan_input*(0.99997726 - x_sq*(0.33262347 - x_sq*(0.19354346 - x_sq*(0.11643287 - x_sq*(0.05265332 - x_sq*0.01172120)))))
    
    # Adjust the result based on the original input quadrant
    if swap:
        if atan_input >= 0.0:
            atan_result = 0.5*pi - atan_result
        else:
            atan_result = -0.5*pi - atan_result
    
    # Adjust for the correct quadrant
    if x < 0.0:
        if y >= 0.0:
            atan_result += pi
        else:
            atan_result += -pi
    
    return atan_result


def heading_diff_within_threshold(a_vec_theta_rad: float, b_vec_x: float, b_vec_y: float, cos_threshold: float) -> bool:
    '''a and b are direction vectors. This checks whether the absolute heading difference between them are within a threshold angle.'''
    # This function would work the same way as if we took the ship's heading, and found the absolute wrapped difference between that and the direction vector of the asteroid, and then compared against the threshold
    # But this method is faster as it avoids the atan2 calculation, and instead does a sin and cos and does the threshold comparison with the cos value of the angle, instead of the angle itself
    # The old method is shown below for reference:
    # theta = degrees(atan2(b_vec_y, b_vec_x))
    # threshold_angle = acos(cos_threshold)
    # return abs(angle_difference_deg(theta, degrees(a_vec_theta_rad))) <= degrees(threshold_angle)
    
    #a_vec_norm = 1.0
    a_vec_x = cos(a_vec_theta_rad)
    a_vec_y = sin(a_vec_theta_rad)
    dot_product = a_vec_x*b_vec_x + a_vec_y*b_vec_y
    magnitude = sqrt(b_vec_x*b_vec_x + b_vec_y*b_vec_y)
    if magnitude != 0:
        cos_theta = dot_product/magnitude # a_vec_norm is 1.0 since it's a unit vector, so we don't need to multiply by that
        return cos_theta >= cos_threshold
    else:
        return True


@lru_cache
def get_min_respawn_per_timestep_search_iterations(lives: i64, average_fitness: float) -> i64:
    assert 0.0 <= average_fitness < 1.0
    lives_lookup_index = min(3, lives)
    fitness_lookup_index = floor(average_fitness*10.0) # Integer from 0 to 9
    return MIN_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS_LUT[fitness_lookup_index][lives_lookup_index - 1]


@lru_cache
def get_min_respawn_per_period_search_iterations(lives: i64, average_fitness: float) -> i64:
    assert 0.0 <= average_fitness < 1.0
    lives_lookup_index = min(3, lives)
    fitness_lookup_index = floor(average_fitness*10.0) # Integer from 0 to 9
    return MIN_RESPAWN_PER_PERIOD_SEARCH_ITERATIONS_LUT[fitness_lookup_index][lives_lookup_index - 1]


@lru_cache
def get_min_maneuver_per_timestep_search_iterations(lives: i64, average_fitness: float) -> i64:
    assert 0.0 <= average_fitness < 1.0
    lives_lookup_index = min(3, lives)
    fitness_lookup_index = floor(average_fitness*10.0) # Integer from 0 to 9
    return MIN_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS_LUT[fitness_lookup_index][lives_lookup_index - 1]


@lru_cache
def get_min_maneuver_per_period_search_iterations(lives: i64, average_fitness: float) -> i64:
    assert 0.0 <= average_fitness < 1.0
    lives_lookup_index = min(3, lives)
    fitness_lookup_index = floor(average_fitness*10.0) # Integer from 0 to 9
    return MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_LUT[fitness_lookup_index][lives_lookup_index - 1]


@lru_cache
def get_min_maneuver_per_period_search_iterations_if_will_die(lives: i64, average_fitness: float) -> i64:
    # If we plan to die, we might as well search a bunch more iterations to try and avoid the death.
    # Because if we die, we need to do an expensive respawn maneuver search anyway, so it's more optimal to try and avoid that.
    assert 0.0 <= average_fitness < 1.0
    lives_lookup_index = min(3, lives)
    fitness_lookup_index = floor(average_fitness*10.0) # Integer from 0 to 9
    return MIN_MANEUVER_PER_PERIOD_SEARCH_ITERATIONS_IF_WILL_DIE_LUT[fitness_lookup_index][lives_lookup_index - 1]


@lru_cache()
def set_up_mine_fis() -> control.ControlSystemSimulation:
    # set up mine FIS
    mines_left = control.Antecedent(arange(0, 4, 1), 'mines_left')
    lives_left = control.Antecedent(arange(1, 4, 1), 'lives_left')
    asteroids_hit = control.Antecedent(arange(0, 51, 1), 'asteroids_hit')
    drop_mine = control.Consequent(arange(0, 11, 1), 'drop_mine')

    # Defining the membership functions
    mines_left['few'] = trimf(mines_left.universe, [1, 1, 3])
    mines_left['many'] = trimf(mines_left.universe, [1.5, 3, 3])
    lives_left['few'] = trimf(lives_left.universe, [1, 1, 3])
    lives_left['many'] = trimf(lives_left.universe, [1.5, 3, 3])
    asteroids_hit['few'] = trimf(asteroids_hit.universe, [0, 0, ASTEROIDS_HIT_OKAY_CENTER])
    asteroids_hit['okay'] = trimf(asteroids_hit.universe, [0, ASTEROIDS_HIT_OKAY_CENTER, ASTEROIDS_HIT_VERY_GOOD])
    asteroids_hit['many'] = trimf(asteroids_hit.universe, [ASTEROIDS_HIT_OKAY_CENTER, ASTEROIDS_HIT_VERY_GOOD, ASTEROIDS_HIT_VERY_GOOD])

    drop_mine['no'] = trimf(drop_mine.universe, [0, 0, 5])
    drop_mine['yes'] = trimf(drop_mine.universe, [5, 10, 10])

    rules = (
        control.Rule(mines_left['few'] & lives_left['few'] & asteroids_hit['few'], drop_mine['no']),
        control.Rule(mines_left['few'] & lives_left['few'] & asteroids_hit['okay'], drop_mine['no']),
        control.Rule(mines_left['few'] & lives_left['few'] & asteroids_hit['many'], drop_mine['yes']),
        control.Rule(mines_left['few'] & lives_left['many'] & asteroids_hit['few'], drop_mine['no']),
        control.Rule(mines_left['few'] & lives_left['many'] & asteroids_hit['okay'], drop_mine['yes']),
        control.Rule(mines_left['few'] & lives_left['many'] & asteroids_hit['many'], drop_mine['yes']),
        control.Rule(mines_left['many'] & lives_left['few'] & asteroids_hit['few'], drop_mine['no']),
        control.Rule(mines_left['many'] & lives_left['few'] & asteroids_hit['okay'], drop_mine['no']),
        control.Rule(mines_left['many'] & lives_left['few'] & asteroids_hit['many'], drop_mine['yes']),
        control.Rule(mines_left['many'] & lives_left['many'] & asteroids_hit['few'], drop_mine['yes']),
        control.Rule(mines_left['many'] & lives_left['many'] & asteroids_hit['okay'], drop_mine['yes']),
        control.Rule(mines_left['many'] & lives_left['many'] & asteroids_hit['many'], drop_mine['yes']),
    )

    mine_dropping_control = control.ControlSystem(rules)
    mine_dropping_fis = control.ControlSystemSimulation(mine_dropping_control)
    return mine_dropping_fis


@lru_cache()
def generate_mine_fis_lookup_table() -> dict[tuple[i64, i64, i64], bool]:
    mine_dropping_fis = set_up_mine_fis()
    lookup_table = {}

    for mines_left in range(1, 4):  # 0 to 3
        for lives_left in range(1, 4):  # 1 to 3
            for asteroids_hit in range(1, ASTEROIDS_HIT_VERY_GOOD + 1):
                # Input setup
                mine_dropping_fis.input['mines_left'] = mines_left
                mine_dropping_fis.input['lives_left'] = lives_left
                mine_dropping_fis.input['asteroids_hit'] = asteroids_hit
                mine_dropping_fis.compute()
                # Decision based on output
                drop_decision = cast(float, mine_dropping_fis.output['drop_mine'])
                should_drop_mine: bool = drop_decision > 5
                lookup_table[(mines_left, lives_left, asteroids_hit)] = should_drop_mine
    return lookup_table


@lru_cache(maxsize=None)
def mine_fis(num_mines_left: i64, num_lives_left: i64, num_asteroids_hit: i64) -> bool:
    # debug_print("Mine fis inputs", num_mines_left, num_lives_left, num_asteroids_hit)
    if num_mines_left == 0 or num_asteroids_hit < 8:
        return False
    num_mines_left = min(num_mines_left, 3)
    num_lives_left = min(num_lives_left, 3)
    num_asteroids_hit = min(num_asteroids_hit, ASTEROIDS_HIT_VERY_GOOD)
    # debug_print(f"Mine fis: Mines left: {num_mines_left}, lives left: {num_lives_left}, asteroids hit: {num_asteroids_hit}")
    mine_dropping_fis = set_up_mine_fis()
    mine_dropping_fis.input['mines_left'] = num_mines_left
    mine_dropping_fis.input['lives_left'] = num_lives_left
    mine_dropping_fis.input['asteroids_hit'] = num_asteroids_hit
    # Compute the output
    mine_dropping_fis.compute()
    drop_decision = cast(float, mine_dropping_fis.output['drop_mine'])
    # Interpreting the output
    should_drop_mine: bool = drop_decision > 5  # True for drop, False for don't drop
    return should_drop_mine


@lru_cache()
def mine_fis_lookup_table(num_mines_left: i64, num_lives_left: i64, num_asteroids_hit: i64) -> bool:
    # The lru cache is simpler and just as fast as creating my own lookup table! Don't use this!
    if num_mines_left == 0 or num_asteroids_hit < 8:
        return False
    lookup_table = generate_mine_fis_lookup_table()
    num_mines_left = min(num_mines_left, 3)
    num_lives_left = min(num_lives_left, 3)
    num_asteroids_hit = min(num_asteroids_hit, ASTEROIDS_HIT_VERY_GOOD)

    return lookup_table[(num_mines_left, num_lives_left, num_asteroids_hit)]


def check_mine_opportunity(ship_state: Ship, game_state: GameState, other_ships: list[Ship]) -> bool:
    if len(game_state.mines) > 1:
        return False
    mine_ast_count = count_asteroids_in_mine_blast_radius(game_state, ship_state.position[0], ship_state.position[1], round(MINE_FUSE_TIME*FPS))
    for other_ship in other_ships:
        #if check_collision(ship_state.position[0], ship_state.position[1], MINE_BLAST_RADIUS - MINE_OTHER_SHIP_RADIUS_FUDGE, other_ship.position[0], other_ship.position[1], other_ship.radius):
        delta_x = ship_state.position[0] - other_ship.position[0]
        delta_y = ship_state.position[1] - other_ship.position[1]
        separation = (MINE_BLAST_RADIUS - MINE_OTHER_SHIP_RADIUS_FUDGE) + other_ship.radius
        if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
            # print(f"Potentially bombing other ship. Giving reward of 10 asts!")
            mine_ast_count += MINE_OTHER_SHIP_ASTEROID_COUNT_EQUIVALENT
    if ship_state.bullets_remaining == 0:
        # If ship is out of bullets, fudge the numbers to make the mine fis more likely to activate
        mine_ast_count *= 10
        if len(game_state.mines) > 0:
            # We want to conserve mines more if we only have mines left and not bullets, and laying multiple mines at once is risky because the first one may blow asteroids away from the second, so the second one would be a waste
            return False
    # debug_print(f"Mine count inside: {mine_ast_count} compared to average density amount inside: {average_asteroids_inside_blast_radius}")
    return mine_fis(ship_state.mines_remaining, ship_state.lives_remaining, mine_ast_count)


@lru_cache()
def setup_heuristic_maneuver_fis() -> control.ControlSystemSimulation:
    K = 0.8*FPS
    # Antecedents (Inputs)
    imminent_asteroid_speed = control.Antecedent(arange(0, 301, 1), 'imminent_asteroid_speed')
    imminent_asteroid_relative_heading = control.Antecedent(arange(0, 361, 1), 'imminent_asteroid_relative_heading')
    largest_gap_relative_heading = control.Antecedent(arange(0, 361, 1), 'largest_gap_relative_heading')
    nearby_asteroid_average_speed = control.Antecedent(arange(0, 301, 1), 'nearby_asteroid_average_speed')
    nearby_asteroid_count = control.Antecedent(arange(0, 16, 1), 'nearby_asteroid_count')

    # Consequents (Outputs)
    ship_accel_turn_rate = control.Consequent(arange(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE + 1, 1), 'ship_accel_turn_rate')
    ship_cruise_speed = control.Consequent(arange(-SHIP_MAX_SPEED, SHIP_MAX_SPEED + 1, 1), 'ship_cruise_speed')
    ship_cruise_turn_rate = control.Consequent(arange(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE + 1, 1), 'ship_cruise_turn_rate')
    ship_cruise_timesteps = control.Consequent(arange(0, K + 1, 1), 'ship_cruise_timesteps')
    ship_thrust_direction = control.Consequent(arange(-1, 2, 1), 'ship_thrust_direction')

    # Membership Functions for Antecedents
    imminent_asteroid_speed.automf(names=['slow', 'medium', 'fast'])
    # imminent_asteroid_relative_heading.automf(names=['F', 'FL', 'L', 'BL', 'B', 'BR', 'R', 'FR'])
    # largest_gap_relative_heading.automf(names=['F', 'FL', 'L', 'BL', 'B', 'BR', 'R', 'FR'])
    nearby_asteroid_average_speed.automf(names=['slow', 'medium', 'fast'])
    nearby_asteroid_count.automf(names=['few', 'medium', 'many'])

    # Define manual membership functions with 90 degrees width, dominant in 45 degrees
    # Forward (F) - Bimodal covering 0 to 45 and 315 to 360
    imminent_asteroid_relative_heading['F'] = fmax(
        trimf(imminent_asteroid_relative_heading.universe, [0, 0, 45]),  # Extends to 0 to 45
        trimf(imminent_asteroid_relative_heading.universe, [315, 360, 360])  # Wraps from 315 to 360
    )

    # Forward Left (FL)
    imminent_asteroid_relative_heading['FL'] = trimf(imminent_asteroid_relative_heading.universe, [0, 45, 90])

    # Left (L)
    imminent_asteroid_relative_heading['L'] = trimf(imminent_asteroid_relative_heading.universe, [45, 90, 135])

    # Back Left (BL)
    imminent_asteroid_relative_heading['BL'] = trimf(imminent_asteroid_relative_heading.universe, [90, 135, 180])

    # Back (B)
    imminent_asteroid_relative_heading['B'] = trimf(imminent_asteroid_relative_heading.universe, [135, 180, 225])

    # Back Right (BR)
    imminent_asteroid_relative_heading['BR'] = trimf(imminent_asteroid_relative_heading.universe, [180, 225, 270])

    # Right (R)
    imminent_asteroid_relative_heading['R'] = trimf(imminent_asteroid_relative_heading.universe, [225, 270, 315])

    # Forward Right (FR)
    imminent_asteroid_relative_heading['FR'] = trimf(imminent_asteroid_relative_heading.universe, [270, 315, 360])

    # Define manual membership functions with 90 degrees width, dominant in 45 degrees
    # Forward (F) - Bimodal covering 0 to 45 and 315 to 360
    largest_gap_relative_heading['F'] = fmax(
        trimf(largest_gap_relative_heading.universe, [0, 0, 45]),  # Extends to 0 to 45
        trimf(largest_gap_relative_heading.universe, [315, 360, 360])  # Wraps from 315 to 360
    )

    # Forward Left (FL)
    largest_gap_relative_heading['FL'] = trimf(largest_gap_relative_heading.universe, [0, 45, 90])

    # Left (L)
    largest_gap_relative_heading['L'] = trimf(largest_gap_relative_heading.universe, [45, 90, 135])

    # Back Left (BL)
    largest_gap_relative_heading['BL'] = trimf(largest_gap_relative_heading.universe, [90, 135, 180])

    # Back (B)
    largest_gap_relative_heading['B'] = trimf(largest_gap_relative_heading.universe, [135, 180, 225])

    # Back Right (BR)
    largest_gap_relative_heading['BR'] = trimf(largest_gap_relative_heading.universe, [180, 225, 270])

    # Right (R)
    largest_gap_relative_heading['R'] = trimf(largest_gap_relative_heading.universe, [225, 270, 315])

    # Forward Right (FR)
    largest_gap_relative_heading['FR'] = trimf(largest_gap_relative_heading.universe, [270, 315, 360])
    # largest_gap_relative_heading['F'].view()

    # Membership Functions for Consequents

    # Acceleration Turn Rate
    ship_accel_turn_rate['R'] = trimf(ship_accel_turn_rate.universe, [-SHIP_MAX_TURN_RATE, -SHIP_MAX_TURN_RATE/2, 0])
    ship_accel_turn_rate['M'] = trimf(ship_accel_turn_rate.universe, [-SHIP_MAX_TURN_RATE/4, 0, SHIP_MAX_TURN_RATE/4])
    ship_accel_turn_rate['L'] = trimf(ship_accel_turn_rate.universe, [0, SHIP_MAX_TURN_RATE/2, SHIP_MAX_TURN_RATE])
    # ship_accel_turn_rate.view()
    # Cruise Speed
    ship_cruise_speed['slow'] = trimf(ship_cruise_speed.universe, [0, 0, SHIP_MAX_SPEED/2])
    ship_cruise_speed['medium'] = trimf(ship_cruise_speed.universe, [0, SHIP_MAX_SPEED/2, SHIP_MAX_SPEED])
    ship_cruise_speed['fast'] = trimf(ship_cruise_speed.universe, [SHIP_MAX_SPEED/2, SHIP_MAX_SPEED, SHIP_MAX_SPEED])
    # ship_cruise_speed.view()
    # Cruise Turn Rate
    ship_cruise_turn_rate['R'] = trimf(ship_cruise_turn_rate.universe, [-SHIP_MAX_TURN_RATE, -SHIP_MAX_TURN_RATE/2, 0])
    ship_cruise_turn_rate['M'] = trimf(ship_cruise_turn_rate.universe, [-SHIP_MAX_TURN_RATE/4, 0, SHIP_MAX_TURN_RATE/4])
    ship_cruise_turn_rate['L'] = trimf(ship_cruise_turn_rate.universe, [0, SHIP_MAX_TURN_RATE/2, SHIP_MAX_TURN_RATE])
    # ship_cruise_turn_rate.view()
    # Cruise Time Steps
    ship_cruise_timesteps['short'] = trimf(ship_cruise_timesteps.universe, [0, K/4, K/2])
    ship_cruise_timesteps['medium'] = trimf(ship_cruise_timesteps.universe, [K/4, K/2, 3*K/4])
    ship_cruise_timesteps['long'] = trimf(ship_cruise_timesteps.universe, [K/2, 3*K/4, K])
    # ship_cruise_timesteps.view()

    ship_thrust_direction['F'] = trimf(ship_thrust_direction.universe, [0, 1, 1])
    ship_thrust_direction['B'] = trimf(ship_thrust_direction.universe, [-1, -1, 0])

    rules = (
        # The faster the imminent asteroid approaches, the quicker we want to get out of there
        control.Rule(imminent_asteroid_speed['slow'], ship_cruise_speed['slow']),
        control.Rule(imminent_asteroid_speed['medium'], ship_cruise_speed['medium']),
        control.Rule(imminent_asteroid_speed['fast'], ship_cruise_speed['fast']),
        # Adjust ship speed depending on the surrounding asteroids' speeds
        control.Rule(nearby_asteroid_average_speed['slow'], ship_cruise_speed['slow']),
        control.Rule(nearby_asteroid_average_speed['medium'], ship_cruise_speed['medium']),
        control.Rule(nearby_asteroid_average_speed['fast'], ship_cruise_speed['fast']),
        # Cruise length should depend on asteroid density a bit, since a lower density means we should be able to travel farther without hitting anything
        control.Rule(nearby_asteroid_count['few'], (ship_cruise_timesteps['long'], ship_cruise_speed['fast'])),
        control.Rule(nearby_asteroid_count['medium'], (ship_cruise_timesteps['medium'], ship_cruise_speed['medium'])),
        control.Rule(nearby_asteroid_count['many'], (ship_cruise_timesteps['short'], ship_cruise_speed['slow'])),
        # Whichever direction the imminent asteroid is from us, we want to avoid running directly at that asteroid
        control.Rule(imminent_asteroid_relative_heading['F'], (ship_thrust_direction['B'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['R'])),
        control.Rule(imminent_asteroid_relative_heading['B'], (ship_thrust_direction['F'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['R'])),
        control.Rule(imminent_asteroid_relative_heading['L'], (ship_thrust_direction['B'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['R'])),  # , ~(ship_thrust_direction['B'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['R'])),
        control.Rule(imminent_asteroid_relative_heading['R'], (ship_thrust_direction['B'], ship_accel_turn_rate['L'], ship_cruise_turn_rate['L'])),  # & ~(ship_thrust_direction['B'], ship_accel_turn_rate['L'], ship_cruise_turn_rate['L'])),
        control.Rule(imminent_asteroid_relative_heading['FL'], (ship_thrust_direction['B'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['R'])),  # & ~(ship_thrust_direction['F'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['L'])),
        control.Rule(imminent_asteroid_relative_heading['FR'], (ship_thrust_direction['B'], ship_accel_turn_rate['L'], ship_cruise_turn_rate['L'])),  # & ~(ship_thrust_direction['F'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['R'])),
        control.Rule(imminent_asteroid_relative_heading['BL'], (ship_thrust_direction['B'], ship_accel_turn_rate['L'], ship_cruise_turn_rate['L'])),  # & ~(ship_thrust_direction['B'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['R'])),
        control.Rule(imminent_asteroid_relative_heading['BR'], (ship_thrust_direction['B'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['R'])),  # & ~(ship_thrust_direction['B'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['L'])),
        # Wherever the largest gap to escape is, we want to go there
        control.Rule(largest_gap_relative_heading['F'], (ship_thrust_direction['F'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['M'])),
        control.Rule(largest_gap_relative_heading['B'], (ship_thrust_direction['B'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['M'])),
        control.Rule(largest_gap_relative_heading['L'], (ship_thrust_direction['F'], ship_accel_turn_rate['L'], ship_cruise_turn_rate['L'])),  # | (ship_thrust_direction['B'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['R'])),
        control.Rule(largest_gap_relative_heading['R'], (ship_thrust_direction['F'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['R'])),  # | (ship_thrust_direction['B'], ship_accel_turn_rate['L'], ship_cruise_turn_rate['L'])),
        control.Rule(largest_gap_relative_heading['FL'], (ship_thrust_direction['F'], ship_accel_turn_rate['L'], ship_cruise_turn_rate['M'])),  # | (ship_thrust_direction['F'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['L'])),
        control.Rule(largest_gap_relative_heading['FR'], (ship_thrust_direction['F'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['M'])),  # | (ship_thrust_direction['F'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['R'])),
        control.Rule(largest_gap_relative_heading['BL'], (ship_thrust_direction['B'], ship_accel_turn_rate['R'], ship_cruise_turn_rate['M'])),  # | (ship_thrust_direction['B'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['R'])),
        control.Rule(largest_gap_relative_heading['BR'], (ship_thrust_direction['B'], ship_accel_turn_rate['L'], ship_cruise_turn_rate['M'])),  # | (ship_thrust_direction['B'], ship_accel_turn_rate['M'], ship_cruise_turn_rate['L'])),
    )

    # Create the control system
    maneuver_control_system = control.ControlSystem(rules)
    maneuver_fis = control.ControlSystemSimulation(maneuver_control_system)
    return maneuver_fis


def maneuver_heuristic_fis(imminent_asteroid_speed: float, imminent_asteroid_relative_heading: float, largest_gap_relative_heading: float, nearby_asteroid_average_speed: float, nearby_asteroid_count: i64) -> tuple[float, float, float, float, float]:
    maneuver_fis = setup_heuristic_maneuver_fis()
    # print(f"imminent_asteroid_speed: {imminent_asteroid_speed}, imminent_asteroid_relative_heading: {imminent_asteroid_relative_heading}, largest_gap_relative_heading: {largest_gap_relative_heading}, nearby_asteroid_average_speed: {nearby_asteroid_average_speed}, nearby_asteroid_count: {nearby_asteroid_count}")
    maneuver_fis.input['imminent_asteroid_speed'] = imminent_asteroid_speed
    maneuver_fis.input['imminent_asteroid_relative_heading'] = imminent_asteroid_relative_heading
    maneuver_fis.input['largest_gap_relative_heading'] = largest_gap_relative_heading
    maneuver_fis.input['nearby_asteroid_average_speed'] = nearby_asteroid_average_speed
    maneuver_fis.input['nearby_asteroid_count'] = nearby_asteroid_count
    maneuver_fis.compute()

    acceleration_turn_rate = cast(float, maneuver_fis.output['ship_accel_turn_rate'])
    cruise_speed = cast(float, maneuver_fis.output['ship_cruise_speed'])
    cruise_turn_rate = cast(float, maneuver_fis.output['ship_cruise_turn_rate'])
    cruise_timesteps = cast(float, maneuver_fis.output['ship_cruise_timesteps'])
    thrust_direction = cast(float, maneuver_fis.output['ship_thrust_direction'])
    # debug_print(f"FIS Acceleration Turn Rate: {acceleration_turn_rate}, Cruise Speed: {cruise_speed}, Cruise Turn Rate: {cruise_turn_rate}, Cruise Timesteps: {cruise_timesteps}, Thrust Direction: {thrust_direction}")
    return acceleration_turn_rate, cruise_speed, cruise_turn_rate, cruise_timesteps, thrust_direction


def sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    """
    Compute the logistic sigmoid function with scaling and shift.

    Parameters:
    - x: The input value or array of values.
    - k: The scaling factor for steepness. Higher values make the curve steeper. Positive k makes an increasing function, while negative k make a decreasing function.
    - x0: The shift factor. Positive values shift the curve to the right, negative to the left.

    Returns:
    - The logistic sigmoid function value(s) for the given input.
    """
    return 1.0/(1.0 + exp(-k*(x - x0)))


def linear(x: float, point1: tuple[float, float], point2: tuple[float, float]) -> float:
    """
    Interpolate a linear function between two points. If the input is outside the range, then the output is clamped to the nearest extreme output.
    """
    x1, y1 = point1
    x2, y2 = point2
    if x <= x1:
        return y1
    elif x >= x2:
        return y2
    else:
        return y1 + (x - x1)*(y2 - y1)/(x2 - x1)


def weighted_average(numbers: Sequence[float | i64], weights: Optional[Sequence[float | i64]] = None) -> float:
    """
    Calculate the weighted average of a list of numbers.
    If no weights are provided, the regular average is calculated.
    Raises ValueError if weights are provided but do not match the length of numbers.

    Parameters:
    - numbers: list of numbers.
    - weights: Corresponding weights for each number.

    Returns:
    - The weighted (or regular) average of the numbers.

    Raises:
    - ValueError: If weights are provided but their length doesn't match the numbers list.
    """
    if weights is not None:
        if len(weights) != len(numbers):
            raise ValueError("Length of weights must match length of numbers.")
        # Calculate weighted average
        total_weighted = sum(float(number)*float(weight) for number, weight in zip(numbers, weights))
        total_weight = sum(weights)
        return float(total_weighted)/float(total_weight)
    else:
        # Calculate regular average if no weights are provided
        return sum(numbers)/len(numbers) if numbers else 0.0


def weighted_harmonic_mean(numbers: Sequence[float], weights: Optional[Sequence[float]] = None, offset: float = 0.0) -> float:
    """
    Calculate the weighted harmonic mean of a list of numbers.
    If no weights are provided, the regular harmonic mean is calculated.
    Raises ValueError if weights are provided but do not match the length of numbers.

    Parameters:
    - numbers: list of numbers.
    - weights: Corresponding weights for each number.

    Returns:
    - The weighted (or regular) harmonic mean of the numbers.

    Raises:
    - ValueError: If weights are provided but their length doesn't match the numbers list.
    """
    if numbers:
        if weights is not None:
            if len(weights) != len(numbers):
                raise ValueError("Length of weights must match length of numbers.")
            # Calculate weighted harmonic mean
            weight_sum = sum(weights)
            weighted_reciprocals_sum = sum(w/max(x + offset, TAD) for w, x in zip(weights, numbers))
            weighted_harmonic_mean = weight_sum/weighted_reciprocals_sum - offset
            return weighted_harmonic_mean  # Apparently without this redundant assignment of this variable, I get a compiler error in mypyc and it says I have an error on line -1 LOLL
        else:
            # Calculate regular harmonic mean if no weights are provided
            weight_sum = len(numbers)
            weighted_reciprocals_sum = sum(1.0/max(x + offset, TAD) for x in numbers)
            weighted_harmonic_mean = weight_sum/weighted_reciprocals_sum - offset
            return weighted_harmonic_mean
    else:
        return 0.0


def compare_asteroids(ast_a: Asteroid, ast_b: Asteroid) -> bool:
    for i in range(2):
        if ast_a.position[i] != ast_b.position[i]:
            return False
        if ast_a.velocity[i] != ast_b.velocity[i]:
            return False
    if ast_a.size != ast_b.size:
        return False
    if ast_a.mass != ast_b.mass:
        return False
    if ast_a.radius != ast_b.radius:
        return False
    return True


def compare_bullets(bul_a: Bullet, bul_b: Bullet) -> bool:
    for i in range(2):
        if bul_a.position[i] != bul_b.position[i]:
            return False
        if bul_a.velocity[i] != bul_b.velocity[i]:
            return False
    if bul_a.heading != bul_b.heading:
        return False
    if bul_a.mass != bul_b.mass:
        return False
    return True


def compare_mines(mine_a: Mine, mine_b: Mine) -> bool:
    for i in range(2):
        if not mine_a.position[i] == mine_b.position[i]:
            return False
    if not mine_a.mass == mine_b.mass:
        return False
    if not mine_a.fuse_time == mine_b.fuse_time:
        return False
    if not mine_a.remaining_time == mine_b.remaining_time:
        return False
    return True


def compare_gamestates(gamestate_a: GameState, gamestate_b: GameState) -> bool:
    # The game state consists of asteroids, ships, bullets, mines
    asteroids_a = gamestate_a.asteroids
    asteroids_b = gamestate_b.asteroids
    if len(asteroids_a) != len(asteroids_b):
        print("Asteroids lists are different lengths!")
        return False
    for i in range(len(asteroids_a)):
        if not compare_asteroids(asteroids_a[i], asteroids_b[i]):
            print(f"Asteroids don't match! {asteroids_a[i]} vs {asteroids_b[i]}")
            return False

    bullets_a = gamestate_a.bullets
    bullets_b = gamestate_b.bullets
    if len(bullets_a) != len(bullets_b):
        print("Bullet lists are different lengths!")
        return False
    for i in range(len(bullets_a)):
        if not compare_bullets(bullets_a[i], bullets_b[i]):
            print(f"Bullets don't match! {bullets_a[i]} vs {bullets_b[i]}")
            return False

    mines_a = gamestate_a.mines
    mines_b = gamestate_b.mines
    if len(mines_a) != len(mines_b):
        print("Mine lists are different lengths!")
        return False
    for i in range(len(mines_a)):
        if not compare_mines(mines_a[i], mines_b[i]):
            print("Mines don't match!")
            return False
    # No need to compare trivial stuff like timesteps that are in the game state
    return True


def compare_shipstates(ship_a: Ship, ship_b: Ship) -> bool:
    # Compare booleans and integers
    if ship_a.is_respawning != ship_b.is_respawning:
        return False
    if ship_a.id != ship_b.id:
        print("Ship ID's don't match!")
        return False
    if ship_a.team != ship_b.team:
        print("Ship teams don't match!")
        return False
    if ship_a.lives_remaining != ship_b.lives_remaining:
        print("Ship lives remaining don't match!")
        return False
    if ship_a.bullets_remaining != ship_b.bullets_remaining:
        print("Ship bullets remaining don't match!")
        print(ship_a.bullets_remaining)
        print(ship_b.bullets_remaining)
        return False
    if ship_a.mines_remaining != ship_b.mines_remaining:
        print("Ship mines remaining don't match!")
        return False
    # TODO: Enable this
    # if 'can_fire' in ship_a and 'can_fire' in ship_b:
    #    if ship_a.can_fire != ship_b.can_fire:
    #        print("Ship can fire don't match!")
    #        return False
    if ship_a.max_speed != ship_b.max_speed:
        print("Ship max speeds don't match!")
        return False

    # Compare positions and velocities
    for i in range(2):
        if ship_a.position[i] != ship_b.position[i]:
            print("Ship positions don't match!")
            return False
        if ship_a.velocity[i] != ship_b.velocity[i]:
            print("Ship velocities don't match!")
            return False

    # Compare other floating-point values
    if ship_a.speed != ship_b.speed:
        print("Ship speeds don't match!")
        return False
    if ship_a.heading != ship_b.heading:
        print("Ship headings don't match!")
        return False
    if ship_a.mass != ship_b.mass:
        print("Ship masses don't match!")
        return False
    if ship_a.radius != ship_b.radius:
        print("Ship radii don't match!")
        return False
    if ship_a.fire_rate != ship_b.fire_rate:
        print("Ship fire rates don't match!")
        return False
    if ship_a.drag != ship_b.drag:
        print("Ship drags don't match!")
        return False

    # Compare tuple values
    for i in range(2):
        if ship_a.thrust_range[i] != ship_b.thrust_range[i]:
            print("Ship thrust ranges don't match!")
            return False
        if ship_a.turn_rate_range[i] != ship_b.turn_rate_range[i]:
            print("Ship turn rates don't match!")
            return False
    return True


def preprocess_bullets(bullets: list[Bullet]) -> list[Bullet]:
    for b in bullets:
        bullet_tail_delta = (-BULLET_LENGTH*cos(radians(b.heading)), -BULLET_LENGTH*sin(radians(b.heading)))
        b.tail_delta = bullet_tail_delta
    return bullets


def preprocess_bullets_in_gamestate(game_state: GameState) -> GameState:
    # UNUSED
    game_state.bullets = preprocess_bullets(game_state.bullets)
    return game_state


def print_explanation(message: str, current_timestep: i64) -> None:
    if not PRINT_EXPLANATIONS:
        return
    global explanation_messages_with_timestamps

    # Check if the message was printed within the time threshold
    last_timestep_printed = explanation_messages_with_timestamps.get(message, INT_NEG_INF)
    if current_timestep - last_timestep_printed >= i64(EXPLANATION_MESSAGE_SILENCE_INTERVAL_S*FPS):
        print(message)
        explanation_messages_with_timestamps[message] = current_timestep


def debug_print(*messages: Any) -> None:
    if DEBUG_MODE:
        print(*messages)


def inspect_scenario(game_state: GameState, ship_state: Ship) -> None:
    asteroids = game_state.asteroids
    width = game_state.map_size[0]
    height = game_state.map_size[1]
    asteroids_count, current_count = asteroid_counter(asteroids)
    if current_count == 0:
        print_explanation("There's no asteroids on the screen! I'm lonely.", 0)
        return
    print_explanation(f"The starting field has {current_count} asteroids on the screen, with a total of {asteroids_count} counting splits.", 0)
    print_explanation(f"At my max shot rate, it'll take {float(asteroids_count)*SHIP_FIRE_TIME:.01f} seconds to clear the field.", 0)
    if ship_state.bullets_remaining == -1:
        print_explanation(f"Yay I have unlimited bullets!", 0)
    elif ship_state.bullets_remaining == 0:
        print_explanation("Oh no, I haven't been given any bullets. I'll just hopefully put on a good show and dodge asteroids until the end of time.", 0)
    else:
        print_explanation(f"Bullets are limited to letting me shoot {float(ship_state.bullets_remaining)/float(asteroids_count):.0%} of the asteroids. If there's another ship, I'll be careful not to let them steal my shots! Otherwise, I'll shoot away!", 0)

    def asteroid_density() -> float:
        total_asteroid_coverage_area = 0.0
        for a in asteroids:
            total_asteroid_coverage_area += ASTEROID_AREA_LOOKUP[a.size]
        total_screen_size = width*height
        if total_screen_size == 0.0:
            return 0.0
        else:
            return total_asteroid_coverage_area/total_screen_size

    def average_velocity() -> tuple[float, float]:
        total_x_velocity = 0.0
        total_y_velocity = 0.0
        for a in asteroids:
            ast_vel: tuple[float, float] = a.velocity
            total_x_velocity += ast_vel[0]
            total_y_velocity += ast_vel[1]
        num_asteroids = len(asteroids)
        if num_asteroids == 0:
            return (0, 0)
        else:
            return (total_x_velocity/num_asteroids, total_y_velocity/num_asteroids)

    def average_speed() -> float:
        total_speed = 0.0
        for a in asteroids:
            total_speed += sqrt(a.velocity[0]*a.velocity[0] + a.velocity[1]*a.velocity[1])
        num_asteroids = len(asteroids)
        if num_asteroids == 0:
            return 0.0
        else:
            return total_speed/num_asteroids

    # average_density = asteroid_density()
    # current_asteroids, total_asteroids = asteroid_counter(asteroids)
    # average_vel = average_velocity()
    # avg_speed = average_speed()
    # print(f"Average asteroid density: {average_density}, average vel: {average_vel}, average speed: {avg_speed}")


def asteroid_counter(asteroids: list[Asteroid]) -> tuple[i64, i64]:
    current_count = len(asteroids)
    total_count: i64 = 0
    for a in asteroids:
        total_count += ASTEROID_COUNT_LOOKUP[a.size]
    return total_count, current_count


class GameStatePlotter:
    # Use matplotlib to plot out the gamestate to view debug traces and stuff
    def __init__(self, game_state: GameState) -> None:
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, game_state.map_size[0]])
        self.ax.set_ylim([0, game_state.map_size[1]])
        self.ax.set_aspect('equal', adjustable='box')
        self.game_state = game_state

    def fuse_time_to_color(self, remaining_time: float) -> str:
        if remaining_time >= MINE_FUSE_TIME:
            return "#00FF00"  # Green
        elif remaining_time <= 0.0:
            return "#FF0000"  # Red
        else:
            # Linearly interpolate between green and red
            # From green to yellow
            if remaining_time > 1.5:
                # Interpolate between green and yellow
                green_to_yellow_ratio = (remaining_time - 1.5)/1.5
                red = i64(255*(1 - green_to_yellow_ratio))
                green = 255
            # From yellow to red
            else:
                # Interpolate between yellow and red
                yellow_to_red_ratio = remaining_time/1.5
                red = 255
                green = i64(255*yellow_to_red_ratio)

            # Convert to hex
            return f"#{red:02x}{green:02x}00"

    def update_plot(self, asteroids: Optional[list[Asteroid]] = None, ship_state: Optional[Ship] = None, bullets: Optional[list[Bullet]] = None, special_bullets: Optional[list[Bullet]] = None, circled_asteroids: Optional[list[Asteroid]] = None, ghost_asteroids: Optional[list[Asteroid]] = None, forecasted_asteroids: Optional[list[Asteroid]] = None, mines: Optional[list[Mine]] = None, clear_plot: bool = True, pause_time: float = EPS, plot_title: str = '') -> None:
        if asteroids is None:
            asteroids = []
        if bullets is None:
            bullets = []
        if special_bullets is None:
            special_bullets = []
        if circled_asteroids is None:
            circled_asteroids = []
        if ghost_asteroids is None:
            ghost_asteroids = []
        if forecasted_asteroids is None:
            forecasted_asteroids = []
        if mines is None:
            mines = []
        if clear_plot:
            self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.set_xlim([0, self.game_state.map_size[0]])
        self.ax.set_ylim([0, self.game_state.map_size[1]])
        self.ax.set_aspect('equal', adjustable='box')

        self.ax.set_title(plot_title, fontsize=14, color='black')

        # Draw asteroids and their velocities
        for a in asteroids:
            if a:
                asteroid_circle = patches.Circle(a.position, a.radius, color='gray', fill=True)
                self.ax.add_patch(asteroid_circle)
                self.ax.arrow(a.position[0], a.position[1], a.velocity[0]*DELTA_TIME, a.velocity[1]*DELTA_TIME, head_width=3, head_length=5, fc='white', ec='white')
        for a in ghost_asteroids:
            if a:
                asteroid_circle = patches.Circle(a.position, a.radius, color='#333333', fill=True, zorder=-100)
                self.ax.add_patch(asteroid_circle)
                # self.ax.arrow(a.position[0], a.position[1], a.velocity[0]*delta_time, a.velocity[1]*delta_time, head_width=3, head_length=5, fc='white', ec='white')
        for a in forecasted_asteroids:
            if a:
                asteroid_circle = patches.Circle(a.position, a.radius, color='#440000', fill=True, zorder=100, alpha=0.4)
                self.ax.add_patch(asteroid_circle)
        # print(highlighted_asteroids)
        for a in circled_asteroids:
            if a:
                # print('asteroid', a)
                highlight_circle = patches.Circle(a.position, a.radius + 5, color='orange', fill=False)
                self.ax.add_patch(highlight_circle)

        for m in mines:
            if m:
                highlight_circle = patches.Circle(m.position, MINE_BLAST_RADIUS, color=self.fuse_time_to_color(m.remaining_time), fill=False)
                self.ax.add_patch(highlight_circle)

        # Hard code a circle so I can see what a coordinate on screen is for debugging
        circle_hardcoded_coordinate = False
        if circle_hardcoded_coordinate:
            highlight_circle = patches.Circle((1017.3500530032204, 423.2001881178426), 25, color='red', fill=False)
            self.ax.add_patch(highlight_circle)

        if ship_state:
            # Draw the ship as an elongated triangle
            ship_size_base = SHIP_RADIUS
            ship_size_tip = SHIP_RADIUS
            ship_heading = ship_state.heading
            ship_position = ship_state.position
            angle_rad = radians(ship_heading)
            ship_vertices = [
                (ship_position[0] + ship_size_tip*cos(angle_rad), ship_position[1] + ship_size_tip*sin(angle_rad)),
                (ship_position[0] + ship_size_base*cos(angle_rad + pi*3/4), ship_position[1] + ship_size_base*sin(angle_rad + pi*3/4)),
                (ship_position[0] + ship_size_base*cos(angle_rad - pi*3/4), ship_position[1] + ship_size_base*sin(angle_rad - pi*3/4)),
            ]
            ship = patches.Polygon(ship_vertices, color='green', fill=True)
            self.ax.add_patch(ship)

            # Draw the ship's hitbox as a blue circle
            ship_circle = patches.Circle(ship_position, SHIP_RADIUS, color='blue', fill=False)
            self.ax.add_patch(ship_circle)

        # Draw arrow line segments for bullets
        for b in bullets:
            if b:
                bullet_tail = (b.position[0] - BULLET_LENGTH*cos(radians(b.heading)), b.position[1] - BULLET_LENGTH*sin(radians(b.heading)))
                self.ax.arrow(bullet_tail[0], bullet_tail[1], b.position[0] - bullet_tail[0], b.position[1] - bullet_tail[1], head_width=3, head_length=5, fc='red', ec='red')
        for b in special_bullets:
            if b:
                bullet_tail = (b.position[0] - BULLET_LENGTH*cos(radians(b.heading)), b.position[1] - BULLET_LENGTH*sin(radians(b.heading)))
                self.ax.arrow(bullet_tail[0], bullet_tail[1], b.position[0] - bullet_tail[0], b.position[1] - bullet_tail[1], head_width=3, head_length=5, fc='green', ec='green')
        plt.draw()
        plt.pause(pause_time)


def append_dict_to_file(dict_data: dict[Any, Any], file_path: str) -> None:
    """
    This function takes a dictionary and appends it as a string in a JSON-like format to a text file.

    :param dict_data: dictionary to be converted to a string and appended.
    :param file_path: Path of the text file to append the data to.
    """
    import json

    # Convert the dictionary to a JSON-like string
    dict_string = json.dumps(dict_data, indent=4)

    # Append the string to the file
    with open(file_path, 'a') as file:
        file.write(dict_string + "\n")


def get_other_ships(game_state: GameState, self_ship_id: i64) -> list[Ship]:
    return [ship for ship in game_state.ships if ship.id != self_ship_id]


def angle_difference_rad(angle1: float, angle2: float) -> float:
    # Calculate the raw difference, and use modulo to wrap around the angle to between -pi to pi
    return (angle1 - angle2 + pi) % TAU - pi


def angle_difference_deg(angle1: float, angle2: float) -> float:
    # Calculate the raw difference, and use modulo to wrap around the angle to between -180 to 180
    return (angle1 - angle2 + 180.0) % 360.0 - 180.0


def get_ship_maneuver_move_sequence(ship_heading_angle: float, ship_cruise_speed: float, ship_accel_turn_rate: float, ship_cruise_timesteps: i64, ship_cruise_turn_rate: float, ship_starting_speed: float = 0.0) -> list[Action]:
    # This is a silly code duplication to get a super optimized class for getting the ship's move sequence. Using the entire Simulation class is overkill and too much.
    move_sequence: list[Action] = []
    ship_speed = ship_starting_speed
    # assert isclose(ship_starting_speed, 0.0), f"The ship maneuver should start with 0 speed! It's actually {ship_starting_speed}. Unless... we just were saved by the other ship, so we're starting a maneuver while in the middle of another maneuver! In which case, you can safely ignore this assertion."
    #if not is_close(ship_starting_speed, 0.0):
    #    print(f"The ship maneuver should start with 0 speed! It's actually {ship_starting_speed}. Unless... we just were saved by the other ship, so we're starting a maneuver while in the middle of another maneuver!")
    def rotate_heading(heading_difference_deg: float) -> None:
        nonlocal move_sequence
        if abs(heading_difference_deg) < GRAIN:
            return
        still_need_to_turn = heading_difference_deg
        while abs(still_need_to_turn) > SHIP_MAX_TURN_RATE*DELTA_TIME:
            update(0.0, SHIP_MAX_TURN_RATE*sign(heading_difference_deg))
            still_need_to_turn -= SHIP_MAX_TURN_RATE*sign(heading_difference_deg)*DELTA_TIME
        if abs(still_need_to_turn) > EPS:
            update(0.0, still_need_to_turn*FPS)

    def accelerate(target_speed: float, turn_rate: float) -> None:
        nonlocal move_sequence
        nonlocal ship_speed
        # Keep in mind speed can be negative
        # Drag will always slow down the ship
        while abs(target_speed - ship_speed) > EPS:
            drag = -SHIP_DRAG*sign(ship_speed)
            drag_amount = SHIP_DRAG*DELTA_TIME
            if drag_amount > abs(ship_speed):
                # The drag amount is reduced if it would make the ship cross 0 speed on its own
                adjust_drag_by = abs((drag_amount - abs(ship_speed))*FPS)
                drag -= adjust_drag_by*sign(drag)
            delta_speed_to_target = target_speed - ship_speed
            thrust_amount = delta_speed_to_target*FPS - drag
            thrust_amount = min(max(-SHIP_MAX_THRUST, thrust_amount), SHIP_MAX_THRUST)
            update(thrust_amount, turn_rate)

    def cruise(cruise_timesteps: i64, cruise_turn_rate: float) -> None:
        nonlocal move_sequence
        nonlocal ship_speed
        # Maintain current speed
        for _ in range(cruise_timesteps):
            update(sign(ship_speed)*SHIP_DRAG, cruise_turn_rate)

    def update(thrust: float, turn_rate: float) -> None:
        nonlocal move_sequence
        nonlocal ship_speed
        # Apply drag. Fully stop the ship if it would cross zero speed in this time (prevents oscillation)
        drag_amount = SHIP_DRAG*DELTA_TIME
        if drag_amount > abs(ship_speed):
            ship_speed = 0.0
        else:
            ship_speed -= drag_amount*sign(ship_speed)
        # Bounds check the thrust
        thrust = min(max(-SHIP_MAX_THRUST, thrust), SHIP_MAX_THRUST)
        # Apply thrust
        ship_speed += thrust*DELTA_TIME
        # Bounds check the speed
        if ship_speed > SHIP_MAX_SPEED:
            ship_speed = SHIP_MAX_SPEED
        elif ship_speed < -SHIP_MAX_SPEED:
            ship_speed = -SHIP_MAX_SPEED
        #if not isclose(ship_starting_speed, 0.0):
            #print(f"New ship speed after thrusting by {thrust}: {ship_speed}")
        move_sequence.append(Action(thrust=thrust, turn_rate=turn_rate, fire=False))

    rotate_heading(ship_heading_angle)
    accelerate(ship_cruise_speed, ship_accel_turn_rate)
    cruise(ship_cruise_timesteps, ship_cruise_turn_rate)
    accelerate(0.0, 0.0)  # TODO: If we remove this, we get some interesting results and emergent behavior. Neo would spazz around the map, going from one maneuver directly into another. I might even be able to tweak it to work. Maybe in the stationary targeting, apply a thrust to slow down the ship, so the targeting works a bit better. That could fix the issue, and it might be worth exploring if I have time! But for now, let’s just eat the slight time loss and regain the control of node based movement without drifting around like driftwood.
    #if not is_close(ship_starting_speed, 0.0):
    #    print(f"Interesting move sequence! The ship didn't start stationary. It had speed {ship_starting_speed}, and we used sequence {move_sequence} to return the ship to speed {ship_speed}")
    if not move_sequence:
        # We still need a null sequence here, so that we don't end up with a 0 frame maneuver!
        move_sequence.append(Action(thrust=0.0, turn_rate=0.0, fire=False))
    return move_sequence


def analyze_gamestate_for_heuristic_maneuver(game_state: GameState, ship_state: Ship) -> tuple[float, float, float, float, i64, float, i64, i64]:
    # This is a helper function to analyze and prepare the gamestate, to give the maneuver FIS useful information, to heuristically command a maneuver to try out
    def calculate_angular_width(radius: float, distance: float) -> float:
        # From the ship's point of view, find the angular width of an asteroid
        if distance == 0.0:
            return TAU
        sin_theta = radius/distance
        if -1.0 <= sin_theta <= 1.0:
            return 2.0*super_fast_asin(sin_theta)
        else:
            return TAU

    def average_velocity(asteroids: list[Asteroid]) -> tuple[float, float]:
        total_x_velocity = 0.0
        total_y_velocity = 0.0
        for a in asteroids:
            ast_vel: tuple[float, float] = a.velocity
            total_x_velocity += ast_vel[0]
            total_y_velocity += ast_vel[1]
        num_asteroids = len(asteroids)
        if num_asteroids == 0.0:
            return (0, 0)
        else:
            return (total_x_velocity/num_asteroids, total_y_velocity/num_asteroids)

    def find_largest_gap(asteroids: list[Asteroid], ship_position: tuple[float, float]) -> tuple[float, float]:
        # Find the largest angular gap around the ship, and this is the gap I'll try escaping through
        if not asteroids:
            # No asteroids mean the entire space is a gap.
            return 0.0, TAU

        angles: list[tuple[float, bool]] = []
        initial_cover_count: i64 = 0  # Counter for asteroids covering angle 0

        for asteroid in asteroids:
            x = asteroid.position[0] - ship_position[0]
            y = asteroid.position[1] - ship_position[1]
            distance = sqrt(x*x + y*y)
            angle = super_fast_atan2(y, x) % TAU
            angular_width = calculate_angular_width(asteroid.radius, distance)
            start_angle = (angle - 0.5*angular_width) % TAU
            end_angle = (angle + 0.5*angular_width) % TAU

            # Check if this asteroid covers the angle 0 (or equivalently, 2π)
            if start_angle > end_angle:  # This means the asteroid wraps around angle 0
                initial_cover_count += 1

            # Add angles in the original and offset positions
            # True is for start and False is for end
            angles.append((start_angle, True))
            angles.append((end_angle, False))
            angles.append((start_angle + TAU, True))
            angles.append((end_angle + TAU, False))

        # Sort by angle
        angles.sort(key=lambda x: x[0])

        # Initialize counter with the number of asteroids covering angle 0
        counter = initial_cover_count
        largest_gap_midpoint = 0.0
        largest_gap = 0.0
        gap_start: Optional[float] = None

        for angle, marker in angles:
            if marker:
                # Start
                if counter == 0 and gap_start is not None:
                    # Calculate and check the gap size
                    gap = angle - gap_start
                    if gap > largest_gap:
                        largest_gap = gap
                        largest_gap_midpoint = 0.5*(gap_start + angle) % TAU
                counter += 1
            else:
                # End
                counter -= 1
                if counter == 0:
                    # Mark the start of a new gap
                    gap_start = angle

        # No need to adjust for wraparound explicitly due to "doubling" the angles list
        return largest_gap_midpoint, largest_gap

    asteroids = list(game_state.asteroids)
    other_ships = []
    for ship in game_state.ships:
        if ship.id != ship_state.id:
            other_ships.append(ship)
    for ship in other_ships:
        # Fake ships as asteroids
        #asteroids.append({'position': ship.position, 'velocity': (0, 0), 'radius': ship.radius, 'size': -1, 'mass': -1.0})
        asteroids.append(Asteroid(position=ship.position, velocity=(0.0, 0.0), radius=ship.radius, size=-1, mass=-1.0))
    ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y = ship_state.position[0], ship_state.position[1], ship_state.velocity[0], ship_state.velocity[1]
    most_imminent_collision_time_s = inf
    most_imminent_asteroid = None
    most_imminent_asteroid_speed = None
    nearby_asteroid_total_speed = 0.0
    nearby_asteroid_count = 0
    nearby_threshold_square = 40000.0 #200.0**2
    nearby_asteroids = []
    for asteroid in asteroids:
        for a in unwrap_asteroid(asteroid, game_state.map_size[0], game_state.map_size[1], UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON, False):
            imminent_collision_time_s = predict_next_imminent_collision_time_with_asteroid(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, SHIP_RADIUS, a.position[0], a.position[1], a.velocity[0], a.velocity[1], a.radius)
            delta_x = a.position[0] - ship_pos_x
            delta_y = a.position[1] - ship_pos_y
            asteroid_speed = None
            if delta_x*delta_x + delta_y*delta_y <= nearby_threshold_square:
                asteroid_speed = sqrt(a.velocity[0]*a.velocity[0] + a.velocity[1]*a.velocity[1])
                nearby_asteroid_total_speed += asteroid_speed
                nearby_asteroid_count += 1
                nearby_asteroids.append(a)
            if imminent_collision_time_s < most_imminent_collision_time_s:
                most_imminent_collision_time_s = imminent_collision_time_s
                most_imminent_asteroid = a
                if asteroid_speed is not None:
                    most_imminent_asteroid_speed = asteroid_speed
                else:
                    most_imminent_asteroid_speed = None
    if most_imminent_asteroid is None:
        most_imminent_asteroid_speed = 0.0
        imminent_asteroid_relative_heading_deg = 0.0
    else:
        if most_imminent_asteroid_speed is None:
            most_imminent_asteroid_speed = sqrt(most_imminent_asteroid.velocity[0]*most_imminent_asteroid.velocity[0] + most_imminent_asteroid.velocity[1]*most_imminent_asteroid.velocity[1])
        imminent_asteroid_relative_heading_deg = degrees(super_fast_atan2(most_imminent_asteroid.position[1] - ship_pos_y, most_imminent_asteroid.position[0] - ship_pos_x))

    largest_gap_absolute_heading_rad, _ = find_largest_gap(nearby_asteroids, (ship_pos_x, ship_pos_y))
    largest_gap_absolute_heading_deg = degrees(largest_gap_absolute_heading_rad)
    largest_gap_relative_heading_deg = (largest_gap_absolute_heading_deg - ship_state.heading) % TAU
    if nearby_asteroid_count == 0:
        nearby_asteroid_average_speed = 0.0
    else:
        nearby_asteroid_average_speed = nearby_asteroid_total_speed/nearby_asteroid_count

    average_directional_velocity = average_velocity(asteroids)
    average_directional_speed = sqrt(average_directional_velocity[0]*average_directional_velocity[0] + average_directional_velocity[1]*average_directional_velocity[1])
    total_asteroid_count, current_asteroids_count = asteroid_counter(asteroids)
    return most_imminent_asteroid_speed, imminent_asteroid_relative_heading_deg, largest_gap_relative_heading_deg, nearby_asteroid_average_speed, nearby_asteroid_count, average_directional_speed, total_asteroid_count, current_asteroids_count


def check_collision(a_x: float, a_y: float, a_r: float, b_x: float, b_y: float, b_r: float) -> bool:
    # Since this is called so often, it's faster to inline this function instead of calling it
    delta_x = a_x - b_x
    delta_y = a_y - b_y
    separation = a_r + b_r
    # Because most of the time the assumption is that there will be no collision, it's faster to do a quick rejection check, and only when it's possible for them to collide, we do the slightly more expensive squaring check
    if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
        return True
    else:
        return False


def collision_prediction(Oax: float, Oay: float, Dax: float, Day: float, ra: float, Obx: float, Oby: float, Dbx: float, Dby: float, rb: float) -> tuple[float, float]:
    # Given two circles moving at constant velocities, will they collide, and when? This can be solved in constant time with a calculation
    # https://stackoverflow.com/questions/11369616/circle-circle-collision-prediction/
    separation = ra + rb
    delta_x = Oax - Obx
    delta_y = Oay - Oby
    if is_close_to_zero(Dax) and is_close_to_zero(Day) and is_close_to_zero(Dbx) and is_close_to_zero(Dby):
        # If both objects are stationary, then we only have to check the collision right now and not do any fancy math
        # This should speed up scenarios where most asteroids are stationary
        #if check_collision(Oax, Oay, ra, Obx, Oby, rb):
        if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
            return -inf, inf
        else:
            return nan, nan
    else:
        # Compared to the stack overflow, the following has been factored to reduce intermediate computations and save some clock cycles
        vel_delta_x = Dax - Dbx
        vel_delta_y = Day - Dby
        a = vel_delta_x*vel_delta_x + vel_delta_y*vel_delta_y
        b = 2.0*(delta_x*vel_delta_x + delta_y*vel_delta_y)
        c = delta_x*delta_x + delta_y*delta_y - separation*separation
        return solve_quadratic(a, b, c)


def predict_next_imminent_collision_time_with_asteroid(ship_pos_x: float, ship_pos_y: float, ship_vel_x: float, ship_vel_y: float, ship_r: float, ast_pos_x: float, ast_pos_y: float, ast_vel_x: float, ast_vel_y: float, ast_radius: float) -> float:
    # print("ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius")
    # print(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius)
    t1, t2 = collision_prediction(ship_pos_x, ship_pos_y, ship_vel_x, ship_vel_y, ship_r, ast_pos_x, ast_pos_y, ast_vel_x, ast_vel_y, ast_radius)
    # If we're already colliding with something, then return 0 as the next imminent collision time
    if isnan(t1) or isnan(t2):
        next_imminent_collision_time = inf
    else:
        if t1 <= t2:
            start_collision_time = t1
            end_collision_time = t2
        else:
            start_collision_time = t2
            end_collision_time = t1
        if end_collision_time < 0.0:
            next_imminent_collision_time = inf
        elif start_collision_time <= 0.0:
            # start_collision_time <= 0.0 <= end_collision_time
            next_imminent_collision_time = 0.0
        else:
            # start_collision_time > 0.0 and 0.0 <= end_collision_time
            next_imminent_collision_time = start_collision_time
    return next_imminent_collision_time


# It helps a bit to LRU cache this, but I turned this into a generator function so we can't do that anymore, RIP
def calculate_border_crossings(x0: float, y0: float, vx: float, vy: float, W: float, H: float, c: float) -> list[tuple[i64, i64]]:
    # Initialize lists to hold crossing times
    x_crossings_times = []
    y_crossings_times = []
    x_crossings = 0
    y_crossings = 0

    # Calculate crossing times for x (if vx is not zero to avoid division by zero)
    abs_vx = abs(vx)
    if abs_vx > EPS:
        # Calculate time to first x-boundary crossing based on direction of vx
        x_crossing_interval = W/abs_vx
        # print(f"x_crossing_interval: {x_crossing_interval}")
        time_to_first_x_crossing = ((W - x0)/vx if vx > 0.0 else -x0/vx)
        x_crossings_times.append(time_to_first_x_crossing)
        x_crossings += 1
        # Add additional crossings until time c is reached
        while (next_time := x_crossings_times[-1] + x_crossing_interval) <= c:
            x_crossings_times.append(next_time)
            x_crossings += 1
    # print(f"x crossing times: {x_crossings_times}")
    # Calculate crossing times for y (if vy is not zero)
    abs_vy = abs(vy)
    if abs_vy > EPS:
        # Calculate time to first y-boundary crossing based on direction of vy
        y_crossing_interval = H/abs_vy
        # print(f"y_crossing_interval: {y_crossing_interval}")
        time_to_first_y_crossing = ((H - y0)/vy if vy > 0.0 else -y0/vy)
        y_crossings_times.append(time_to_first_y_crossing)
        y_crossings += 1
        # Add additional crossings until time c is reached
        while (next_time := y_crossings_times[-1] + y_crossing_interval) <= c:
            y_crossings_times.append(next_time)
            y_crossings += 1
    # print(f"y crossing times: {y_crossings_times}")
    # Merge the two lists while tracking the origin of each time
    #merged_times = []
    border_crossing_sequence: list[bool] = []
    i = 0
    j = 0

    # True is for x, False is for y
    while i < x_crossings and j < y_crossings:
        if x_crossings_times[i] < y_crossings_times[j]:
            #merged_times.append(x_crossings_times[i])
            border_crossing_sequence.append(True)
            i += 1
        else:
            #merged_times.append(y_crossings_times[j])
            border_crossing_sequence.append(False)
            j += 1

    # Add any remaining times from the x_crossings_times list
    while i < x_crossings:
        #merged_times.append(x_crossings_times[i])
        border_crossing_sequence.append(True)
        i += 1

    # Add any remaining times from the y_crossings_times list
    while j < y_crossings:
        #merged_times.append(y_crossings_times[j])
        border_crossing_sequence.append(False)
        j += 1

    # Initialize current universe coordinates and list of visited universes
    current_universe_x: i64 = 0
    current_universe_y: i64 = 0
    universe_increment_direction_x: i64 = 1 if vx > 0.0 else -1
    universe_increment_direction_y: i64 = 1 if vy > 0.0 else -1

    # Iterate through merged crossing times and sequence
    #universes = [(current_universe_x, current_universe_y)]
    universes = []
    for crossing in border_crossing_sequence:
        if crossing: # Crossing is for x
            current_universe_x += universe_increment_direction_x
        else:  # crossing is for y
            current_universe_y += universe_increment_direction_y
        universes.append((current_universe_x, current_universe_y))
    return universes


def unwrap_asteroid(asteroid: Asteroid, max_x: float, max_y: float, time_horizon_s: float = 10.0, use_cache: bool = True) -> list[Asteroid]:
    #start_time = time.perf_counter()
    #global unwrap_total_time
    #time_horizon_s = UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON
    #use_cache = UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON == time_horizon_s
    #hash_tuple = (asteroid.position[0], asteroid.position[1], asteroid.velocity[0], asteroid.velocity[1], time_horizon_s)
    if use_cache:
        ast_hash = asteroid.int_hash()
        global unwrap_cache
        if ast_hash in unwrap_cache:
            #print("CACHE HIT")
            #global unwrap_cache_hits
            #unwrap_cache_hits += 1
            #unwrap_total_time += time.perf_counter() - start_time
            return unwrap_cache[ast_hash]
    #print("CACHE MISS")
    #global unwrap_cache_misses
    #unwrap_cache_misses += 1
    unwrapped_asteroids: list[Asteroid] = [asteroid.copy()]
    if abs(asteroid.velocity[0]) < EPS and abs(asteroid.velocity[1]) < EPS:
        # An asteroid that is stationary will never move across borders
        if use_cache:
            unwrap_cache[ast_hash] = unwrapped_asteroids
        #unwrap_total_time += time.perf_counter() - start_time
        return unwrapped_asteroids

    # The idea is to track which universes the asteroid visits from t=t_0 until t=t_0 + time_horizon_s.
    # The current universe is (0, 0) and if the asteroid wraps to the right, it visits (1, 0). If it wraps down, it visits (0, -1). If it wraps right and then down, it starts in (0, 0), visits (1, 0), and finally (1, -1).
    #border_crossings = calculate_border_crossings(asteroid.position[0], asteroid.position[1], asteroid.velocity[0], asteroid.velocity[1], max_x, max_y, time_horizon_s)
    for universe in calculate_border_crossings(asteroid.position[0], asteroid.position[1], asteroid.velocity[0], asteroid.velocity[1], max_x, max_y, time_horizon_s):
        # We negate the directions because we're using the frame of reference of the ship now, not the asteroid
        dx = -float(universe[0])*max_x
        dy = -float(universe[1])*max_y
        #unwrapped_asteroid: Asteroid = asteroid.copy()
        #unwrapped_asteroid.position = (unwrapped_asteroid.position[0] + dx, unwrapped_asteroid.position[1] + dy)
        #unwrapped_asteroids.append(unwrapped_asteroid)
        unwrapped_asteroids.append(Asteroid(
            position=(asteroid.position[0] + dx, asteroid.position[1] + dy),
            velocity=asteroid.velocity,
            size=asteroid.size,
            mass=asteroid.mass,
            radius=asteroid.radius,
            timesteps_until_appearance=asteroid.timesteps_until_appearance
        ))
        #yield unwrapped_asteroid
    # print(f"Returning unwrapped asteroids: {unwrapped_asteroids}")
    if use_cache:
        unwrap_cache[ast_hash] = unwrapped_asteroids
    #unwrap_total_time += time.perf_counter() - start_time
    return unwrapped_asteroids


def check_coordinate_bounds(game_state: GameState, x: float, y: float) -> bool:
    if 0 <= x <= game_state.map_size[0] and 0 <= y <= game_state.map_size[1]:
        return True
    else:
        return False


def check_coordinate_bounds_exact(game_state: GameState, x: float, y: float) -> bool:
    x_wrapped = x % game_state.map_size[0]
    y_wrapped = y % game_state.map_size[1]
    # TODO: Maybe optimize?
    if is_close(x, x_wrapped) and is_close(y, y_wrapped):
        return True
    else:
        return False


def solve_quadratic(a: float, b: float, c: float) -> tuple[float, float]:
    # This solves a*x*x + b*x + c = 0 for x
    # This handles the case where a, b, or c are 0.
    d = b*b - 4.0*a*c
    if d < 0.0:
        # No real solutions.
        r1 = nan
        r2 = nan
    elif a == 0.0:
        # This is a linear equation. Handle this case separately.
        if b != 0.0:
            r1 = -c/b
            r2 = nan
        else:
            # I doubt this case will ever get hit, but include anyway
            if c == 0.0:
                r1 = 0.0
                r2 = nan
            else:
                r1 = nan
                r2 = nan
    else:
        # This handles the case where b or c are 0
        # If d is 0, technically there's only one solution but this will give two duplicated solutions. It's not worth checking each time for this since it's so rare
        if b > 0.0:
            u = -b - sqrt(d)
        else:
            u = -b + sqrt(d)
        r1 = u/(2.0*a)
        if u != 0.0:
            r2 = 2.0*c/u
        else:
            r2 = nan
    return r1, r2


def calculate_interception(ship_pos_x: float, ship_pos_y: float, asteroid_pos_x: float, asteroid_pos_y: float, asteroid_vel_x: float, asteroid_vel_y: float, asteroid_r: float, ship_heading_deg: float, game_state: GameState, future_shooting_timesteps: i64 = 0) -> tuple[bool, float, float, float, float, float, float]:
    # This is a simplified version of solve_interception(). This will, given the position of the ship and an asteroid, tell you which angle you need to fire at after future_shooting_timesteps to shoot the asteroid
    # The bullet's head originates from the edge of the ship's radius.
    # We want to set the position of the bullet to the center of the bullet, so we have to do some fanciness here so that at t=0, the bullet's center is where it should be
    t_0 = 0.0175 # t_0 = (SHIP_RADIUS - 0.5*BULLET_LENGTH)/BULLET_SPEED
    # Positions are relative to the ship. We set the origin to the ship's position. Remember to translate back!
    origin_x = ship_pos_x
    origin_y = ship_pos_y
    avx = asteroid_vel_x
    avy = asteroid_vel_y
    ax = asteroid_pos_x - origin_x + avx*DELTA_TIME  # We project the asteroid one timestep ahead, since by the time we shoot our bullet, the asteroid would have moved one more timestep!
    ay = asteroid_pos_y - origin_y + avy*DELTA_TIME

    vb = BULLET_SPEED
    vb_sq = vb*vb
    theta_0 = radians(ship_heading_deg)

    # Calculate constants for naive_desired_heading_calc
    a = avx*avx + avy*avy - vb_sq

    time_until_can_fire_s = float(future_shooting_timesteps)*DELTA_TIME
    ax_delayed = ax + time_until_can_fire_s*avx  # We add a delay to account for the timesteps until we can fire delay
    ay_delayed = ay + time_until_can_fire_s*avy

    b = 2.0*(ax_delayed*avx + ay_delayed*avy - vb_sq*t_0)
    c = ax_delayed*ax_delayed + ay_delayed*ay_delayed - vb_sq*t_0*t_0

    #solutions = []
    for t in solve_quadratic(a, b, c):
        if isnan(t) or t < 0.0:
            # Invalid interception time
            continue
        x = ax_delayed + t*avx
        y = ay_delayed + t*avy
        theta = fast_atan2(y, x)
        # If the asteroid is out of bounds, then it will wrap around and this shot isn't feasible
        # However, if an unwrapped asteroid was passed into this function and the interception is inbounds, then it's a feasible shot
        intercept_x = x + origin_x
        intercept_y = y + origin_y
        feasible = check_coordinate_bounds(game_state, intercept_x, intercept_y)
        if not feasible:
            continue
        asteroid_dist = sqrt(x*x + y*y)
        if asteroid_r < asteroid_dist:
            shot_heading_tolerance_rad = super_fast_asin((asteroid_r - ASTEROID_AIM_BUFFER_PIXELS)/asteroid_dist)
        else:
            shot_heading_tolerance_rad = 0.5*pi
        return feasible, angle_difference_rad(theta, theta_0), shot_heading_tolerance_rad, t, intercept_x, intercept_y, asteroid_dist
    #if len(solutions) > 1:
    #    print(len(solutions))
    #return solutions
    return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    # feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time_s + 0*future_shooting_timesteps*delta_time, intercept_x, intercept_y, asteroid_dist, asteroid_dist_during_interception


def forecast_asteroid_bullet_splits_from_heading(a: Asteroid, timesteps_until_appearance: i64, bullet_heading_deg: float, game_state: GameState) -> list[Asteroid]:
    # Look at asteroid.py in the Kessler game's code
    bullet_heading_rad = radians(bullet_heading_deg)
    bullet_vel_x = cos(bullet_heading_rad)*BULLET_SPEED
    bullet_vel_y = sin(bullet_heading_rad)*BULLET_SPEED
    vfx = (1/(BULLET_MASS + a.mass))*(BULLET_MASS*bullet_vel_x + a.mass*a.velocity[0])
    vfy = (1/(BULLET_MASS + a.mass))*(BULLET_MASS*bullet_vel_y + a.mass*a.velocity[1])
    v = sqrt(vfx*vfx + vfy*vfy)
    return forecast_asteroid_splits(a, timesteps_until_appearance, vfx, vfy, v, 15.0, game_state)


def forecast_instantaneous_asteroid_bullet_splits_from_velocity(a: Asteroid, bullet_velocity: tuple[float, float], game_state: GameState) -> list[Asteroid]:
    # Look at asteroid.py in the Kessler game's code
    bullet_vel_x, bullet_vel_y = bullet_velocity
    vfx = (1/(BULLET_MASS + a.mass))*(BULLET_MASS*bullet_vel_x + a.mass*a.velocity[0])
    vfy = (1/(BULLET_MASS + a.mass))*(BULLET_MASS*bullet_vel_y + a.mass*a.velocity[1])
    v = sqrt(vfx*vfx + vfy*vfy)
    return forecast_asteroid_splits(a, 0, vfx, vfy, v, 15.0, game_state)


def forecast_asteroid_mine_instantaneous_splits(asteroid: Asteroid, mine: Mine, game_state: GameState) -> list[Asteroid]:
    delta_x = mine.position[0] - asteroid.position[0]
    delta_y = mine.position[1] - asteroid.position[1]
    dist = sqrt(delta_x*delta_x + delta_y*delta_y)
    F = (-dist/MINE_BLAST_RADIUS + 1.0)*MINE_BLAST_PRESSURE*2.0*asteroid.radius
    a = F/asteroid.mass
    # calculate "impulse" based on acc
    if dist != 0.0:
        cos_theta = (asteroid.position[0] - mine.position[0])/dist
        sin_theta = (asteroid.position[1] - mine.position[1])/dist
        vfx = asteroid.velocity[0] + a*cos_theta
        vfy = asteroid.velocity[1] + a*sin_theta
        v = sqrt(vfx*vfx + vfy*vfy)
        split_angle = 15.0
    else:
        vfx = asteroid.velocity[0]
        vfy = asteroid.velocity[1]
        v = sqrt(vfx*vfx + vfy*vfy + a*a)
        split_angle = 120.0
    return forecast_asteroid_splits(asteroid, 0, vfx, vfy, v, split_angle, game_state)


def forecast_asteroid_ship_splits(asteroid: Asteroid, timesteps_until_appearance: i64, ship_velocity: tuple[float, float], game_state: GameState) -> list[Asteroid]:
    vfx = (1/(SHIP_MASS + asteroid.mass))*(SHIP_MASS*ship_velocity[0] + asteroid.mass*asteroid.velocity[0])
    vfy = (1/(SHIP_MASS + asteroid.mass))*(SHIP_MASS*ship_velocity[1] + asteroid.mass*asteroid.velocity[1])
    v = sqrt(vfx*vfx + vfy*vfy)
    return forecast_asteroid_splits(asteroid, timesteps_until_appearance, vfx, vfy, v, 15.0, game_state)


def forecast_asteroid_splits(a: Asteroid, timesteps_until_appearance: i64, vfx: float, vfy: float, v: float, split_angle: float, game_state: GameState) -> list[Asteroid]:
    # Calculate angle of center asteroid for split (degrees)
    theta = degrees(atan2(vfy, vfx)) # DO NOT USE AN APPROXIMATION FOR ATAN2!! This needs to match Kessler or else we can get desyncs.
    # Split angle is the angle off of the new velocity vector for the two asteroids to the sides, the center child asteroid continues on the new velocity path
    angles = (radians(theta + split_angle), radians(theta), radians(theta - split_angle))
    # We redundantly convert to degrees, add split angle, and back to radians. But it has to do this to match Kessler. No way to optimize without risking desyncs.
    # This is wacky because we're back-extrapolation the position of the asteroid BEFORE IT WAS BORN!!!!11!
    new_size = a.size - 1
    new_mass = ASTEROID_MASS_LOOKUP[new_size]
    new_radius = ASTEROID_RADII_LOOKUP[new_size]

    if timesteps_until_appearance == 0:
        return [
            Asteroid(
                position=a.position,
                velocity=(v*cos(angle), v*sin(angle)),
                size=new_size,
                mass=new_mass,
                radius=new_radius,
                timesteps_until_appearance=0
            ) for angle in angles
        ]
    else:
        return [
            Asteroid(
                position=(
                    (a.position[0] + a.velocity[0]*DELTA_TIME*float(timesteps_until_appearance) - float(timesteps_until_appearance)*cos(angle)*v*DELTA_TIME) % game_state.map_size[0], 
                    (a.position[1] + a.velocity[1]*DELTA_TIME*float(timesteps_until_appearance) - float(timesteps_until_appearance)*sin(angle)*v*DELTA_TIME) % game_state.map_size[1]
                ),
                velocity=(v*cos(angle), v*sin(angle)),
                size=new_size,
                mass=new_mass,
                radius=new_radius,
                timesteps_until_appearance=timesteps_until_appearance
            ) for angle in angles
        ]


def maintain_forecasted_asteroids(forecasted_asteroid_splits: list[Asteroid], game_state: GameState) -> list[Asteroid]:
    # Maintain the list of projected split asteroids by advancing the position, decreasing the timestep, and facilitate removal
    updated_asteroids = [
        Asteroid(
            position=(
                (forecasted_asteroid.position[0] + forecasted_asteroid.velocity[0] * DELTA_TIME) % game_state.map_size[0],
                 (forecasted_asteroid.position[1] + forecasted_asteroid.velocity[1] * DELTA_TIME) % game_state.map_size[1]
            ),
            velocity=forecasted_asteroid.velocity,
            size=forecasted_asteroid.size,
            mass=forecasted_asteroid.mass,
            radius=forecasted_asteroid.radius,
            timesteps_until_appearance=forecasted_asteroid.timesteps_until_appearance - 1
        ) for forecasted_asteroid in forecasted_asteroid_splits if forecasted_asteroid.timesteps_until_appearance > 1
    ]
    return updated_asteroids

def is_asteroid_in_list(list_of_asteroids: list[Asteroid], asteroid: Asteroid, game_state: GameState) -> bool:
    # Since floating point comparison isn't a good idea, break apart the asteroid dict and compare each element manually in a fuzzy way
    for a in list_of_asteroids:
        # The reason we do the seemingly redundant checks for position, is that we need to account for wrap. If the game field was 1000 pixels wide, and one asteroid is at 0.0000000001 and the other is at 999.9999999999, they're basically the same asteroid, so we need to realize that.
        if (is_close(a.position[0], asteroid.position[0]) or is_close_to_zero(game_state.map_size[0] - abs(a.position[0] - asteroid.position[0]))) and (is_close(a.position[1], asteroid.position[1]) or is_close_to_zero(game_state.map_size[1] - abs(a.position[1] - asteroid.position[1]))) and is_close(a.velocity[0], asteroid.velocity[0]) and is_close(a.velocity[1], asteroid.velocity[1]) and a.size == asteroid.size:
            #assert is_close(a.position[0], asteroid.position[0]) and is_close(a.position[1], asteroid.position[1]) and a.velocity[0] == asteroid.velocity[0] and a.velocity[1] == asteroid.velocity[1] and a.size == asteroid.size
            return True
    return False


def count_asteroids_in_mine_blast_radius(game_state: GameState, mine_x: float, mine_y: float, future_check_timesteps: i64) -> i64:
    count = 0
    for a in game_state.asteroids:
        # Extrapolate the asteroid position into the time of the mine detonation to check its bounds
        asteroid_future_pos_x = (a.position[0] + float(future_check_timesteps)*a.velocity[0]*DELTA_TIME) % game_state.map_size[0]
        asteroid_future_pos_y = (a.position[1] + float(future_check_timesteps)*a.velocity[1]*DELTA_TIME) % game_state.map_size[1]
        #if check_collision(asteroid_future_pos_x, asteroid_future_pos_y, a.radius, mine_x, mine_y, MINE_BLAST_RADIUS - MINE_ASTEROID_COUNT_FUDGE_DISTANCE):
        delta_x = asteroid_future_pos_x - mine_x
        delta_y = asteroid_future_pos_y - mine_y
        separation = a.radius + (MINE_BLAST_RADIUS - MINE_ASTEROID_COUNT_FUDGE_DISTANCE)
        if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
            count += 1
    return count


def predict_ship_mine_collision(ship_pos_x: float, ship_pos_y: float, mine: Mine, future_timesteps: i64 = 0) -> float:
    # Predicts whether a ship staying where it currently is will be hit by a mine, and when
    if mine.remaining_time >= float(future_timesteps)*DELTA_TIME:
        #if check_collision(ship_pos_x, ship_pos_y, SHIP_RADIUS, mine.position[0], mine.position[1], MINE_BLAST_RADIUS):
        delta_x = ship_pos_x - mine.position[0]
        delta_y = ship_pos_y - mine.position[1]
        separation = SHIP_RADIUS + MINE_BLAST_RADIUS
        if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
            return mine.remaining_time
        else:
            return inf
    else:
        # This mine exploded in the past, so won't ever collide
        return inf


def calculate_timesteps_until_bullet_hits_asteroid(time_until_asteroid_center_s: float, asteroid_radius: float) -> i64:
    # time_until_asteroid_center is the time it takes for the bullet to travel from the center of the ship to the center of the asteroid
    # The bullet originates from the ship's edge, and the collision can happen as early as it touches the radius of the asteroid
    # We have to add 1, because it takes 1 timestep for the bullet to originate at the start, before it starts moving
    return 1 + ceil((time_until_asteroid_center_s*BULLET_SPEED - asteroid_radius - SHIP_RADIUS)/BULLET_SPEED*FPS)


def asteroid_bullet_collision(bullet_head_position: tuple[float, float], bullet_tail_position: tuple[float, float], asteroid_center: tuple[float, float], asteroid_radius: float) -> bool:
    # This is an optimized version of circle_line_collision() from the Kessler source code
    # First, do a rough check if there's no chance the collision can occur
    # Avoid the use of min/max because it should be a bit faster
    if bullet_head_position[0] < bullet_tail_position[0]:
        x_min = bullet_head_position[0] - asteroid_radius
        if asteroid_center[0] < x_min:
            return False
        x_max = bullet_tail_position[0] + asteroid_radius
    else:
        x_min = bullet_tail_position[0] - asteroid_radius
        if asteroid_center[0] < x_min:
            return False
        x_max = bullet_head_position[0] + asteroid_radius
    if asteroid_center[0] > x_max:
        return False

    if bullet_head_position[1] < bullet_tail_position[1]:
        y_min = bullet_head_position[1] - asteroid_radius
        if asteroid_center[1] < y_min:
            return False
        y_max = bullet_tail_position[1] + asteroid_radius
    else:
        y_min = bullet_tail_position[1] - asteroid_radius
        if asteroid_center[1] < y_min:
            return False
        y_max = bullet_head_position[1] + asteroid_radius
    if asteroid_center[1] > y_max:
        return False

    # A collision is possible.
    # Create a triangle between the center of the asteroid, and the two ends of the bullet.
    a = dist(bullet_head_position, asteroid_center)
    b = dist(bullet_tail_position, asteroid_center)
    #c = BULLET_LENGTH

    # Heron's formula to calculate area of triangle and resultant height (distance from circle center to line segment)
    s = 0.5*(a + b + BULLET_LENGTH)

    squared_area = s*(s - a)*(s - b)*(s - BULLET_LENGTH)
    #triangle_height = 2.0/c*sqrt(max(0.0, squared_area))
    triangle_height = TWICE_BULLET_LENGTH_RECIPROCAL*sqrt(max(0.0, squared_area))

    # If triangle's height is less than the asteroid's radius, the bullet is colliding with it
    return triangle_height < asteroid_radius


@lru_cache()  # This function gets called with the same params all the time, so just cache the return value the first time
def get_simulated_ship_max_range(max_cruise_seconds: float) -> tuple[float, i64]:
    # UNUSED
    dummy_game_state = GameState(
        asteroids=[],
        ships=[],
        bullets=[],
        mines=[],
        map_size=(100000.0, 100000.0),
        time=0.0,
        delta_time=DELTA_TIME,
        sim_frame=0
    )
    dummy_ship_state = Ship(
        speed=0.0,
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        heading=0.0,
        bullets_remaining=0,
        lives_remaining=1
    )
    max_ship_range_test = Matrix(dummy_game_state, dummy_ship_state, 0)
    max_ship_range_test.accelerate(SHIP_MAX_SPEED)
    max_ship_range_test.cruise(round(max_cruise_seconds*FPS))
    max_ship_range_test.accelerate(0)
    state_sequence = max_ship_range_test.get_state_sequence()
    ship_random_range = dist(state_sequence[0].ship_state.position, state_sequence[-1].ship_state.position)
    ship_random_max_maneuver_length = len(state_sequence)
    return ship_random_range, ship_random_max_maneuver_length


def simulate_ship_movement_with_inputs(game_state: GameState, ship_state: Ship, move_sequence: list[Action]) -> Ship:
    # UNUSED
    # TODO: Replace cast with GameState instantiation
    dummy_game_state = cast(GameState, {'asteroids': [], 'ships': [], 'bullets': [], 'mines': [], 'map_size': game_state.map_size, 'time': 0.0, 'delta_time': 1/30, 'sim_frame': 0})
    ship_movement_sim = Matrix(dummy_game_state, ship_state, 0, 0.0)
    ship_movement_sim.apply_move_sequence(move_sequence)
    return ship_movement_sim.get_ship_state()


def get_adversary_interception_time_lower_bound(asteroid: Asteroid, adversary_ships: list[Ship], game_state: GameState, adversary_rotation_timestep_fudge: i64 = ADVERSARY_ROTATION_TIMESTEP_FUDGE) -> float:
    if not adversary_ships:
        return inf
    # The interception time is just from firing to hitting. It doesn't include the aiming time!
    feasible, _, aiming_timesteps_required, interception_time_s, _, _, _ = solve_interception(asteroid, adversary_ships[0], game_state, 0)
    if feasible:
        return max(0.0, interception_time_s + float(aiming_timesteps_required - adversary_rotation_timestep_fudge)*DELTA_TIME)
    else:
        return inf


def solve_interception(asteroid: Asteroid, ship_state: Ship, game_state: GameState, timesteps_until_can_fire: i64 = 0) -> tuple[bool, float, i64, float, float, float, float]:
    # The bullet's head originates from the edge of the ship's radius.
    # We want to set the position of the bullet to the center of the bullet, so we have to do some fanciness here so that at t=0, the bullet's center is where it should be
    t_0 = 0.0175 # t_0 = (SHIP_RADIUS - 0.5*BULLET_LENGTH)/BULLET_SPEED
    # Positions are relative to the ship. We set the origin to the ship's position. Remember to translate back!
    asteroid_velocity: tuple[float, float] = asteroid.velocity
    asteroid_position: tuple[float, float] = asteroid.position
    ship_position: tuple[float, float] = ship_state.position
    origin_x: float = ship_position[0]
    origin_y: float = ship_position[1]
    avx: float = asteroid_velocity[0]
    avy: float = asteroid_velocity[1]
    ax: float = asteroid_position[0] - origin_x + avx*DELTA_TIME  # We project the asteroid one timestep ahead, since by the time we shoot our bullet, the asteroid would have moved one more timestep!
    ay: float = asteroid_position[1] - origin_y + avy*DELTA_TIME

    vb = BULLET_SPEED
    vb_sq = vb*vb
    theta_0 = radians(ship_state.heading)

    # Calculate constants for naive_desired_heading_calc
    a = avx*avx + avy*avy - vb_sq

    # Calculate constants for root_function, root_function_derivative, root_function_second_derivative
    k1 = ay*vb - avy*vb*t_0
    k2 = ax*vb - avx*vb*t_0
    k3 = avy*ax - avx*ay

    def naive_desired_heading_calc(timesteps_until_fire: i64 = 0) -> tuple[float, float, i64, float, float, float]:
        # Here's a good resource to learn about this: https://www.youtube.com/watch?v=MpUUsDDE1sI
        # https://medium.com/andys-coding-blog/ai-projectile-intercept-formula-for-gaming-without-trigonometry-37b70ef5718b
        time_until_can_fire_s = float(timesteps_until_fire)*DELTA_TIME
        ax_delayed = ax + time_until_can_fire_s*avx  # We add a delay to account for the timesteps until we fire delay
        ay_delayed = ay + time_until_can_fire_s*avy

        # a is calculated outside of this function since it's a constant
        b = 2.0*(ax_delayed*avx + ay_delayed*avy - vb_sq*t_0)
        c = ax_delayed*ax_delayed + ay_delayed*ay_delayed - vb_sq*t_0*t_0

        #solutions = []
        for t in solve_quadratic(a, b, c):
            if isnan(t) or t < 0.0:
                continue
            x = ax_delayed + t*avx
            y = ay_delayed + t*avy
            theta = fast_atan2(y, x)
            # If the asteroid is out of bounds, then it will wrap around and this shot isn't feasible
            # However, if an unwrapped asteroid was passed into this function and the interception is inbounds, then it's a feasible shot
            intercept_x = x + origin_x
            intercept_y = y + origin_y
            # Return the first answer we get
            return t, angle_difference_rad(theta, theta_0), timesteps_until_fire, intercept_x, intercept_y, dist(ship_position, (intercept_x, intercept_y))
        #if len(solutions) > 1:
        #    print(len(solutions))
        #return solutions
        return math.nan, math.nan, 0, math.nan, math.nan, math.nan
        # Returned tuple is (interception time in seconds from firing to hit, delta theta rad, timesteps until fire, None, intercept_x, intercept_y, dist)

    def naive_root_function(theta: float, time_until_can_fire_s: float = 0.0) -> float:
        # Can be optimized more by expanding out the terms
        # Convert heading error to absolute heading
        theta += theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        ax_delayed = ax + time_until_can_fire_s*avx  # We add a delay to account for the timesteps until we can fire delay
        ay_delayed = ay + time_until_can_fire_s*avy
        return (vb*cos_theta - avx)*(ay_delayed - vb*t_0*sin_theta) - (vb*sin_theta - avy)*(ax_delayed - vb*t_0*cos_theta)

    def naive_time_function(theta: float) -> float:
        # UNUSED
        # Convert heading error to absolute heading
        theta += theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        return (ax + ay - vb*t_0*(sin_theta + cos_theta))/(vb*(sin_theta + cos_theta) - avx - avy)

    def naive_time_function_for_plotting(theta: float) -> float:
        # Just scale up the number so that it fits on the same scale and will be visible after plotting
        return max(-200000.0, min(naive_time_function(theta)*100000.0, 200000.0))

    def root_function(theta: float) -> float:
        # Convert heading error to absolute heading
        theta += theta_0
        # Domain of this function is theta_0 - pi to theta_0 + pi
        # Make this function periodic by wrapping inputs outside this range, to within the range
        if not (theta_0 - pi <= theta <= theta_0 + pi):
            theta = (theta - theta_0 + pi) % TAU - pi + theta_0
        abs_delta_theta = abs(theta - theta_0)
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        sinusoidal_component = k1*cos_theta - k2*sin_theta + k3
        wacky_component = vb*abs_delta_theta/pi*(avy*cos_theta - avx*sin_theta)
        return sinusoidal_component + wacky_component

    def root_function_derivative(theta: float) -> float:
        # Convert heading error to absolute heading
        theta += theta_0
        # Domain of this function is theta_0 - pi to theta_0 + pi
        # Make this function periodic by wrapping inputs outside this range, to within the range
        if not (theta_0 - pi <= theta <= theta_0 + pi):
            theta = (theta - theta_0 + pi) % TAU - pi + theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        sinusoidal_component = -k1*sin_theta - k2*cos_theta
        wacky_component = -vb*sign(theta - theta_0)/pi*(avx*sin_theta - avy*cos_theta + (theta - theta_0)*(avx*cos_theta + avy*sin_theta))
        return sinusoidal_component + wacky_component

    def root_function_second_derivative(theta: float) -> float:
        # Convert heading error to absolute heading
        theta += theta_0
        # Domain of this function is theta_0 - pi to theta_0 + pi
        # Make this function periodic by wrapping inputs outside this range, to within the range
        if not (theta_0 - pi <= theta <= theta_0 + pi):
            theta = (theta - theta_0 + pi) % TAU - pi + theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        sinusoidal_component = -k1*cos_theta + k2*sin_theta
        wacky_component = -vb*sign(theta - theta_0)/pi*(2.0*(avx*cos_theta + avy*sin_theta) - (theta - theta_0)*(avx*sin_theta - avy*cos_theta))
        return sinusoidal_component + wacky_component

    # I inlined the functions, but you could add them back to the function call.
    def turbo_rootinator_5000(initial_guess: float, tolerance: float = EPS, max_iterations: i64 = 4) -> float:  # function: Callable[[float], float], derivative_function: Callable[[float], float], second_derivative_function: Callable[[float], float]
        theta_old: float = initial_guess
        # debug_print(f"Our initial guess is {initial_guess} which gives function value {function(initial_guess)}")
        initial_func_value: Optional[float] = None
        theta_new: float
        func_value: float
        for _ in range(max_iterations):
            func_value = root_function(theta_old)
            if abs(func_value) < TAD:
                return theta_old
            if not initial_func_value:
                initial_func_value = func_value
            derivative_value = root_function_derivative(theta_old)
            second_derivative_value = root_function_second_derivative(theta_old)

            # Update the estimate using Halley's method
            denominator = 2.0*derivative_value*derivative_value - func_value*second_derivative_value
            if denominator == 0.0:
                return nan
            theta_new = theta_old - (2.0*func_value*derivative_value)/denominator
            # The value has jumped past the periodic boundary. Clamp it to right past the boundary just so things don't get too crazy.
            if theta_new < -pi:
                theta_new = pi - GRAIN
            elif pi < theta_new:
                theta_new = -pi + GRAIN
            elif -pi <= theta_old <= 0.0 <= theta_new <= pi:
                # The value jumped past the kink in the middle of the graph. set it to right past the kink so the value doesn't jump around like crazy
                theta_new = GRAIN
            elif -pi <= theta_new <= 0.0 <= theta_old <= pi:
                theta_new = -GRAIN

            # debug_print(f"After iteration {iteration + 1}, our new theta value is {theta_new}. Func value is {func_value}")
            # Check for convergence
            # It converged if the theta value isn't changing much, and the function itself takes a value that is close to zero (magnitude at most 1% of the original func value)
            if abs(theta_new - theta_old) < tolerance and abs(func_value) < 0.1*abs(initial_func_value):
                return theta_new
            theta_old = theta_new
        return nan

    def rotation_time(delta_theta_rad: float) -> float:
        return abs(delta_theta_rad)*SHIP_MAX_TURN_RATE_RAD_RECIPROCAL

    def bullet_travel_time(theta: float, t_rot: float) -> float:
        # Convert heading error to absolute heading
        theta += theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        denominator_x = avx - vb*cos_theta
        denominator_y = avy - vb*sin_theta
        if denominator_x == 0.0 and denominator_y == 0.0:
            return inf
        # At least one of the denominators is nonzero, so if we choose the one with the larger magnitude, we'll avoid division by zero as well as get the more accurate answer
        if abs(denominator_x) > abs(denominator_y):
            t_bul = (vb*t_0*cos_theta - ax - avx*t_rot)/denominator_x
        else:
            t_bul = (vb*t_0*sin_theta - ay - avy*t_rot)/denominator_y
        return t_bul

    def bullet_travel_time_for_plot(theta: float) -> float:
        # USED FOR DEBUGGING
        # Convert heading error to absolute heading
        theta += theta_0
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        t_rot = rotation_time(theta - theta_0)

        denominator = avx*sin_theta - avy*cos_theta
        if denominator == 0.0:
            return inf
        else:
            return max(-200000.0, min(((cos_theta*(ay + avy*t_rot) - sin_theta*(ax + avx*t_rot))/denominator)*100000.0, 200000.0))

    def plot_function() -> None:
        # USED FOR DEBUGGING
        naive_theta_ans_list = [naive_desired_heading_calc(timesteps_until_can_fire)]  # Assuming this function returns a list of angles
        theta_0 = radians(ship_state.heading)
        # theta_range = linspace(theta_0 - pi, theta_0 + pi, 400)
        theta_delta_range = linspace(-pi, pi, 400)

        # Vectorize the functions for numpy compatibility
        vectorized_function = vectorize(root_function)
        vectorized_derivative = vectorize(root_function_derivative)
        vectorized_second_derivative = vectorize(root_function_second_derivative)
        vectorized_bullet_time = vectorize(bullet_travel_time_for_plot)
        vectorized_naive_function = vectorize(naive_root_function)
        vectorized_naive_time = vectorize(naive_time_function_for_plotting)

        # Calculate function values
        function_values = vectorized_function(theta_delta_range)
        derivative_values = vectorized_derivative(theta_delta_range)
        # alt_derivative_values = vectorized_second_derivative(theta_delta_range)
        bullet_times = vectorized_bullet_time(theta_delta_range)
        naive_function_values = vectorized_naive_function(theta_delta_range, float(timesteps_until_can_fire)*DELTA_TIME)
        naive_times = vectorized_naive_time(theta_delta_range)

        plt.figure(figsize=(12, 6))

        # Plot the function and its derivatives
        plt.plot(theta_delta_range, function_values, label="Function")
        plt.plot(theta_delta_range, derivative_values, label="Derivative", color="orange")
        # plt.plot(theta_delta_range, alt_derivative_values, label="Second Derivative", color="blue", linestyle=':')
        plt.plot(theta_delta_range, bullet_times, label="Bullet Time", color="green", linestyle='-')
        plt.plot(theta_delta_range, naive_function_values, label="Naive Function", color="magenta", linestyle='-')
        plt.plot(theta_delta_range, naive_times, label="Naive Times", color="purple", linestyle='-')

        # Add vertical lines for each naive_theta_ans
        fudge = 0
        for theta_ans in naive_theta_ans_list:
            plt.axvline(x=theta_ans[1] + fudge, color='yellow', linestyle='--', label=f"Naive Theta Ans at {theta_ans[1]:.2f}")

            zero = turbo_rootinator_5000(theta_ans[1] + fudge, TAD, 15)  # root_function, root_function_derivative, root_function_second_derivative
            if not isnan(zero):
                delta_theta_solution = zero
                if not (-pi <= delta_theta_solution <= pi):
                    # print(f"SOLUTION WAS OUT OUT BOUNDS AT {delta_theta_solution} AND WRAPPED TO -pi, pi")
                    delta_theta_solution = (delta_theta_solution + pi) % TAU - pi
                plt.axvline(x=delta_theta_solution, color='green', linestyle='--', label=f"Theta Ans Converged at {delta_theta_solution:.2f}")
            else:
                pass
                # print('Root finder gave up rip')

        # Add a horizontal line at y=0
        plt.axhline(y=0, color='black', linewidth=1.5, label="y=0")

        plt.xlabel("Theta")
        plt.ylabel("Values")
        plt.title("Function and Derivatives Plot")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    # plot_function()
    #valid_solutions = []
    # print_debug = False
    amount_we_can_turn_before_we_can_shoot_rad = float(timesteps_until_can_fire)*SHIP_MAX_TURN_RATE_RAD_TS
    naive_solution = naive_desired_heading_calc(timesteps_until_can_fire)
    # naive_solution is this tuple: (interception time in seconds from firing to hit, delta theta rad, timesteps until fire, None, intercept_x, intercept_y, None)
    # debug_print("Evaluating naive solution:", naive_solution)
    if abs(naive_solution[1]) <= amount_we_can_turn_before_we_can_shoot_rad + EPS:
        # The naive solution works because there's no turning delay
        # debug_print('Naive solution works!', naive_solution)
        if check_coordinate_bounds(game_state, naive_solution[3], naive_solution[4]):
            # Tuple is: (feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception)
            return True, degrees(naive_solution[1]), timesteps_until_can_fire, naive_solution[0], naive_solution[3], naive_solution[4], naive_solution[5]
    else:
        if abs(avx) < GRAIN and abs(avy) < GRAIN:
            # The asteroid is pretty much stationary. Naive solution works fine.
            # debug_print("The asteroid is pretty much stationary. Naive solution works fine.")
            delta_theta_solution = naive_solution[1]
        else:
            # Use more advanced solution
            # debug_print('Using more advanced root finder')
            delta_theta_solution = turbo_rootinator_5000(naive_solution[1], TAD, 4)  # root_function, root_function_derivative, root_function_second_derivative
            # debug_print('Root finder gave us:', sol)
        if isnan(delta_theta_solution):
            return False, nan, -1, nan, nan, nan, nan
        absolute_theta_solution = delta_theta_solution + theta_0
        # if not (-pi <= delta_theta_solution <= pi):
            # debug_print(f"SOLUTION WAS OUT OUT BOUNDS AT {delta_theta_solution} AND WRAPPED TO -pi, pi")
            # delta_theta_solution = (delta_theta_solution + pi)%TAU - pi
        # Check validity of solution to make sure time is positive and stuff
        delta_theta_solution_deg = degrees(delta_theta_solution)
        t_rot = rotation_time(delta_theta_solution)
        t_bullet = bullet_travel_time(delta_theta_solution, t_rot)
        # debug_print(f't_bullet: {t_bullet}')
        if t_bullet < 0:
            return False, nan, -1, nan, nan, nan, nan
        # t_total = t_rot + t_bullet

        bullet_travel_dist = vb*(t_bullet + t_0)
        intercept_x = origin_x + bullet_travel_dist*cos(absolute_theta_solution)
        intercept_y = origin_y + bullet_travel_dist*sin(absolute_theta_solution)
        # debug_print(f"Intercept_x ({intercept_x}) = origin_x ({origin_x}) + vb*cos({absolute_theta_solution})*(t_bullet ({t_bullet}) + t_0 ({t_0}))")
        # debug_print(f"Intercept_y ({intercept_y}) = origin_y ({origin_y}) + vb*sin({absolute_theta_solution})*(t_bullet ({t_bullet}) + t_0 ({t_0}))")

        if check_coordinate_bounds(game_state, intercept_x, intercept_y):
            # debug_print(f"The coordinates of {intercept_x}, {intercept_y} are GUCCI! We'd have to turn this many ts: {t_rot*FPS}")
            # Since half timesteps don't exist, we need to discretize this solution by rounding up the amount of timesteps, and now we can use the naive method to confirm and get the exact angle
            # We max this with ts until can fire, because that's the floor and we can't go below it
            t_rot_ts = max(timesteps_until_can_fire, ceil(t_rot*FPS))
            # debug_print(f"The rotation timesteps we've calculated is {t_rot_ts}, from a t_rot of {t_rot}")
            # valid_solutions.append((True, delta_theta_solution_deg, t_rot_ts, None, intercept_x, intercept_y, None))
            discrete_solution = naive_desired_heading_calc(t_rot_ts)
            if not isnan(discrete_solution[0]):
                if not abs(degrees(discrete_solution[1])) - EPS <= float(t_rot_ts)*SHIP_MAX_TURN_RATE_DEG_TS:
                    return False, nan, -1, nan, nan, nan, nan
                if check_coordinate_bounds(game_state, discrete_solution[3], discrete_solution[4]):
                    # debug_print('Valid solution found!', disc_sol)
                    return True, degrees(discrete_solution[1]), t_rot_ts, discrete_solution[0], discrete_solution[3], discrete_solution[4], discrete_solution[5]
    
    return False, nan, -1, nan, nan, nan, nan
    # The returned interception time is the time AFTER FIRING! NOT INCLUDING TURNING!
    # return (feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception)


def track_asteroid_we_shot_at(asteroids_pending_death: dict[i64, list[Asteroid]], current_timestep: i64, game_state: GameState, bullet_travel_timesteps: i64, original_asteroid: Asteroid) -> None:
    #global asteroid_tracking_total_time
    #start_time = time.perf_counter()
    # This modifies asteroids_pending_death in place instead of returning it
    # Make a copy of the asteroid so we don't mess up the original object
    asteroid = original_asteroid.copy()
    # Wrap asteroid position
    asteroid.position = (asteroid.position[0] % game_state.map_size[0], asteroid.position[1] % game_state.map_size[1])
    # Project the asteroid into the future, to where it would be on the timestep of its death

    for future_timesteps in range(0, bullet_travel_timesteps + 1):
        timestep = current_timestep + future_timesteps
        if timestep not in asteroids_pending_death:
            asteroids_pending_death[timestep] = [asteroid.copy()]
        else:
            asteroids_pending_death[timestep].append(asteroid.copy())
        # Advance the asteroid to the next position
        if future_timesteps != bullet_travel_timesteps:
            # Skip this operation on the last for loop iteration
            asteroid.position = ((asteroid.position[0] + asteroid.velocity[0]*DELTA_TIME) % game_state.map_size[0], (asteroid.position[1] + asteroid.velocity[1]*DELTA_TIME) % game_state.map_size[1])
    #asteroid_tracking_total_time += time.perf_counter() - start_time


def check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(asteroids_pending_death: dict[i64, list[Asteroid]], current_timestep: i64, game_state: GameState, asteroid: Asteroid) -> bool:
    # global asteroid_new_track_total_time
    #start_time = time.perf_counter()
    # This assumes all asteroids are wrapped/within the game bounds!
    # print(f"Checking pending shots for timestep {current_timestep}")

    # Check whether the asteroid has already been shot at, or if we can shoot at it again
    #asteroid = asteroid.copy()
    #asteroid.position = (asteroid.position[0] % game_state.map_size[0], asteroid.position[1] % game_state.map_size[1])
    if current_timestep in asteroids_pending_death:
        # return_value = not is_asteroid_in_list(asteroids_pending_death[current_timestep], asteroid, game_state)
        #asteroid_new_track_total_time += time.perf_counter() - start_time
        return not is_asteroid_in_list(asteroids_pending_death[current_timestep], asteroid, game_state)
    else:
        #asteroid_new_track_total_time += time.perf_counter() - start_time
        return True


def time_travel_asteroid(asteroid: Asteroid, timesteps: i64, game_state: GameState) -> Asteroid:
    '''Project an asteroid into the future or past. This automatically wraps the asteroid's position'''
    # The ts=0 shortcut isn't good because it doesn't wrap the asteroid!
    #if timesteps == 0:
        #print("WARNING: Time travelling asteroid by 0 timesteps! If this is rare, it's no cause for concern")
        #return asteroid.copy()
    return Asteroid(
        position=((asteroid.position[0] + float(timesteps)*asteroid.velocity[0]*DELTA_TIME) % game_state.map_size[0], (asteroid.position[1] + float(timesteps)*asteroid.velocity[1]*DELTA_TIME) % game_state.map_size[1]),
        velocity=asteroid.velocity,
        size=asteroid.size,
        mass=asteroid.mass,
        radius=asteroid.radius,
        timesteps_until_appearance=asteroid.timesteps_until_appearance
    )


def time_travel_asteroid_s(asteroid: Asteroid, time: float, game_state: GameState) -> Asteroid:
    '''Project an asteroid into the future or past. This automatically wraps the asteroid's position'''
    # The ts=0 shortcut isn't good because it doesn't wrap the asteroid!
    #if timesteps == 0:
        #print("WARNING: Time travelling asteroid by 0 timesteps! If this is rare, it's no cause for concern")
        #return asteroid.copy()
    return Asteroid(
        position=((asteroid.position[0] + time*asteroid.velocity[0]) % game_state.map_size[0], (asteroid.position[1] + time*asteroid.velocity[1]) % game_state.map_size[1]),
        velocity=asteroid.velocity,
        size=asteroid.size,
        mass=asteroid.mass,
        radius=asteroid.radius,
        timesteps_until_appearance=asteroid.timesteps_until_appearance
    )


class Matrix():
    # Simulates kessler_game.py and ship.py and other game mechanics
    # Has built-in controllers to do stationary targeting, maneuvers, and respawn maneuvers
    # Has fitness function to evaluate the set of moves, along with the end state
    # Can then extract the maneuver along with the future state, to execute the maneuver, and begin planning based on the future state before we actually get there
    def __init__(self, game_state: GameState, ship_state: Ship, initial_timestep: i64, respawn_timer: float = 0.0, asteroids_pending_death: Optional[dict[i64, list[Asteroid]]] = None, forecasted_asteroid_splits: Optional[list[Asteroid]] = None, last_timestep_fired: i64 = INT_NEG_INF, last_timestep_mined: i64 = INT_NEG_INF, mine_positions_placed: Optional[set[tuple[float, float]]] = None, halt_shooting: bool = False, fire_first_timestep: bool = False, verify_first_shot: bool = False, verify_maneuver_shots: bool = True, last_timestep_colliding: i64 = -1, game_state_plotter: Optional[GameStatePlotter] = None) -> None:
        #if not is_close_to_zero(ship_state.speed):
        #    print(f"WARNING: The ship speed when starting the sim is not zero! It's {ship_state.speed=}, {ship_state.velocity=}")
        if asteroids_pending_death is None:
            asteroids_pending_death = {}  # Keys are timesteps, and the values are the asteroids that still have midair bullets travelling toward them, so we don't want to shoot at them again
        if forecasted_asteroid_splits is None:
            forecasted_asteroid_splits = []
        self.initial_timestep = initial_timestep
        self.future_timesteps: i64 = 0
        self.last_timestep_fired = last_timestep_fired
        self.last_timestep_mined = last_timestep_mined
        self.game_state: GameState = game_state.copy()
        self.ship_state: Ship = ship_state.copy()
        self.game_state.asteroids = [a.copy() for a in game_state.asteroids]
        self.game_state.ships = [s.copy() for s in game_state.ships]
        self.game_state.bullets = [b.copy() for b in game_state.bullets]
        self.game_state.mines = [m.copy() for m in game_state.mines]
        self.other_ships = get_other_ships(self.game_state, ship_state.id)
        self.ship_move_sequence: list[Action] = []
        self.state_sequence: list[SimState] = []
        self.asteroids_shot: i64 = 0
        self.asteroids_pending_death: dict[i64, list[Asteroid]] = {timestep: list(l) for timestep, l in asteroids_pending_death.items()}
        self.forecasted_asteroid_splits: list[Asteroid] = [a.copy() for a in forecasted_asteroid_splits]
        self.halt_shooting: bool = halt_shooting # This probably means we're doing a respawn maneuver
        self.fire_next_timestep_flag: bool = False
        self.fire_first_timestep: bool = fire_first_timestep
        self.game_state_plotter: Optional[GameStatePlotter] = game_state_plotter
        self.sim_id = random.randint(1, 100000)
        #if self.sim_id == 333:
        #    print(f"Starting sim 333 with ship state {ship_state}")
        self.explanation_messages: list[str] = []
        self.safety_messages: list[str] = []
        self.respawn_timer: float = respawn_timer
        self.plot_this_sim = False #( and GAMESTATE_PLOTTING)# or (self.sim_id in [2238])
        self.ship_crashed = False
        self.backed_up_game_state_before_post_mutation: Optional[GameState] = None
        self.fitness_breakdown: Optional[tuple[float, float, float, float, float, float, float, float, float]] = None
        self.cancel_firing_first_timestep: bool = False
        self.verify_first_shot: bool = verify_first_shot
        self.intended_move_sequence: list[Action] = []
        self.sim_placed_a_mine: bool = False
        self.verify_maneuver_shots: bool = verify_maneuver_shots
        self.mine_positions_placed = mine_positions_placed if mine_positions_placed is not None else set()
        # This is to facilitate my two-pass respawn maneuvers. The first pass doesn't shoot, and records when we will no longer hit asteroids. The second pass will begin targeting after the ship is clear from asteroids, since after shooting the respawn invincibility will be gone
        self.last_timestep_colliding: i64 = last_timestep_colliding if last_timestep_colliding != -1 else self.initial_timestep - 1
        # 0 - Not a respawn maneuver, 1 - First pass of respawn maneuver, 2 - Second pass of respawn maneuver
        self.respawn_maneuver_pass_number: i64 = 0 if (not self.halt_shooting and last_timestep_colliding == -1) else (1 if last_timestep_colliding == -1 else 2)

    def get_last_timestep_colliding(self) -> i64:
        return self.last_timestep_colliding

    def get_mine_positions_placed(self) -> set[tuple[float, float]]:
        return self.mine_positions_placed

    def get_cancel_firing_first_timestep(self) -> bool:
        return self.cancel_firing_first_timestep

    def get_explanations(self) -> list[str]:
        return self.explanation_messages

    def get_safety_messages(self) -> list[str]:
        return self.safety_messages

    def get_sim_id(self) -> i64:
        return self.sim_id

    def get_respawn_timer(self) -> float:
        return self.respawn_timer

    def get_ship_state(self) -> Ship:
        return self.ship_state.copy()

    def get_game_state(self) -> GameState:
        if self.backed_up_game_state_before_post_mutation is not None:
            # In the process of waiting out mines, the game state got messed up so we have to use a backed-up copy
            return self.backed_up_game_state_before_post_mutation.copy()
        else:
            return self.game_state.copy()

    def get_fire_next_timestep_flag(self) -> bool:
        return self.fire_next_timestep_flag

    def set_fire_next_timestep_flag(self, fire_next_timestep_flag: bool) -> None:
        self.fire_next_timestep_flag = fire_next_timestep_flag

    def get_asteroids_pending_death(self) -> dict[i64, list[Asteroid]]:
        return self.asteroids_pending_death

    def get_forecasted_asteroid_splits(self) -> list[Asteroid]:
        return self.forecasted_asteroid_splits

    def get_instantaneous_asteroid_collision(self, asteroids: Optional[list[Asteroid]] = None, ship_position: Optional[tuple[float, float]] = None) -> bool:
        # UNUSED
        if ship_position is not None:
            position = ship_position
        else:
            position = self.ship_state.position

        for a in (asteroids if asteroids is not None else self.game_state.asteroids):
            if check_collision(position[0], position[1], SHIP_RADIUS, a.position[0], a.position[1], a.radius):
                return True
        return False

    def get_instantaneous_ship_collision(self) -> bool:
        # UNUSED. This is too inaccurate, and there's better ways to handle avoiding the other ship rather than being overconfident in giving a binary yes you will collide/no you will not collide
        for ship in self.other_ships:
            # The faster the other ship is going, the bigger of a bubble around it I'm going to draw, since they can deviate from their path very quickly and run into me even though I thought I was in the clear
            if check_collision(self.ship_state.position[0], self.ship_state.position[1], SHIP_RADIUS, ship.position[0], ship.position[1], ship.radius + SHIP_AVOIDANCE_PADDING + sqrt(ship.velocity[0]**2 + ship.velocity[1]**2)*SHIP_AVOIDANCE_SPEED_PADDING_RATIO):
                return True
        return False

    def get_instantaneous_mine_collision(self) -> bool:
        # UNUSED
        mine_collision = False
        mine_remove_idxs = []
        for i, m in enumerate(self.game_state.mines):
            if m.remaining_time < EPS:
                if check_collision(self.ship_state.position[0], self.ship_state.position[1], SHIP_RADIUS, m.position[0], m.position[1], MINE_BLAST_RADIUS):
                    mine_collision = True
                mine_remove_idxs.append(i)
        if mine_remove_idxs:
            self.game_state.mines = [mine for idx, mine in enumerate(self.game_state.mines) if idx not in mine_remove_idxs]
        return mine_collision

    def get_next_extrapolated_asteroid_collision_time(self, additional_timesteps_to_blow_up_mines: i64 = 0) -> float:
        # debug_print(f"Inside get fitness, we shot {self.asteroids_shot} asteroids. getting extrapolated collision time. The ship's velocity is: ({self.ship_state.velocity[0]}, {self.ship_state.velocity[1]})")
        # Assume constant velocity from here
        next_imminent_asteroid_collision_time = inf
        # print('Extrapolating stuff at rest in end')
        # The asteroids from the game state could have been from the future since we waited out the mines, but the forecasted splits are from present time, so we need to treat them differently and only back-extrapolate the existing asteroids and not the forecasted ones
        # print(f"Forecasted splits: {self.forecasted_asteroid_splits}")
        for ast_idx, asteroid in enumerate(chain(self.game_state.asteroids, self.forecasted_asteroid_splits)):
            asteroid_is_born: bool = ast_idx < len(self.game_state.asteroids)
            # print(f"Ast is born: {asteroid_is_born}")
            # print(f"Checking collision with asteroid: {ast_to_string(asteroid)} on timestep {self.initial_timestep + self.future_timesteps}")
            # debug_print(f"Future timesteps: {self.future_timesteps}, timesteps to not check collision for: {self.timesteps_to_not_check_collision_for}")
            for a in unwrap_asteroid(asteroid, self.game_state.map_size[0], self.game_state.map_size[1], UNWRAP_ASTEROID_COLLISION_FORECAST_TIME_HORIZON, False):
                # if self.future_timesteps >= self.timesteps_to_not_check_collision_for:

                predicted_collision_time_from_future = predict_next_imminent_collision_time_with_asteroid(self.ship_state.position[0], self.ship_state.position[1], self.ship_state.velocity[0], self.ship_state.velocity[1], SHIP_RADIUS, a.position[0], a.position[1], a.velocity[0], a.velocity[1], a.radius)
                predicted_collision_time = predicted_collision_time_from_future + (DELTA_TIME*float(additional_timesteps_to_blow_up_mines) if asteroid_is_born else 0.0)
                
                if isinf(predicted_collision_time):
                    continue
                # The predicted collision time is finite and after the end of the sim
                # TODO: Verify there isn't an off by one error
                if not (asteroid.timesteps_until_appearance > 0 and float(asteroid.timesteps_until_appearance)*DELTA_TIME > predicted_collision_time + EPS):
                    # The asteroid either exists, or will come into existence before our collision time
                    # Check the canonical asteroid, and not the unwrapped one!
                    # print(f"{additional_timesteps_to_blow_up_mines=}")
                    if asteroid_is_born and additional_timesteps_to_blow_up_mines != 0:
                        ast_to_check = time_travel_asteroid(asteroid, -additional_timesteps_to_blow_up_mines, self.game_state)
                    else:
                        ast_to_check = asteroid
                    if not check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, ast_to_check):
                        # We're already shooting the asteroid. Check whether the imminent collision time is before or after the asteroid is eliminated
                        predicted_collision_ts = floor(predicted_collision_time*FPS)
                        if asteroid_is_born:
                            future_asteroid_during_imminent_collision_time = time_travel_asteroid(a, predicted_collision_ts - additional_timesteps_to_blow_up_mines, self.game_state)
                        else:
                            future_asteroid_during_imminent_collision_time = time_travel_asteroid(a, predicted_collision_ts, self.game_state)
                        # if self.sim_id == 13974:
                        # print(f"Getting extrapolated next coll time for unwrapped asteroid {ast_to_string(a)}. Initial TS: {self.initial_timestep}, future TS: {self.future_timesteps}, additional timesteps to blow up mines: {additional_timesteps_to_blow_up_mines}, predicted collision additional ts: {predicted_collision_ts}")
                        # print(self.asteroids_pending_death)
                        if check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + predicted_collision_ts, self.game_state, future_asteroid_during_imminent_collision_time):
                            # In the future time the asteroid has already been eliminated, so there won't actually be a collision
                            # debug_print("In the future time the asteroid has already been eliminated, so there won't actually be a collision")
                            continue
                        else:
                            next_imminent_asteroid_collision_time = min(next_imminent_asteroid_collision_time, predicted_collision_time)
                    else:
                        # We're not eliminating this asteroid, and if it was forecasted, it comes into existence before our collision time. Therefore our collision is real and should be considered.
                        next_imminent_asteroid_collision_time = min(next_imminent_asteroid_collision_time, predicted_collision_time)
                #else:
                    # There is no collision since the asteroid is born after our predicted collision time, and an unborn asteroid can't collide with anything
                    # debug_print("There is no collision since the unborn asteroid can't collide with anything")
                #    pass
            # else:
            #    debug_print(f"Inside extrapolated coll time checker. We already have a pending shot for this so we'll ignore this asteroid: {ast_to_string(asteroid)}")
        return next_imminent_asteroid_collision_time

    def get_next_extrapolated_mine_collision_times_and_pos(self) -> list[tuple[float, tuple[float, float]]]:
        times_and_mine_pos = []
        for m in self.game_state.mines:
            # print(f"{self.ship_state.velocity=}")
            next_imminent_mine_collision_time = predict_ship_mine_collision(self.ship_state.position[0], self.ship_state.position[1], m, 0)
            if not isinf(next_imminent_mine_collision_time):
                times_and_mine_pos.append((next_imminent_mine_collision_time, m.position))
        return times_and_mine_pos

    def coordinates_in_same_wrap(self, pos1: tuple[float, float], pos2: tuple[float, float]) -> bool:
        # UNUSED
        # Checks whether the coordinates are in the same universe
        max_width = self.game_state.map_size[0]
        max_height = self.game_state.map_size[1]
        # TODO: Honestly this is kinda sketch and using modulo is probably more accurate.
        return bool(pos1[0]//max_width == pos2[0]//max_width and pos1[1]//max_height == pos2[1]//max_height)

    def get_fitness(self) -> float:
        if self.fitness_breakdown:
            raise Exception("Do not call get_fitness twice!")
        # This is meant to be the last method called from this class. This is rather destructive!
        # print(f"Getting fitness from timestep {self.initial_timestep=} {self.future_timesteps=} in sim id {self.sim_id}")
        # This will return a scalar number representing how good of an action/state sequence we just went through
        # If these moves will keep us alive for a long time and shoot many asteroids along the way, then the fitness is good
        # If these moves result in us getting into a dangerous spot, or if we don't shoot many asteroids at all, then the fitness will be bad
        # The HIGHER the fitness score, the BETTER!
        # This fitness function is unified to consider all three types of moves. Stationary targeting, maneuvers, and respawn maneuvers.

        def get_asteroid_safe_time_fitness(next_extrapolated_asteroid_collision_time: float, displacement: float, move_sequence_length_s: float) -> float:
            # NOTE THAT the move sequence length is discounted, because if we're deciding between maneuvers, we really mostly care about how long you're safe for after the maneuver is done!
            if displacement < EPS:
                # Stationary
                # TODO: See whether there's something wrong with counting the move sequence length in this. It might be fine to count it all, or maybe use a discount factor
                # Only has to be safe for 4 seconds to get the max score, to encourage staying put and eliminating threats by shooting rather than by maneuvering and missing shot opportunities
                # asteroid_safe_time_fitness = max(0, min(5, 5 - 5/4*next_extrapolated_asteroid_collision_time)) # Goes through (0, 0) (4, 1) SCRATCH THIS
                asteroid_safe_time_fitness = sigmoid(next_extrapolated_asteroid_collision_time + move_sequence_length_s*0.25, 1.4, 3.0)  # sigmoid(next_extrapolated_asteroid_collision_time, 2.1*4/5, 2.5)
            else:
                # Maneuvering
                # asteroid_safe_time_fitness = max(0, min(5, 5 - next_extrapolated_asteroid_collision_time)) # Goes through (0, 0) (5, 1) SCRATCH THIS
                asteroid_safe_time_fitness = sigmoid(next_extrapolated_asteroid_collision_time + move_sequence_length_s*0.25, 1.4, 3.0)
            return asteroid_safe_time_fitness

        def get_asteroid_shot_frequency_fitness(asteroids_shot: i64, move_sequence_length_s: float) -> float:
            # How many asteroids did we shoot? The more the better.
            if asteroids_shot < 0:
                # This is to signal that we won't hit anything ever if we're staying here, so we should defer to the maneuver subcontroller to force a move
                # This is less effective with this new fitness system, but it should still work eventually.
                # debug_print(f"Deferring to maneuver subcontroller! Forcing a move.")
                return -0.9
            else:
                fudged_asteroids_shot: float
                if asteroids_shot == 0:
                    fudged_asteroids_shot = 0.1  # Avoid division by zero
                else:
                    fudged_asteroids_shot = float(asteroids_shot)
                time_per_asteroids_shot = move_sequence_length_s/fudged_asteroids_shot

                # Applying the sigmoid function to smooth the transition
                asteroids_fitness = sigmoid(time_per_asteroids_shot, -0.5*FPS, 10.8*DELTA_TIME)
            return asteroids_fitness

        def get_mine_safety_fitness(next_extrapolated_mine_collision_times: list[tuple[float, tuple[float, float]]]) -> tuple[float, float]:
            # If there's no mine in the final ship position's range, the fitness is perfect.
            # For each additional mine within the range, the fitness will go down.
            # Having two mines which are freshly placed isn't as bad as a single mine that's about to blow up.
            # For each mine, as the time goes down, the danger should go up more than linearly
            if not next_extrapolated_mine_collision_times:
                return 1.0, inf
            # Regardless of stationary or maneuvering, the mine safe time score is calculated the same way
            mines_threat_level = 0.0
            next_extrapolated_mine_collision_time = inf
            if next_extrapolated_mine_collision_times:
                for mine_collision_time, mine_pos in next_extrapolated_mine_collision_times:
                    next_extrapolated_mine_collision_time = min(next_extrapolated_mine_collision_time, mine_collision_time)
                    # next_extrapolated_mine_collision_time = max(0, min(3, next_extrapolated_mine_collision_time))
                    dist_to_ground_zero = dist(self.ship_state.position, mine_pos)
                    # This is a linear function that is maximum when I'm right over the mine, and minimum at 0 when I'm just touching the blast radius of it
                    # This will penalize being at ground zero more than penalizing being right at the edge of the blast, where it's easier to get out
                    mine_ground_zero_fudge = linear(dist_to_ground_zero, (0.0, 1.0), (MINE_BLAST_RADIUS + SHIP_RADIUS, 0.6))
                    # mine_ground_zero_fudge = max(0.0, (MINE_BLAST_RADIUS + SHIP_RADIUS - dist_to_ground_zero)/(MINE_BLAST_RADIUS + SHIP_RADIUS))
                    mines_threat_level += (MINE_FUSE_TIME - next_extrapolated_mine_collision_time)**2.0/9.0*mine_ground_zero_fudge
            mine_safe_time_fitness = sigmoid(mines_threat_level, -6.8, 0.232)
            return mine_safe_time_fitness, next_extrapolated_mine_collision_time

        def get_asteroid_aiming_cone_fitness() -> float:
            # Iterate over all asteroids and get their heading angle from the ship's final position/heading, and see whether it's within +-30 degrees
            #ship_heading = self.ship_state.heading
            ship_pos_x, ship_pos_y = self.ship_state.position
            ship_heading_rad = radians(self.ship_state.heading)
            asts_within_cone = 0
            for a in chain(self.game_state.asteroids, self.forecasted_asteroid_splits):
                # Actually for performance reasons, I think I won’t even check that the asteroid is one we don’t have a pending shot for.
                # My logic is that, well first of all, this is a heuristic.
                # But also, in what situation would I have already shot at those asteroids?
                # If I shot at them way in the past, they would already be dead, and we won’t have a pending shot.
                # We’d only have a pending shot if we JUST shot them.
                # But say we have case A where I do a maneuver and don’t shoot the asteroids.
                # Is that really better than a case where I did a maneuver, but I shot the asteroids and now there’s no asteroids in the cone in front of me? The latter is preferable.
                #if check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, a):
                
                if heading_diff_within_threshold(ship_heading_rad, a.position[0] - ship_pos_x, a.position[1] - ship_pos_y, AIMING_CONE_FITNESS_CONE_WIDTH_HALF_COSINE):
                    asts_within_cone += 1
                #theta = degrees(super_fast_atan2(a.position[1] - ship_pos[1], a.position[0] - ship_pos[0]))
                #if abs(angle_difference_deg(theta, ship_heading)) <= 30.0:
                #    asts_within_cone += 1
                #if abs(ship_heading - theta) <= 30.0 or abs(360 - abs(ship_heading - theta)) <= 30.0:
                #    asts_within_cone += 1
            '''
            if asts_within_cone == 0:
                asteroid_aiming_cone_score = 0.0
            elif asts_within_cone == 1:
                asteroid_aiming_cone_score = 0.8
            elif asts_within_cone == 2:
                asteroid_aiming_cone_score = 0.85
            elif asts_within_cone == 3:
                asteroid_aiming_cone_score = 0.90
            elif asts_within_cone == 4:
                asteroid_aiming_cone_score = 0.95
            else:
                asteroid_aiming_cone_score = 1.0
            '''
            return sigmoid(asts_within_cone, 1.0, 2.4)

        def get_crash_fitness() -> float:
            if self.ship_crashed:
                crash_fitness = 0.0
            else:
                crash_fitness = 1.0
            return crash_fitness

        def get_sequence_length_fitness(move_sequence_length_s: float, displacement: float) -> float:
            if self.respawn_maneuver_pass_number > 0:
                # This is a respawn maneuver
                return sigmoid(move_sequence_length_s, -2.8, 1.7)
            else:
                if displacement < EPS:
                    # If this is stationary targetting, we give an incentive to do that by discounting the length it takes to aim
                    # move_sequence_length_s = min(0.25, 0.5*move_sequence_length_s)
                    # pass
                    return sigmoid(move_sequence_length_s, -2.8, 1.7)
                else:
                    return sigmoid(move_sequence_length_s, -5.7, 0.8)

        def get_other_ship_proximity_fitness(self_positions: list[tuple[float, float]]) -> float:
            # Penalize being too close to the other ship. If the other ship is moving, penalize that as well by effectively treating the distance as closer to the other ship
            # There is no maximum detection distance.
            # On a 1000x800 board, the max distance between two ships is 640.3 pixels
            for other_ship in self.other_ships:
                # It's assume there's only one ship, so this returns after the first ship is checked
                other_ship_pos_x, other_ship_pos_y = other_ship.position
                other_ship_vel_x, other_ship_vel_y = other_ship.velocity
                other_ship_speed = sqrt(other_ship_vel_x*other_ship_vel_x + other_ship_vel_y*other_ship_vel_y)
                # prox_score_speed_mul = min(1, other_ship_speed/100)*0.7 + 0.3
                # This multiplier effectively decreases the distance between the ships, at least when the fitness function considers it
                # If the other ship is stationary, then the multiplier is 1. But if the other ship is moving quickly, I treat the distance between us to be smaller than it actually is, down to 30% the actual distance
                other_ship_speed_dist_mul = linear(other_ship_speed, (0, 1), (SHIP_MAX_SPEED, 0.3))
                #total_separation_dist = 0.0
                separation_dists = []
                for self_pos in self_positions:
                    self_pos_x, self_pos_y = self_pos  # self.ship_state.position
                    # Account for wrap. We know that the farthest two objects can be separated within the screen in the x and y axes, is by half the width, and half the height
                    abs_sep_x = abs(self_pos_x - other_ship_pos_x)
                    abs_sep_y = abs(self_pos_y - other_ship_pos_y)
                    sep_x = min(abs_sep_x, self.game_state.map_size[0] - abs_sep_x)
                    sep_y = min(abs_sep_y, self.game_state.map_size[1] - abs_sep_y)
                    separation_dist = sqrt(sep_x*sep_x + sep_y*sep_y)
                    #total_separation_dist += separation_dist
                    separation_dists.append(separation_dist)
                # TODO: Maybe weigh the more recent separation distances more than the earlier ones
                #mean_separation_dist = total_separation_dist/len(self_positions)
                mean_separation_dist = weighted_harmonic_mean(separation_dists)
                #print(f"{self.sim_id=} {separation_dists=} {mean_separation_dist=}")
                # other_ship_proximity_fitness += prox_score_speed_mul*ship_proximity_max_penalty/(ship_proximity_detection_radius**ship_prox_exponent)*(ship_proximity_detection_radius - separation_dist)**ship_prox_exponent
                return sigmoid(mean_separation_dist*other_ship_speed_dist_mul, 0.032, 120)
            else:
                return 1.0

        # Grab the states first before they get screwed up
        states = self.get_state_sequence()

        move_sequence_length_s = float(self.get_sequence_length() - 1)*DELTA_TIME
        # print(f"{self.ship_state.position=}, {self.ship_state.velocity=}, {self.ship_state.speed=} maneuver length: {move_sequence_length_s}")
        next_extrapolated_mine_collision_times = self.get_next_extrapolated_mine_collision_times_and_pos()

        mine_safe_time_fitness, next_extrapolated_mine_collision_time = get_mine_safety_fitness(next_extrapolated_mine_collision_times)
        asteroids_fitness = get_asteroid_shot_frequency_fitness(self.asteroids_shot, move_sequence_length_s)
        asteroid_aiming_cone_fitness = get_asteroid_aiming_cone_fitness()

        # If mines still have yet to blow up, what we're gonna do is simulate the game until the mines blow up and we know how the mines will blast the asteroids
        # That way, we can accurately tell what the next extrapolated asteroid collision time is
        additional_timesteps_to_blow_up_mines = 0
        next_extrapolated_asteroid_collision_time = self.get_next_extrapolated_asteroid_collision_time()
        if self.game_state.mines:
            # Alternative to doing a deepcopy. Selectively copy the asteroids, mines, and bullets.
            self.backed_up_game_state_before_post_mutation = self.game_state.copy()
            self.backed_up_game_state_before_post_mutation.asteroids = [a.copy() for a in self.game_state.asteroids]
            self.backed_up_game_state_before_post_mutation.mines = [m.copy() for m in self.game_state.mines]
            self.backed_up_game_state_before_post_mutation.bullets = [b.copy() for b in self.game_state.bullets]
            while self.game_state.mines:
                additional_timesteps_to_blow_up_mines += 1
                # print(f"Calling update from get fitness to wait out mines:")
                self.update(0.0, 0.0, False, None, True)
        # if additional_timesteps_to_blow_up_mines != 0:
        #    debug_print(f"In get fitness, waited an additional {additional_timesteps_to_blow_up_mines} timesteps to blow up mines!")
        if additional_timesteps_to_blow_up_mines == 0:
            # No mines exist, and it's a straightforward prediction for asteroid collisions
            safe_time_after_maneuver_s = min(next_extrapolated_asteroid_collision_time, next_extrapolated_mine_collision_time)
        else:
            # Because mines existed and blew up which changed the trajectory of asteroids, we need to do a prediction before and after the mine blew up, and predict both paths
            # This doesn't properly account for multiple mines! But that's alright, I don't want to make this too complicated and have to do an asteroids prediction after every mine blows up and split the timeline into intervals where asteroids change trajectories, although that is possible to implement
            next_extrapolated_asteroid_collision_time_after_mines = self.get_next_extrapolated_asteroid_collision_time(additional_timesteps_to_blow_up_mines)
            if not isinf(next_extrapolated_asteroid_collision_time) and next_extrapolated_asteroid_collision_time <= additional_timesteps_to_blow_up_mines*DELTA_TIME:
                # Before the mine blows up, the ship will get hit by an asteroid if we stay here
                safe_time_after_maneuver_s = min(next_extrapolated_asteroid_collision_time, next_extrapolated_asteroid_collision_time_after_mines, next_extrapolated_mine_collision_time)
                # This assertion is false because it doesn't take into account the existence of other mines. If there was a mine about to blow up, and another fresh mine, then the first mine can blow me up before an asteroid or the fresh mine hits me
                #assert is_close(safe_time_after_maneuver_s, next_extrapolated_asteroid_collision_time), f"{safe_time_after_maneuver_s=} != {next_extrapolated_asteroid_collision_time=}, and {next_extrapolated_asteroid_collision_time_after_mines=}, {next_extrapolated_mine_collision_time=}, {additional_timesteps_to_blow_up_mines*DELTA_TIME=}"
            else:
                safe_time_after_maneuver_s = min(next_extrapolated_asteroid_collision_time_after_mines, next_extrapolated_mine_collision_time)
        # if not isinf(next_extrapolated_asteroid_collision_time_before_mines_blew_up):
        #    print(f"Next extrap time before mines: {next_extrapolated_asteroid_collision_time_before_mines_blew_up}, and after mines: {next_extrapolated_asteroid_collision_time}, and we waited this many ts for mines to blow up: {additional_timesteps_to_blow_up_mines} which is {additional_timesteps_to_blow_up_mines*DELTA_TIME}s")
        #states = self.get_state_sequence()
        overall_safe_time_fitness = sigmoid(safe_time_after_maneuver_s, 2.9, 1.4)

        ship_start_position = states[0].ship_state.position
        ship_end_position = states[-1].ship_state.position

        if len(states) >= 2:
            displacement = dist(ship_start_position, ship_end_position)
        else:
            displacement = 0.0
        asteroid_safe_time_fitness = get_asteroid_safe_time_fitness(next_extrapolated_asteroid_collision_time, displacement, move_sequence_length_s)
        if displacement < EPS or self.respawn_maneuver_pass_number > 0:
            # Stationary targeting or respawn maneuver
            self_ship_positions = [ship_end_position]
        else:
            # Regular maneuver
            #self_ship_positions = [states[len(states)//2]['ship_state'].position, states[len(states)*3//4]['ship_state'].position, ship_end_position]
            self_ship_positions = [s.ship_state.position for s in states]
            # This assertion won't hold when the ship takes a long time to rotate at the start
            # assert not (isclose(ship_start_position[0], self_ship_positions[0][0]) and isclose(ship_start_position[1], self_ship_positions[0][1])), f"Ship states: {[s['ship_state'] for s in states]}"

        other_ship_proximity_fitness = get_other_ship_proximity_fitness(self_ship_positions)
        sequence_length_fitness = get_sequence_length_fitness(move_sequence_length_s, displacement)

        crash_fitness = get_crash_fitness()

        if self.sim_placed_a_mine:
            #if mine_safe_time_fitness > 0.8:
            #    placed_mine_fitness = 1.0
            #else:
            #    placed_mine_fitness = 0.5
            placed_mine_fitness = mine_safe_time_fitness
        else:
            placed_mine_fitness = 0.0

        # debug_print(f"Fitness: {asteroid_safe_time_fitness + mine_safe_time_fitness + asteroids_fitness + sequence_length_fitness + other_ship_proximity_fitness + crash_fitness}, Ast safe time score: {asteroid_safe_time_fitness} (safe time after maneuver is {safe_time_after_maneuver_s} s, and current sim mode is {'stationary' if displacement < EPS else 'maneuver'}), asteroids score: {asteroids_fitness}, sequence length score: {sequence_length_fitness}, other ship prox score: {other_ship_proximity_fitness}")
        # self.explanation_messages.append(f"Fitness: {asteroid_safe_time_score + mine_safe_time_score + asteroids_score + sequence_length_score + displacement_score}, Ast safe time score: {asteroid_safe_time_score} (safe time after maneuver is {safe_time_after_maneuver_s} s, mine safe time score: {mine_safe_time_score}, and current sim mode is {'stationary' if displacement < EPS else 'maneuver'}), asteroids score: {asteroids_score}, sequence length score: {sequence_length_score}, displacement score: {displacement_score}, other ship prox score: {other_ship_proximity_score}")
        # debug_print(f"Fitness: {asteroid_safe_time_fitness + mine_safe_time_fitness + asteroids_fitness + sequence_length_fitness + other_ship_proximity_fitness + crash_fitness}, Ast safe time score: {asteroid_safe_time_fitness} (safe time after maneuver is {safe_time_after_maneuver_s} s, mine safe time score: {mine_safe_time_fitness}, and current sim mode is {'stationary' if displacement < EPS else 'maneuver'}), asteroids score: {asteroids_fitness}, sequence length score: {sequence_length_fitness}, other ship prox score: {other_ship_proximity_fitness}, crash_score: {crash_fitness}")
        if asteroid_safe_time_fitness < 0.1:
            self.safety_messages.append("I'm dangerously close to being hit by asteroids. Trying my hardest to maneuver out of this situation.")
        elif asteroid_safe_time_fitness < 0.4:
            self.safety_messages.append("I'm close to being hit by asteroids.")
        elif asteroid_safe_time_fitness < 0.8:
            self.safety_messages.append("I'll eventually get hit by asteroids. Keeping my eye out for a dodge maneuver.")

        if mine_safe_time_fitness < 0.1:
            self.safety_messages.append("I'm dangerously close to being kablooied by a mine. Trying my hardest to maneuver out of this situation.")
        elif mine_safe_time_fitness < 0.4:
            self.safety_messages.append("I'm close to being boomed by a mine.")
        elif mine_safe_time_fitness < 0.9:
            self.safety_messages.append("I'm within the radius of a mine.")

        if other_ship_proximity_fitness < 0.2:
            self.safety_messages.append("I'm dangerously close to the other ship. Get away from me!")
        elif other_ship_proximity_fitness < 0.5:
            self.safety_messages.append("I'm close to the other ship. Being cautious.")

        # Use fuzzy "AND" by averaging the fuzzy outputs
        fitness_breakdown = (asteroid_safe_time_fitness, mine_safe_time_fitness, asteroids_fitness, sequence_length_fitness, other_ship_proximity_fitness, crash_fitness, asteroid_aiming_cone_fitness, placed_mine_fitness, overall_safe_time_fitness)
        self.fitness_breakdown = fitness_breakdown
        if fitness_function_weights is not None:
            fitness_weights = fitness_function_weights
        else:
            # It might seem counterintuitive to weigh being near a mine to be worse than dying. But the issue with being near a mine, is that it can very easily lead to losing MULTIPLE lives, because if you get hit by an asteroid, and then not have time to clear the blast radius, you'll lose two lives. So yes, being near a mine can be worse than dying!
            fitness_weights = DEFAULT_FITNESS_WEIGHTS
        # print(fitness_weights)
        # overall_fitness = weighted_average(fitness_breakdown, fitness_weights)
        # print(fitness_breakdown, fitness_weights)
        overall_fitness = weighted_harmonic_mean(fitness_breakdown, fitness_weights, 1.0)
        # self.explanation_messages.append(f"Chose the sim with fitnesses: {overall_fitness=}, {asteroid_safe_time_fitness=}, {mine_safe_time_fitness=}, {asteroids_fitness=}, {sequence_length_fitness=}, {other_ship_proximity_fitness=}, {crash_fitness=}, {asteroid_aiming_cone_fitness=}")
        if overall_fitness > 0.9:
            self.safety_messages.append("I'm safe and chilling")
        else:
            pass
            # self.safety_messages.append(f"Stationary sim had fitnesses: {overall_fitness=}, {asteroid_safe_time_fitness=}, {mine_safe_time_fitness=}, {asteroids_fitness=}, {sequence_length_fitness=}, {other_ship_proximity_fitness=}, {crash_fitness=}, {asteroid_aiming_cone_fitness=}")
        # The overall_fitness is the fuzzy output. There's no need to defuzzify it since we're using this as a fitness value to rank the actions and future states
        return overall_fitness

    def get_fitness_breakdown(self) -> tuple[float, float, float, float, float, float, float, float, float]:
        # This is used to get the individual fitnesses before they got aggregated into the one fitness that was returned.
        assert self.fitness_breakdown is not None
        return self.fitness_breakdown

    def find_extreme_shooting_angle_error(self, asteroid_list: list[Target], threshold: float, mode: str = 'largest_below') -> Target | None:
        # This takes in a list of targets sorted by the shooting angle required, and it'll let you find the next target above or below a given target
        # This is useful if, say you want to shoot at a target, but the ship can't spin that far before you're able to shoot again, so you can pick a target to shoot along the way. I know this was a bad explanation but hopefully you get the idea.
        # Extract the shooting_angle_error_deg values
        shooting_angles = [d.shooting_angle_error_deg for d in asteroid_list]

        if mode == 'largest_below':
            # Find the index where threshold would be inserted
            #a = time.perf_counter()
            idx = bisect.bisect_left(shooting_angles, threshold)
            #b = time.perf_counter()
            # Adjust the index to get the largest value below the threshold
            if idx > 0:
                idx -= 1
            else:
                return None  # All values are greater than or equal to the threshold
        elif mode == 'smallest_above':
            # Find the index where threshold would be inserted
            #a = time.perf_counter()
            idx = bisect.bisect_right(shooting_angles, threshold)
            #b = time.perf_counter()
            # Check if all values are smaller than the threshold
            if idx >= len(shooting_angles):
                return None
        else:
            raise ValueError("Invalid mode. Choose 'largest_below' or 'smallest_above'")
        #print(f"Time taken to bisect: {b - a}")
        # Return the corresponding dictionary
        return asteroid_list[idx]

    def target_selection(self) -> bool:
        # The job of this method is to calculate how to hit each asteroid, and then pick the best one to try to target
        def simulate_shooting_at_target(target_asteroid_original: Asteroid, target_asteroid_shooting_angle_error_deg: float, target_asteroid_interception_time_s: float, target_asteroid_turning_timesteps: i64) -> tuple[Optional[Asteroid], list[Action], Asteroid, float, float, i64, Optional[i64], Ship]:
            '''
            Uses the bullet sim to check whether we'll hit a target, and which target we end up hitting since we might hit an asteroid in front of our intended target
            target_asteroid_original: Our target asteroid
            target_asteroid_shooting_angle_error_deg: The amount in degrees the ship needs to turn to be able to shoot the target
            target_asteroid_interception_time_s: The time in seconds between firing and hitting the center of the target. The turning time is not included!
            target_asteroid_turning_timesteps: The number of timesteps we need to turn for before shooting. Well, we might require fewer timesteps than this, but this is the prescribed number of timesteps we must wait out until we shoot, to be able to hit our target.
            '''
            
            # Just because we're lined up for a shot doesn't mean our shot will hit, unfortunately.
            # Bullets and asteroids travel in discrete timesteps, and it's possible for the bullet to miss the asteroid hitbox between timesteps, where the interception would have occurred on an intermediate timestep.
            # This is unavoidable, and we just have to choose targets that don't do this.
            # If the asteroids are moving slow enough, this should be rare, but especially if small asteroids are moving very quickly, this issue is common.
            # A simulation will easily show whether this will happen or not
            # debug_print(f"The last timestep fired is {self.last_timestep_fired}")
            aiming_move_sequence = self.get_rotate_heading_move_sequence(target_asteroid_shooting_angle_error_deg)
            #timesteps_until_can_fire: i64
            if self.fire_first_timestep:
                timesteps_until_can_fire = max(0, int(FIRE_COOLDOWN_TS) - len(aiming_move_sequence))
            else:
                timesteps_until_can_fire = max(0, int(FIRE_COOLDOWN_TS) - int(int(self.initial_timestep) + int(self.future_timesteps) + len(aiming_move_sequence) - int(self.last_timestep_fired)))
            # debug_print(f'aiming move seq before append, and ts until can fire is {timesteps_until_can_fire}')
            # debug_print(aiming_move_sequence)
            #aiming_move_sequence.extend([cast(Action, {'thrust': 0.0, 'turn_rate': 0.0, 'fire': False})]*timesteps_until_can_fire)
            aiming_move_sequence.extend([Action(thrust=0.0, turn_rate=0.0, fire=False) for _ in range(timesteps_until_can_fire)])
            # debug_print('aiming move seq after append')
            # debug_print(aiming_move_sequence)
            asteroid_advance_timesteps = len(aiming_move_sequence)
            # debug_print(f"Asteroid advanced timesteps: {asteroid_advance_timesteps}")
            # debug_print(f"Targetting turning timesteps: {target_asteroid_turning_timesteps}")
            if asteroid_advance_timesteps < target_asteroid_turning_timesteps:
                # We're given a budget of target_asteroid_turning_timesteps timesteps to turn, however we find that the turn actually required fewer timesteps than that. We still need to wait the full number, so we just pad with null actions to wait out the time. This case should be super rare.
                # debug_print(f"asteroid_advance_timesteps {asteroid_advance_timesteps} < target_asteroid_turning_timesteps {target_asteroid_turning_timesteps}")
                #aiming_move_sequence.extend([cast(Action, {'thrust': 0.0, 'turn_rate': 0.0, 'fire': False})]*(target_asteroid_turning_timesteps - asteroid_advance_timesteps))
                aiming_move_sequence.extend([Action(thrust=0.0, turn_rate=0.0, fire=False) for _ in range(target_asteroid_turning_timesteps - asteroid_advance_timesteps)])
            target_asteroid = target_asteroid_original.copy()
            target_asteroid = time_travel_asteroid(target_asteroid, asteroid_advance_timesteps, self.game_state)
            # debug_print(f"We're targetting asteroid {ast_to_string(target_asteroid)}")
            # debug_print(f"Entering the bullet target sim, we're on timestep {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)}")
            # debug_print(self.game_state.asteroids)
            #if not (abs(current_ship_state.velocity[0]) < GRAIN and abs(current_ship_state.velocity[1]) < GRAIN):
                # debug_print(f"Current ship velocity is {current_ship_state.velocity}")
            ship_state_after_aiming = self.get_ship_state()
            ship_state_after_aiming.heading = (ship_state_after_aiming.heading + target_asteroid_shooting_angle_error_deg) % 360.0
            actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, _ = self.bullet_sim(ship_state_after_aiming, self.fire_first_timestep, len(aiming_move_sequence))
            return actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming

        # First, find the most imminent asteroid
        # print('\nGOING INTO FUNCTION TO GET ALL FEASIBLE TARGETS FOR ASTEROIDS')
        target_asteroids_list: list[Target] = []
        #dummy_ship_state: Ship = {'is_respawning': False, 'position': self.ship_state.position, 'velocity': (0.0, 0.0), 'speed': 0.0, 'heading': self.ship_state.heading, 'mass': 300.0, 'radius': 20.0, 'id': self.ship_state.id, 'team': self.ship_state.team, 'lives_remaining': 123, 'bullets_remaining': 0, 'mines_remaining': 0, 'can_fire': self.ship_state.can_fire, 'fire_rate': 10.0, 'thrust_range': (-480.0, 480.0), 'turn_rate_range': (-180.0, 180.0), 'max_speed': 240, 'drag': 80.0}
        dummy_ship_state = Ship(
            is_respawning=False, 
            position=self.ship_state.position, 
            velocity=(0.0, 0.0), 
            speed=0.0, 
            heading=self.ship_state.heading, 
            mass=300.0, 
            radius=20.0, 
            id=self.ship_state.id, 
            team=self.ship_state.team,
            lives_remaining=123, 
            bullets_remaining=0, 
            mines_remaining=0, 
            can_fire=self.ship_state.can_fire,
            fire_rate=10.0, 
            thrust_range=(-480.0, 480.0), 
            turn_rate_range=(-180.0, 180.0), 
            max_speed=240, 
            drag=80.0
        )

        timesteps_until_can_fire: i64
        if self.fire_first_timestep:
            timesteps_until_can_fire = FIRE_COOLDOWN_TS
        else:
            timesteps_until_can_fire = max(0, FIRE_COOLDOWN_TS - (self.initial_timestep + self.future_timesteps - self.last_timestep_fired))
        # debug_print(f"\nSimulation starting from timestep {self.initial_timestep + self.future_timesteps}, and we need to wait this many until we can fire: {timesteps_until_can_fire}")
        most_imminent_asteroid_exists = False
        asteroids_still_exist = False
        # print(self.forecasted_asteroid_splits)

        for asteroid in chain(self.game_state.asteroids, self.forecasted_asteroid_splits):
            if check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, asteroid):
                asteroids_still_exist = True
                # print(f"\nOn TS {self.initial_timestep + self.future_timesteps} We do not have a pending shot for the asteroid {ast_to_string(asteroid)}")

                best_feasible_unwrapped_target: Optional[tuple[bool, float, i64, float, float, float, float]] = None
                # Check whether there are any mines that are about to go off, and if so, project this asteroid into the future to when the mine goes off to get a boolean of whether the asteroid will get hit by the mine or not.
                asteroid_will_get_hit_by_my_mine = False
                asteroid_will_get_hit_by_their_mine = False
                for m in self.game_state.mines:
                    #project_asteroid_by_timesteps_num = round(m.remaining_time*FPS)
                    #asteroid_when_mine_explodes = time_travel_asteroid(asteroid, project_asteroid_by_timesteps_num, self.game_state)
                    asteroid_when_mine_explodes = time_travel_asteroid_s(asteroid, m.remaining_time, self.game_state)
                    #if check_collision(asteroid_when_mine_explodes.position[0], asteroid_when_mine_explodes.position[1], asteroid_when_mine_explodes.radius, m.position[0], m.position[1], MINE_BLAST_RADIUS):
                    delta_x = asteroid_when_mine_explodes.position[0] - m.position[0]
                    delta_y = asteroid_when_mine_explodes.position[1] - m.position[1]
                    separation = asteroid_when_mine_explodes.radius + MINE_BLAST_RADIUS
                    if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                        # Keep track of whose mine this is. If it's mine, I want more asteroids to be in its blast radius so it does more damage. If it's theirs, I want to shoot asteroids within its blast radius so it does less damage.
                        if m.position in self.mine_positions_placed:
                            asteroid_will_get_hit_by_my_mine = True
                            #print('WILL BE HIT BY MY MINE')
                            if asteroid_will_get_hit_by_their_mine:
                                break
                        else:
                            #print('WILL BE HIT BY THEIR MINE')
                            asteroid_will_get_hit_by_their_mine = True
                            if asteroid_will_get_hit_by_my_mine:
                                break
                # Iterate through all unwrapped asteroids to find which one of the unwraps is the best feasible target.
                # 99% of the time, only one of the unwraps will have a feasible target, but there's situations where we could either shoot the asteroid before it wraps, or wait for it to wrap and then shoot it.
                # In these cases, we need to pick whichever option is the fastest when factoring in turn time and waiting time.
                unwrapped_asteroids = unwrap_asteroid(asteroid, self.game_state.map_size[0], self.game_state.map_size[1], UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON, True)
                for a in unwrapped_asteroids:
                    feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = solve_interception(a, dummy_ship_state, self.game_state, timesteps_until_can_fire)

                    if feasible:
                        if best_feasible_unwrapped_target is None or aiming_timesteps_required < best_feasible_unwrapped_target[2]:
                            #print((feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception))
                            best_feasible_unwrapped_target = (feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception)
                    else:
                        pass
                        #print(f'INFEASIBLE SHOT for ast {ast_to_string(a)}')
                        #print(feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception)
                if best_feasible_unwrapped_target is not None:
                    feasible, shooting_angle_error_deg, aiming_timesteps_required, interception_time_s, intercept_x, intercept_y, asteroid_dist_during_interception = best_feasible_unwrapped_target
                    imminent_collision_time_s = inf
                    for a in unwrapped_asteroids:
                        imminent_collision_time_s = min(imminent_collision_time_s, predict_next_imminent_collision_time_with_asteroid(self.ship_state.position[0], self.ship_state.position[1], self.ship_state.velocity[0], self.ship_state.velocity[1], SHIP_RADIUS, a.position[0], a.position[1], a.velocity[0], a.velocity[1], a.radius))
                    target_asteroids_list.append(Target(
                        asteroid=asteroid.copy(),  # Record the canonical asteroid even if we're shooting at an unwrapped one
                        feasible=feasible,  # Will be True
                        shooting_angle_error_deg=shooting_angle_error_deg,
                        aiming_timesteps_required=aiming_timesteps_required,
                        interception_time_s=interception_time_s,
                        intercept_x=intercept_x,
                        intercept_y=intercept_y,
                        asteroid_dist_during_interception=asteroid_dist_during_interception,
                        imminent_collision_time_s=imminent_collision_time_s,
                        asteroid_will_get_hit_by_my_mine=asteroid_will_get_hit_by_my_mine,
                        asteroid_will_get_hit_by_their_mine=asteroid_will_get_hit_by_their_mine,
                    ))
                    if imminent_collision_time_s < inf:
                        # debug_print(f"Imminent collision time is less than inf! {imminent_collision_time_s}")
                        most_imminent_asteroid_exists = True
        # Check whether we have enough time to aim at it and shoot it down
        # PROBLEM, what if the asteroid breaks into pieces and I need to shoot those down too? But I have plenty of time, and I still want the fitness function to be good in that case, but there's no easy way to evaluate that. It's hard to decide whether we want to shoot the asteroids that are about to hit us, or to just dodge it by moving myself.

        turn_angle_deg_until_can_fire = float(timesteps_until_can_fire)*SHIP_MAX_TURN_RATE*DELTA_TIME  # Can be up to 18 degrees
        # print(target_asteroids_list)
        # if there’s an imminent shot coming toward me, I will aim at the asteroid that gets me CLOSEST to the direction of the imminent shot.
        # So it only plans one shot at a time instead of a series of shots, and it’ll keep things simpler
        # debug_print(f"Least angular dist: {least_angular_distance_asteroid_shooting_angle_error_deg}")
        # debug_print('Target asts list: ', target_asteroids_list)
        actual_asteroid_hit = None
        aiming_move_sequence: list[Action] = []

        if most_imminent_asteroid_exists:
            # First try to shoot the most imminent asteroids, if they exist

            # debug_print(f"Shooting at most imminent asteroids. Most imminent collision time is {most_imminent_collision_time_s}s with turn angle error {most_imminent_asteroid_shooting_angle_error_deg}")
            # debug_print(most_imminent_asteroid)
            # Find the asteroid I can shoot at that gets me closest to the imminent shot, if I can't reach the imminent shot in time until I can shoot
            # Sort the targets such that we prioritize asteroids that are about to hit me
            # If there's a bullet limit, penalize risky shots more
            sorted_imminent_targets = target_asteroids_list
            if self.other_ships:
                frontrun_score_multiplier = 4.0 if self.ship_state.bullets_remaining > 0 else 3.0
                sorted_imminent_targets.sort(key=lambda t: (
                    min(10, t.imminent_collision_time_s) +
                    ASTEROID_SIZE_SHOT_PRIORITY[t.asteroid.size]*0.25 +
                    t.interception_time_s + # Might be more correct to do t.interception_time_s + t.aiming_timesteps_required*DELTA_TIME - get_adversary_interception_time_lower_bound(t.asteroid, self.other_ships, self.game_state)
                    t.asteroid_dist_during_interception/400.0 +
                    frontrun_score_multiplier*min(0.5, max(0, t.interception_time_s - get_adversary_interception_time_lower_bound(t.asteroid, self.other_ships, self.game_state))) +
                    ((5.0 if t.asteroid.size == 1 else -5.0) if t.asteroid_will_get_hit_by_my_mine else 0.0) +
                    ((3.0 if t.asteroid.size != 1 else -3.0) if t.asteroid_will_get_hit_by_their_mine else 0.0)
                ))
            else:
                sorted_imminent_targets.sort(key=lambda t: (
                    min(10, t.imminent_collision_time_s) +
                    ASTEROID_SIZE_SHOT_PRIORITY[t.asteroid.size]*0.25 +
                    t.asteroid_dist_during_interception/400.0 +
                    ((5.0 if t.asteroid.size == 1 else -5.0) if t.asteroid_will_get_hit_by_my_mine else 0.0)
                ))
            # The frontrun time is bounded by 0 and 0.5 seconds, since anything after half a second away is basically the same and there's no point differentiating between them
            # TODO: For each asteroid, give it a couple feasible times where we wait longer and longer. This way we can choose to wait a timestep to fire again if we'll get luckier with the bullet lining up
            for _, candidate_target in enumerate(sorted_imminent_targets):
                if isinf(candidate_target.imminent_collision_time_s):
                    # Ran through all imminent asteroids! Everything onward won't collide with me anytime soon.
                    break
                most_imminent_asteroid_aiming_timesteps = candidate_target.aiming_timesteps_required
                most_imminent_asteroid = candidate_target.asteroid
                most_imminent_asteroid_shooting_angle_error_deg = candidate_target.shooting_angle_error_deg
                most_imminent_asteroid_interception_time_s = candidate_target.interception_time_s
                # debug_print(f"Shooting at asteroid that's going to hit me: {ast_to_string(most_imminent_asteroid)}")
                if most_imminent_asteroid_aiming_timesteps <= timesteps_until_can_fire:
                    # I can reach the imminent shot without wasting a shot opportunity, so do it
                    #assert isinstance(most_imminent_asteroid, Asteroid)
                    #assert isinstance(most_imminent_asteroid_shooting_angle_error_deg, float)
                    #assert isinstance(most_imminent_asteroid_interception_time_s, float)
                    #assert isinstance(most_imminent_asteroid_aiming_timesteps, i64)
                    actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming = simulate_shooting_at_target(most_imminent_asteroid, most_imminent_asteroid_shooting_angle_error_deg, most_imminent_asteroid_interception_time_s, most_imminent_asteroid_aiming_timesteps)
                    if actual_asteroid_hit is not None:
                        # We can hit the target
                        assert timesteps_until_bullet_hit_asteroid is not None
                        len_aiming_move_sequence = len(aiming_move_sequence)
                        actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len_aiming_move_sequence - timesteps_until_bullet_hit_asteroid, self.game_state)
                        if check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len_aiming_move_sequence, self.game_state, actual_asteroid_hit_when_firing):
                            # We haven't already shot at the target. This is good, so let's use it. Break out of the loop and don't check any more asteroids.
                            break
                else:
                    # Between now and when I can shoot, I don't have enough time to aim at the imminent asteroid.
                    # Instead, find the closest asteroid along the way to shoot
                    # Sort by angular distance, with the unlikely tie broken by shot size
                    sorted_targets = target_asteroids_list
                    sorted_targets.sort(key=lambda t: (round(t.shooting_angle_error_deg), ASTEROID_SIZE_SHOT_PRIORITY[t.asteroid.size]))
                    # debug_print(f"Turn angle deg until we can fire (max 18 degrees): {turn_angle_deg_until_can_fire}")
                    if most_imminent_asteroid_shooting_angle_error_deg > 0.0:
                        # debug_print("The imminent shot requires us to turn the ship to the left")
                        target = self.find_extreme_shooting_angle_error(sorted_targets, turn_angle_deg_until_can_fire, 'largest_below')
                        if target is None or target.shooting_angle_error_deg < 0.0 or target.shooting_angle_error_deg < turn_angle_deg_until_can_fire - TARGETING_AIMING_UNDERTURN_ALLOWANCE_DEG:
                            # We're underturning too much, so instead find the next overturn
                            # debug_print("Underturning too much, so instead find the next overturn to the left")
                            target = self.find_extreme_shooting_angle_error(sorted_targets, turn_angle_deg_until_can_fire, 'smallest_above')
                    else:
                        # debug_print("The imminent shot requires us to turn the ship to the right")
                        target = self.find_extreme_shooting_angle_error(sorted_targets, -turn_angle_deg_until_can_fire, 'smallest_above')
                        # print("Found the next target to the right:")
                        # print(target)
                        if target is None or target.shooting_angle_error_deg > 0.0 or target.shooting_angle_error_deg > -turn_angle_deg_until_can_fire + TARGETING_AIMING_UNDERTURN_ALLOWANCE_DEG:
                            # We're underturning too much, so instead find the next overturn
                            # debug_print("Underturning too much, so instead find the next overturn to the right")
                            target = self.find_extreme_shooting_angle_error(sorted_targets, -turn_angle_deg_until_can_fire, 'largest_below')
                            # print(target)
                    if target is not None:
                        # debug_print('As our target were choosing this one which will be on our way:')
                        # debug_print(target)
                        actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming = simulate_shooting_at_target(target.asteroid, target.shooting_angle_error_deg, target.interception_time_s, target.aiming_timesteps_required)
                        if actual_asteroid_hit is not None:
                            assert timesteps_until_bullet_hit_asteroid is not None
                            len_aiming_move_sequence = len(aiming_move_sequence)
                            actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len_aiming_move_sequence - timesteps_until_bullet_hit_asteroid, self.game_state)
                            if check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len_aiming_move_sequence, self.game_state, actual_asteroid_hit_when_firing):
                                break
                        #    print(f"DANG IT, we're shooting something on the way to the most imminent asteroid, but we'll miss this particular one!")
                    else:
                        # Just gonna have to waste shot opportunities and turn all the way
                        actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming = simulate_shooting_at_target(most_imminent_asteroid, most_imminent_asteroid_shooting_angle_error_deg, most_imminent_asteroid_interception_time_s, most_imminent_asteroid_aiming_timesteps)
                        if actual_asteroid_hit is not None:
                            assert timesteps_until_bullet_hit_asteroid is not None
                            len_aiming_move_sequence = len(aiming_move_sequence)
                            actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len_aiming_move_sequence - timesteps_until_bullet_hit_asteroid, self.game_state)
                            if check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len_aiming_move_sequence, self.game_state, actual_asteroid_hit_when_firing):
                                break
            #if actual_asteroid_hit:
            #    self.explanation_messages.append("Shooting at asteroid on a collision course with me.")
        if not actual_asteroid_hit:
            # Nothing has been hit from the imminent shots so far. Either no imminent asteroids exist, or we tried (simulated) hitting some but we missed them all.
            # Move down the list to trying for convenient shots.
            if target_asteroids_list:
                self.explanation_messages.append("No asteroids on collision course with me. Shooting at asteroids with least turning delay.")
                sorted_targets = target_asteroids_list
                if self.other_ships:
                    # If there's a bullet limit, penalize risky shots more
                    frontrun_score_multiplier = 25.0 if self.ship_state.bullets_remaining > 0 else 15.0
                    # Sort by just convenience (and anything else I'd like)
                    sorted_targets.sort(key=lambda t: (
                        float(t.aiming_timesteps_required)*2.0 +
                        ASTEROID_SIZE_SHOT_PRIORITY[t.asteroid.size] +
                        t.interception_time_s + # Might be more correct to do t.interception_time_s + t.aiming_timesteps_required*DELTA_TIME - get_adversary_interception_time_lower_bound(t.asteroid, self.other_ships, self.game_state)
                        t.asteroid_dist_during_interception/400.0 +
                        frontrun_score_multiplier*min(0.5, max(0, t.interception_time_s - get_adversary_interception_time_lower_bound(t.asteroid, self.other_ships, self.game_state))) +
                        ((20.0 if t.asteroid.size == 1 else -20.0) if t.asteroid_will_get_hit_by_my_mine else 0.0) +
                        ((20.0 if t.asteroid.size != 1 else -20.0) if t.asteroid_will_get_hit_by_their_mine else 0.0)
                    ))
                else:
                    sorted_targets.sort(key=lambda t: (
                        float(t.aiming_timesteps_required)*2.0 +
                        ASTEROID_SIZE_SHOT_PRIORITY[t.asteroid.size] +
                        t.asteroid_dist_during_interception/400.0 +
                        ((20.0 if t.asteroid.size == 1 else -20.0) if t.asteroid_will_get_hit_by_my_mine else 0.0)
                    ))
                for confirmed_target in sorted_targets:
                    least_shot_delay_asteroid = confirmed_target.asteroid
                    least_shot_delay_asteroid_shooting_angle_error_deg = confirmed_target.shooting_angle_error_deg
                    least_shot_delay_asteroid_interception_time_s = confirmed_target.interception_time_s
                    least_shot_delay_asteroid_aiming_timesteps = confirmed_target.aiming_timesteps_required
                    #least_shot_delay_asteroid = cast(Asteroid, least_shot_delay_asteroid)
                    #assert isinstance(least_shot_delay_asteroid_shooting_angle_error_deg, float)
                    #assert isinstance(least_shot_delay_asteroid_interception_time_s, float)
                    #assert isinstance(least_shot_delay_asteroid_aiming_timesteps, i64)
                    actual_asteroid_hit, aiming_move_sequence, target_asteroid, target_asteroid_shooting_angle_error_deg, target_asteroid_interception_time_s, target_asteroid_turning_timesteps, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming = simulate_shooting_at_target(least_shot_delay_asteroid, least_shot_delay_asteroid_shooting_angle_error_deg, least_shot_delay_asteroid_interception_time_s, least_shot_delay_asteroid_aiming_timesteps)
                    # actual_asteroid_hit is at the timestep (self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid)
                    # The timesteps_until_bullet_hit_asteroid time INCLUDES the aiming time!
                    # The reason we can't just check whether the asteroid in the future has a pending shot, and we have to back-extrapolate to when we fire, is that, say I do a shot at time 0, and it’ll hit the ast at time 10. If at time 8, I check whether I’ve shot at the asteroid already, and say it’ll take 5 timesteps to hit it. I need to check at time 8, NOT TIME 13! Time 13 is after the original shot hits it, so I can’t be checking it after. So THAT’S WHY I should be back-projecting the asteroid.
                    if actual_asteroid_hit is not None:
                        assert timesteps_until_bullet_hit_asteroid is not None
                        len_aiming_move_sequence = len(aiming_move_sequence)
                        actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len_aiming_move_sequence - timesteps_until_bullet_hit_asteroid, self.game_state)
                        if check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len_aiming_move_sequence, self.game_state, actual_asteroid_hit_when_firing):  # , f"Dang it. {self.initial_timestep=}, {self.future_timesteps=}, actual_asteroid_hit_when_firing: {ast_to_string(actual_asteroid_hit_when_firing)}, {timesteps_until_bullet_hit_asteroid=}, {len(aiming_move_sequence)=}, self.asteroids_pending_death: {self.asteroids_pending_death}"
                            break
            else:
                # Ain't nothing to shoot at
                # debug_print('Nothing to shoot at!')
                self.explanation_messages.append("There's nothing I can feasibly shoot at!")
                # Pick a direction to turn the ship anyway, just to better line it up with more shots
                if asteroids_still_exist:
                    if random.randint(1, 10) == 1:
                        self.explanation_messages.append("Asteroids exist but we can't hit them. Moving around a bit randomly.")
                        # Setting this to -1 is a signal to set the asteroid fitness really low, so hopefully we'll choose actions that'll move around
                        self.asteroids_shot -= 1
                    turn_direction = 0
                    idle_thrust = 0.0
                else:
                    self.explanation_messages.append("Asteroids no longer exist. We're all done!")
                    #self.explanation_messages.append(f"In this scenario, I have simulated {total_sim_timesteps*DELTA_TIME/60:0.0f} minutes of gameplay! Bullet sim took {bullet_sim_time} s and the sim update took {sim_update_total_time} s with the sim culling taking {sim_cull_total_time} s, unwrapping taking {unwrap_total_time} s, and culling asteroids_pending_death is {asteroids_pending_death_total_cull_time} s, {asteroid_tracking_total_time=}, {asteroid_new_track_total_time=}")
                    #self.explanation_messages.append(f"In this scenario, I have simulated {float(total_sim_timesteps)*DELTA_TIME/60:0.0f} minutes of gameplay!")
                    #self.explanation_messages.append(f"We simulated {total_bullet_sim_timesteps} bullet sim timesteps and {total_bullet_sim_iterations} iterations, and {total_sim_timesteps} timesteps, and {update_ts_multiple_count=} {update_ts_zero_count=}")
                    turn_direction = 0
                    idle_thrust = 0.0
                # We still simulate one iteration of this, because if we had a pending shot from before, this will do the shot!
                sim_complete_without_crash = self.update(idle_thrust, SHIP_MAX_TURN_RATE*turn_direction, False)
                return sim_complete_without_crash

        # debug_print('Closest ang asteroid:')
        # debug_print(target_asteroids_list[least_angular_distance_asteroid_index])
        # debug_print('Second closest ang asteroid:')
        # debug_print(target_asteroids_list[second_least_angular_distance_asteroid_index])

        # print(f"Bullet should have been fired on simulated timestep {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)}")

        # The following code is confusing, because we're working with different times.
        # We need to make sure that when we talk about an asteroid, we talk about its position at a specific time. If the time we talk about is not synced up with the position, everything's wrong.
        # So if an asteroid is on position 1 at time 1, position 2 at time 2, position 3 at time 3, etc, then we must associate the asteroid with a timestep.
        # A smarter, less buggy way to store asteroids is to include not only their position, but their timestep as well. But too late.
        # Timestep [self.initial_timestep + self.future_timesteps] is the current timestep, or the timestep the sim was started on. Future timesteps should be 0 so far since we haven't moved.
        # Timestep [self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)] gives us the timestep after we do our aiming, and when we shoot our bullet
        # Timestep [self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid] gives us the timestep the asteroid got hit

        if actual_asteroid_hit is not None:
            assert timesteps_until_bullet_hit_asteroid is not None
            actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid, self.game_state)
            if not isinf(self.game_state.time_limit) and self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid > floor(FPS*self.game_state.time_limit):
                # Added one to the timesteps to prevent off by one :P
                #print('WITHHOLDING SHOT IN TARGET SELECTION BECAUSE SCENARIO IS ENDING')
                self.fire_next_timestep_flag = False
                sim_complete_without_crash = self.update(0.0, 0.0, False)
                return sim_complete_without_crash
        # TODO: This following statement, doesn't this mean if I accidentally hit something in the way that I already shot, then I just won't even try to shoot at all? Even though the bullet should still hit what's behind it some of the time at least?
        # TODO: Or maybe this case isn't even possible because I do the sim to make sure that I hit whatever I wanted to hit, and if I had a pending shot, that shot would have hit it so this would never happen!
        if actual_asteroid_hit is None or not check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len(aiming_move_sequence), self.game_state, actual_asteroid_hit_when_firing):
            # We can't seem to hit anything on this timestep. Maybe the asteroids are done being shot, or maybe we can't hit them because of the way things are lined up and the bullets skip over the asteroids, idk.
            # if ENABLE_ASSERTIONS and actual_asteroid_hit is not None:
            # TODO: This is testing above theory to see if it holds
            #    assert check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len(aiming_move_sequence), self.game_state, actual_asteroid_hit_when_firing), f"WTH this asteroid: {actual_asteroid_hit_when_firing}, whole freaking dict: {self.asteroids_pending_death}"
            self.fire_next_timestep_flag = False
            # print("We can't hit anything this timestep.")
            if asteroids_still_exist:
                # print('asteroids still exist')
                if random.randint(1, 10) == 1:
                    self.explanation_messages.append("Asteroids exist but we can't hit them. Moving around a bit randomly.")
                    # Setting this to -1 is a signal to set the asteroid fitness really low, so hopefully we'll choose actions that'll move around
                    self.asteroids_shot -= 1
                # turn_direction = random.random()
                # idle_thrust = random.triangular(0, SHIP_MAX_THRUST, 0)
                turn_direction = 0
                idle_thrust = 0.0
            else:
                # print('asteroids DO NOT exist')
                turn_direction = 0
                idle_thrust = 0.0
            # print(f"Calling update from targ sel:")
            sim_complete_without_crash = self.update(idle_thrust, SHIP_MAX_TURN_RATE*turn_direction, False)
            # print(f"Sim id {self.sim_id} is returning from target sim with success value {sim_complete_without_crash}")
            return sim_complete_without_crash
        else:
            # We're able to hit an asteroid! We're committing to it.
            assert timesteps_until_bullet_hit_asteroid is not None
            # print(f"Asserting that we don't have a pending shot for asteroid {ast_to_string(actual_asteroid_hit)} on timestep {self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid}")
            # print(f"Current timestep: {self.initial_timestep + self.future_timesteps}, and the aiming maneuver is {len(aiming_move_sequence)}")
            # print(self.asteroids_pending_death)
            # actual_asteroid_hit_when_firing = time_travel_asteroid(actual_asteroid_hit, len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid, self.game_state)
            # print(f"Asserting that we don't have a pending shot for asteroid {ast_to_string(actual_asteroid_hit_when_firing)} on timestep {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence)}")
                # assert check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + len(aiming_move_sequence) - 1, self.game_state, actual_asteroid_hit_when_firing)

            # actual_asteroid_hit_UNEXTRAPOLATED = dict(actual_asteroid_hit)
            # This back-extrapolates the asteroid to when we're firing our bullet

            #debug_print(f"We used a sim to forecast the asteroid that we'll hit being the one at pos ({actual_asteroid_hit.position[0]}, {actual_asteroid_hit.position[1]}) with vel ({actual_asteroid_hit.velocity[0]}, {actual_asteroid_hit.velocity[1]}) in {timesteps_until_bullet_hit_asteroid - len(aiming_move_sequence)} timesteps")
            #debug_print(f"The primitive forecast would have said we'd hit asteroid at pos ({target_asteroid.position[0]}, {target_asteroid.position[1]}) with vel ({target_asteroid.velocity[0]}, {target_asteroid.velocity[1]}) in {calculate_timesteps_until_bullet_hits_asteroid(target_asteroid_interception_time_s, target_asteroid.radius)} timesteps")
            # print(self.asteroids_pending_death)
            # print(f"\nTracking that we just shot at the asteroid {ast_to_string(actual_asteroid_hit)}, our intended target was {target_asteroid}")
            # actual_asteroid_hit_UNEXTRAPOLATED = extrapolate_asteroid_forward(actual_asteroid_hit, -(len(aiming_move_sequence) - timesteps_until_bullet_hit_asteroid), self.game_state, True)
            actual_asteroid_hit_at_present_time = time_travel_asteroid(actual_asteroid_hit, -timesteps_until_bullet_hit_asteroid, self.game_state)
            # actual_asteroid_hit_tracking_purposes_super_early = extrapolate_asteroid_forward(actual_asteroid_hit, )
            # print(f"Asserting that we don't have a pending shot for asteroid {ast_to_string(actual_asteroid_hit_at_present_time)} on timestep {self.initial_timestep + self.future_timesteps}")
            # assert check_whether_this_is_a_new_asteroid_we_do_not_have_a_pending_shot_for(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + 0*len(aiming_move_sequence), self.game_state, actual_asteroid_hit_at_present_time)
            # print(f'\n\nTracking that we shot at the asteroid {ast_to_string(actual_asteroid_hit_at_present_time)} at the timestep after turning {self.initial_timestep + self.future_timesteps + len(aiming_move_sequence) - 1}')
            # print(self.asteroids_pending_death)
            # debug_print(f"Move seq is len {len(aiming_move_sequence)} and is {aiming_move_sequence}")
            # debug_print(f"We're in the future by {self.future_timesteps} timesteps")
            # actual_asteroid_hit_when_firing['trace'] = f"Asteroid added from target selection sim id {self.sim_id}, self.initial_timestep {self.initial_timestep} self.future_timesteps {self.future_timesteps} aiming_move_sequence {aiming_move_sequence}"

            # debug_print('Were gonna fire the next timestep! Printing out the move sequence:', aiming_move_sequence)
            # TODO: Currently Neo can't fire on timestep 1! Fix this! Although it's only a minor issue.
            future_ts_backup = self.future_timesteps
            # Forecasted splits get progressed while doing the move sequence which includes rotation, so we need to start the forecast before the rotation even starts
            if actual_asteroid_hit_at_present_time.size != 1:
                #print("Calling forecast ast bull splits from targ sel")
                self.forecasted_asteroid_splits.extend(forecast_asteroid_bullet_splits_from_heading(actual_asteroid_hit_at_present_time, timesteps_until_bullet_hit_asteroid, ship_state_after_aiming.heading, self.game_state))
                # print(f"In targeting, Forecasted asteroid splits on ts {self.initial_timestep + self.future_timesteps} is {self.forecasted_asteroid_splits}")
            sim_complete_without_crash = self.apply_move_sequence(aiming_move_sequence)
            if sim_complete_without_crash:
                # We only do, and track the shot if the sim completed without death
                self.asteroids_shot += 1
                self.fire_next_timestep_flag = True
                #if is_close(-10.0, actual_asteroid_hit_when_firing.velocity[0]):
                #    print(f"\nSHOT AT THE ASTEROID IN TARGET SELECTION on timestep {self.initial_timestep=} {self.future_timesteps=}")
                track_asteroid_we_shot_at(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps, self.game_state, timesteps_until_bullet_hit_asteroid - len(aiming_move_sequence), actual_asteroid_hit_when_firing)
                # print(f"self.asteroids_pending_death updated to {self.asteroids_pending_death[100]}")
                # print(self.asteroids_pending_death)
            else:
                # If we died, we're gonna do a respawn maneuver and we won't shoot because that will remove our 3 second respawn invincibility immediately
                self.fire_next_timestep_flag = False
            # print('THE ACTUAL MOVE SEQUENCE WE GET BACK FROM THE SIM:')
            # print(self.ship_move_sequence)
            # debug_print(f"Sim id {self.sim_id} is returning from target sim with success value {sim_complete_without_crash}")
            return sim_complete_without_crash

    def bullet_sim(self, ship_state: Optional[Ship] = None, fire_first_timestep: bool = False, fire_after_timesteps: i64 = 0, skip_half_of_first_cycle: bool = False, current_move_index: Optional[i64] = None, whole_move_sequence: Optional[list[Action]] = None, timestep_limit: i64 = INT_INF, asteroids_to_check: Optional[list[Asteroid]] = None) -> tuple[Optional[Asteroid], i64, bool]:
        # This simulates shooting at an asteroid to tell us whether we'll hit it, when we hit it, and which asteroid we hit (since we might hit an asteroid between us and our intended target)
        # I know there's lots of code duplication between this and update(), but there really are many differences in the way the update cycle is handled and it's very interleaved with the duplicated simulation code, so I ain't touching this lol
        # Assume we shoot on the next timestep, so we'll create a bullet and then track it and simulate it to see what it hits, if anything
        # This sim doesn't modify the state of the simulation class. Everything here is discarded after the sim is over, and this is just to see what my bullet hits, if anything.

        # Do the shallowest copies that we can get away with. copy.deepcopy is super slow so we avoid it
        #asteroids: list[Asteroid]
        if asteroids_to_check is not None:
            asteroids = cast(list[Asteroid], [a.copy() for a in asteroids_to_check])
        else:
            asteroids = cast(list[Asteroid], [a.copy() for a in self.game_state.asteroids])
        mines: list[Mine] = cast(list[Mine], [m.copy() for m in self.game_state.mines])
        bullets: list[Bullet] = cast(list[Bullet], [b.copy() for b in self.game_state.bullets])
        initial_ship_state = self.get_ship_state()
        if whole_move_sequence:
            bullet_sim_ship_state = self.get_ship_state()
        else:
            bullet_sim_ship_state = None
        my_bullet: Optional[Bullet] = None
        ship_not_collided_with_asteroid: bool = True
        timesteps_until_bullet_hit_asteroid: i64
        if skip_half_of_first_cycle:
            timesteps_until_bullet_hit_asteroid = -1
        else:
            timesteps_until_bullet_hit_asteroid = 0
        # Keep iterating until our bullet flies off the edge of the screen, or it hits an asteroid
        #global total_bullet_sim_timesteps
        #global total_bullet_sim_iterations
        #total_bullet_sim_iterations += 1
        asteroid_remove_idxs: set[i64] = set()
        global total_sim_timesteps
        #global bullet_sim_time
        #start_time = time.perf_counter()
        while True:
            #total_bullet_sim_timesteps += 1
            total_sim_timesteps += 1
            # Simplified update() simulation loop
            timesteps_until_bullet_hit_asteroid += 1
            if timesteps_until_bullet_hit_asteroid > timestep_limit:
                #bullet_sim_time += time.perf_counter() - start_time
                return None, i64(-1), ship_not_collided_with_asteroid
            # Simulate bullets
            if not (skip_half_of_first_cycle and timesteps_until_bullet_hit_asteroid == 0):
                bullet_remove_idxs: list[i64] = []
                for b_ind, b in enumerate(bullets):
                    new_bullet_pos = (b.position[0] + b.velocity[0]*DELTA_TIME, b.position[1] + b.velocity[1]*DELTA_TIME)
                    if check_coordinate_bounds(self.game_state, new_bullet_pos[0], new_bullet_pos[1]):
                        b.position = new_bullet_pos
                    else:
                        bullet_remove_idxs.append(b_ind)
                if bullet_remove_idxs:
                    bullets = [bullet for idx, bullet in enumerate(bullets) if idx not in bullet_remove_idxs]
                if my_bullet is not None:
                    my_new_bullet_pos = (my_bullet.position[0] + my_bullet.velocity[0]*DELTA_TIME, my_bullet.position[1] + my_bullet.velocity[1]*DELTA_TIME)
                    if check_coordinate_bounds(self.game_state, my_new_bullet_pos[0], my_new_bullet_pos[1]):
                        my_bullet.position = my_new_bullet_pos
                    else:
                        #bullet_sim_time += time.perf_counter() - start_time
                        return None, i64(-1), ship_not_collided_with_asteroid  # The bullet got shot into the void without hitting anything :(

                for m in mines:
                    m.remaining_time -= DELTA_TIME

                for a in asteroids:
                    a.position = ((a.position[0] + a.velocity[0]*DELTA_TIME) % self.game_state.map_size[0], (a.position[1] + a.velocity[1]*DELTA_TIME) % self.game_state.map_size[1])

            # debug_print(f"TS ahead of sim end: {timesteps_until_bullet_hit_asteroid}")
            # debug_print(asteroids)
            # Create the initial bullet we fire, if we're locked in
            if fire_first_timestep and timesteps_until_bullet_hit_asteroid + (-1 if not skip_half_of_first_cycle else 0) == 0:
                rad_heading = radians(initial_ship_state.heading)
                cos_heading = cos(rad_heading)
                sin_heading = sin(rad_heading)
                bullet_x = initial_ship_state.position[0] + SHIP_RADIUS*cos_heading
                bullet_y = initial_ship_state.position[1] + SHIP_RADIUS*sin_heading
                # Make sure the bullet isn't being fired out into the void
                if check_coordinate_bounds(self.game_state, bullet_x, bullet_y):
                    vx = BULLET_SPEED*cos_heading
                    vy = BULLET_SPEED*sin_heading
                    initial_timestep_fire_bullet = Bullet(
                        position=(bullet_x, bullet_y),
                        velocity=(vx, vy),
                        heading=initial_ship_state.heading,
                        mass=BULLET_MASS,
                        tail_delta=(-BULLET_LENGTH*cos_heading, -BULLET_LENGTH*sin_heading)
                    )
                    bullets.append(initial_timestep_fire_bullet)
            # The new bullet we create will end up at the end of the list of bullets
            if my_bullet is None and timesteps_until_bullet_hit_asteroid + (-1 if not skip_half_of_first_cycle else 0) == fire_after_timesteps:
                if ship_state is not None:
                    bullet_fired_from_ship_heading = ship_state.heading
                    bullet_fired_from_ship_position = ship_state.position
                else:
                    bullet_fired_from_ship_heading = self.ship_state.heading
                    bullet_fired_from_ship_position = self.ship_state.position
                rad_heading = radians(bullet_fired_from_ship_heading)
                cos_heading = cos(rad_heading)
                sin_heading = sin(rad_heading)
                bullet_x = bullet_fired_from_ship_position[0] + SHIP_RADIUS*cos_heading
                bullet_y = bullet_fired_from_ship_position[1] + SHIP_RADIUS*sin_heading
                # Make sure the bullet isn't being fired out into the void
                if not check_coordinate_bounds(self.game_state, bullet_x, bullet_y):
                    #bullet_sim_time += time.perf_counter() - start_time
                    return None, i64(-1), ship_not_collided_with_asteroid  # The bullet got shot into the void without hitting anything :(
                vx = BULLET_SPEED*cos_heading
                vy = BULLET_SPEED*sin_heading
                my_bullet = Bullet(
                    position=(bullet_x, bullet_y),
                    velocity=(vx, vy),
                    heading=bullet_fired_from_ship_heading,
                    mass=BULLET_MASS,
                    tail_delta=(-BULLET_LENGTH*cos_heading, -BULLET_LENGTH*sin_heading)
                )
            if whole_move_sequence:
                assert current_move_index is not None
                assert bullet_sim_ship_state is not None
                if current_move_index + timesteps_until_bullet_hit_asteroid + (-1 if not skip_half_of_first_cycle else 0) < len(whole_move_sequence):
                    # Simulate ship dynamics, if we have the full future list of moves to go off of
                    thrust = whole_move_sequence[current_move_index + timesteps_until_bullet_hit_asteroid + (-1 if not skip_half_of_first_cycle else 0)].thrust
                    turn_rate = whole_move_sequence[current_move_index + timesteps_until_bullet_hit_asteroid + (-1 if not skip_half_of_first_cycle else 0)].turn_rate
                    drag_amount = SHIP_DRAG*DELTA_TIME
                    if drag_amount > abs(bullet_sim_ship_state.speed):
                        bullet_sim_ship_state.speed = 0.0
                    else:
                        bullet_sim_ship_state.speed -= drag_amount*sign(bullet_sim_ship_state.speed)
                    # thrust = min(max(-SHIP_MAX_THRUST, thrust), SHIP_MAX_THRUST)
                    # Apply thrust
                    # print(bullet_sim_ship_state.speed)
                    bullet_sim_ship_state.speed += thrust*DELTA_TIME
                    # if ENABLE_ASSERTIONS:
                    #    assert -SHIP_MAX_SPEED <= bullet_sim_ship_state.speed <= SHIP_MAX_SPEED, f"In bullet sim, ship speed OoB: {bullet_sim_ship_state.speed}"
                    # bullet_sim_ship_state.speed = min(max(-SHIP_MAX_SPEED, bullet_sim_ship_state.speed), SHIP_MAX_SPEED)
                    # turn_rate = min(max(-SHIP_MAX_TURN_RATE, turn_rate), SHIP_MAX_TURN_RATE)
                    # Update the angle based on turning rate
                    bullet_sim_ship_state.heading += turn_rate*DELTA_TIME
                    # Keep the angle within (0, 360)
                    bullet_sim_ship_state.heading %= 360.0
                    # Use speed magnitude to get velocity vector
                    rad_heading = radians(bullet_sim_ship_state.heading)
                    bullet_sim_ship_state.velocity = (cos(rad_heading)*bullet_sim_ship_state.speed, sin(rad_heading)*bullet_sim_ship_state.speed)
                    # Update the position based off the velocities
                    # Do the wrap in the same operation
                    bullet_sim_ship_state.position = ((bullet_sim_ship_state.position[0] + bullet_sim_ship_state.velocity[0]*DELTA_TIME) % self.game_state.map_size[0], (bullet_sim_ship_state.position[1] + bullet_sim_ship_state.velocity[1]*DELTA_TIME) % self.game_state.map_size[1])

            # Check bullet/asteroid collisions
            bullet_remove_idxs = []
            #asteroid_remove_idxs: set[i64] = set()
            len_bullets = len(bullets)
            for b_idx, b in enumerate(chain(bullets, [my_bullet] if my_bullet is not None else [])):
                b_tail = (b.position[0] + b.tail_delta[0], b.position[1] + b.tail_delta[1])
                for a_idx, a in enumerate(asteroids):
                    if a_idx in asteroid_remove_idxs:
                        continue
                    # If collision occurs
                    if asteroid_bullet_collision(b.position, b_tail, a.position, a.radius):
                        if b_idx == len_bullets:
                            # This bullet is my bullet!
                            #bullet_sim_time += time.perf_counter() - start_time
                            return a, timesteps_until_bullet_hit_asteroid, ship_not_collided_with_asteroid
                        else:
                            # Mark bullet for removal
                            bullet_remove_idxs.append(b_idx)
                            # Create asteroid splits and mark it for removal
                            if a.size != 1:
                                asteroids.extend(forecast_instantaneous_asteroid_bullet_splits_from_velocity(a, b.velocity, self.game_state))
                            asteroid_remove_idxs.add(a_idx)
                            # Stop checking this bullet
                            break
            # Remove bullets
            if bullet_remove_idxs:
                bullets = [bullet for idx, bullet in enumerate(bullets) if idx not in bullet_remove_idxs]

            # Check mine/asteroid collisions
            mine_remove_idxs: list[i64] = []
            new_asteroids: list[Asteroid] = []
            for m_idx, mine in enumerate(mines):
                if mine.remaining_time < EPS:
                    # Mine is detonating
                    mine_remove_idxs.append(m_idx)
                    for a_idx, asteroid in enumerate(asteroids):
                        if a_idx in asteroid_remove_idxs:
                            continue
                        #if check_collision(asteroid.position[0], asteroid.position[1], asteroid.radius, mine.position[0], mine.position[1], MINE_BLAST_RADIUS):
                        delta_x = asteroid.position[0] - mine.position[0]
                        delta_y = asteroid.position[1] - mine.position[1]
                        separation = asteroid.radius + MINE_BLAST_RADIUS
                        if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                            if asteroid.size != 1:
                                new_asteroids.extend(forecast_asteroid_mine_instantaneous_splits(asteroid, mine, self.game_state))
                            asteroid_remove_idxs.add(a_idx)
            if mine_remove_idxs:
                mines = [mine for idx, mine in enumerate(mines) if idx not in mine_remove_idxs]
            asteroids.extend(new_asteroids)

            # Check ship/asteroid collisions
            if ship_not_collided_with_asteroid:
                if whole_move_sequence is not None:
                    assert bullet_sim_ship_state is not None
                    ship_position = bullet_sim_ship_state.position
                elif ship_state is not None:
                    ship_position = ship_state.position
                else:
                    ship_position = self.ship_state.position
                for a_idx, asteroid in enumerate(asteroids):
                    if a_idx in asteroid_remove_idxs:
                        continue
                    #if check_collision(ship_position[0], ship_position[1], SHIP_RADIUS, asteroid.position[0], asteroid.position[1], asteroid.radius):
                    delta_x = ship_position[0] - asteroid.position[0]
                    delta_y = ship_position[1] - asteroid.position[1]
                    separation = SHIP_RADIUS + asteroid.radius
                    if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                        # Note that we assume the ship is stationary. In reality, we don't know how the ship will move during the bullet sim. Therefore, these forecasted asteroids may not be accurate if the ship moves.
                        if asteroid.size != 1:
                            asteroids.extend(forecast_asteroid_ship_splits(asteroid, 0, (0.0, 0.0), self.game_state))
                        asteroid_remove_idxs.add(a_idx)
                        # Stop checking this ship's collisions. And also return saying the ship took damage!
                        #print('ship ast coll!')
                        ship_not_collided_with_asteroid = False
                        break
            # Cull asteroids marked for removal
            if len(asteroid_remove_idxs) > 2:
                asteroids = [asteroid for idx, asteroid in enumerate(asteroids) if idx not in asteroid_remove_idxs]
                asteroid_remove_idxs.clear()

    def apply_move_sequence(self, move_sequence: list[Action], allow_firing: bool = False) -> bool:
        # assert isclose(self.ship_state.speed, 0.0), f"When starting in apply move sequence where the sim was safe, the ship speed is not zero! {self.ship_state.speed=}, {self.ship_state.velocity=}. The whole move sequence is {move_sequence}"
        sim_was_safe = True
        if not is_close(self.ship_state.speed, 0.0):
            print(f"When starting in apply move sequence where the sim was safe, the ship speed is not zero! {self.ship_state.speed=}, {self.ship_state.velocity=}. The whole move sequence is {move_sequence}")
        for move in move_sequence:
            #thrust = 0.0
            #turn_rate = 0.0
            #fire = False
            # drop_mine = False
            #if 'thrust' in move:
            thrust = move.thrust
            #if 'turn_rate' in move:
            turn_rate = move.turn_rate
            #if 'fire' in move:
            fire = move.fire if not allow_firing else None
            # if 'drop_mine' in move:
            #    drop_mine = move.drop_mine # TODO: Implement
            # print(f"Calling update from apply move sequence:")
            if not self.update(thrust, turn_rate, fire):
                sim_was_safe = False
                break
            # print(f"After thrusting {thrust}: {self.ship_state.speed=}")
        return sim_was_safe

    def simulate_maneuver(self, move_sequence: list[Action], allow_firing: bool) -> bool:
        self.intended_move_sequence = move_sequence  # Record down the intended move sequence, so if I crash and the recorded move sequence gets cut short, we still have the intended move sequence!
        #flag = False
        #if not is_close_to_zero(self.ship_state.speed) and self.sim_id == 333:
        #    print(f"When starting in simulate maneuver where the sim was safe, the ship speed is not zero! {self.ship_state.speed=}, {self.ship_state.velocity=}. The whole move sequence is REDACTED move_sequence")
            #flag = True
        for move in move_sequence:
            thrust = 0.0
            turn_rate = 0.0
            #if 'thrust' in move:
            thrust = move.thrust
            #if 'turn_rate' in move:
            turn_rate = move.turn_rate
            # print(f"Calling update from sim maneuver:")
            #if self.sim_id == 23215:
                #print(f"Calling update from sim maneuver: with {thrust=}, {turn_rate=}, {allow_firing=}")
            if not self.update(thrust, turn_rate, None if allow_firing else False, move_sequence):
                return False
            #if flag:
            #if self.sim_id == 333:
            #    print(f"In sim {self.sim_id} After thrusting by {thrust} the true simmed ship speed is {self.ship_state.speed}")
        return True

    def update(self, thrust: float = 0.0, turn_rate: float = 0.0, fire: Optional[bool] = None, whole_move_sequence: Optional[list[Action]] = None, wait_out_mines: bool = False) -> bool:
        global total_sim_timesteps
        total_sim_timesteps += 1
        #if fire is not None and not wait_out_mines:
        #    print(f"Calling update in sim {self.sim_id} on future ts {self.future_timesteps} with fire {fire}")
        #global sim_update_total_time, sim_cull_total_time
        #if random.random() < 0.002:
        #    raise Exception("Bad luck exception!")
        #start_time = time.perf_counter()
        # This should exactly match what kessler_game.py does.
        # Being even one timestep off is the difference between life and death!!!
        return_value: Optional[bool] = None
        '''
        if not wait_out_mines:
            if not PRUNE_SIM_STATE_SEQUENCE or self.future_timesteps == 0:
                self.state_sequence.append(cast(SimState, {'timestep': self.initial_timestep + self.future_timesteps, 'ship_state': copy.copy(self.ship_state), 'game_state': self.get_game_state(), 'asteroids_pending_death': dict(self.asteroids_pending_death), 'forecasted_asteroid_splits': [copy.copy(a) for a in self.forecasted_asteroid_splits]}))
            else:
                self.state_sequence.append(cast(SimState, {'timestep': self.initial_timestep + self.future_timesteps, 'ship_state': copy.copy(self.ship_state)}))
        '''
        if not wait_out_mines:
            if PRUNE_SIM_STATE_SEQUENCE and self.future_timesteps != 0:
                # Create a super lightweight state that omits unnecessary stuff
                self.state_sequence.append(SimState(
                    timestep=self.initial_timestep + self.future_timesteps,
                    ship_state=self.ship_state.copy(),
                    # Assuming game_state and other attributes are optional or have default values in SimState definition
                ))
            else:
                self.state_sequence.append(SimState(
                    timestep=self.initial_timestep + self.future_timesteps,
                    ship_state=self.ship_state.copy(),
                    game_state=self.get_game_state(),
                    asteroids_pending_death=dict(self.asteroids_pending_death),
                    forecasted_asteroid_splits=[a.copy() for a in self.forecasted_asteroid_splits]
                ))
        # The simulation starts by evaluating actions and dynamics of the current present timestep, and then steps into the future
        # The game state we're given is actually what we had at the end of the previous timestep
        # The game will take the previous state, and apply current actions and then update to get the result of this timestep
        # Simulation order:
        # Ships are given the game state from after the previous timestep. Ships then decide the inputs.
        # Update bullets/mines/asteroids.
        # Ship has inputs applied and updated. Any new bullets and mines that the ship creates is added to the list.
        # Bullets past the map edge get culled
        # Ships and asteroids beyond the map edge get wrapped
        # Check for bullet/asteroid collisions. Checked for each bullet in list order, check for each asteroid in list order. Bullets and asteroids are removed here, and any new asteroids created are added to the list now.
        # Check mine collisions with asteroids/ships. For each mine in list order, check whether it is detonating and if it collides with first asteroids in list order (and add new asteroids to list), and then ships.
        # Check for asteroid/ship collisions. For each ship in list order, check collisions with each asteroid in list order. New asteroids are added now. Ships and asteroids get culled if they collide.
        # Check ship/ship collisions and cull them.


        asteroid_remove_idxs = set()

        # Simulate dynamics of bullets
        # Kessler will move bullets and cull them in different steps, but we combine them in one operation here
        # So we need to detect when the bullets are crossing the boundary, and delete them if they try to
        # Enumerate and track indices to delete
        bullet_remove_idxs = []
        for b_ind, b in enumerate(self.game_state.bullets):
            new_bullet_pos = (b.position[0] + b.velocity[0]*DELTA_TIME, b.position[1] + b.velocity[1]*DELTA_TIME)
            if check_coordinate_bounds(self.game_state, new_bullet_pos[0], new_bullet_pos[1]):
                b.position = new_bullet_pos
            else:
                bullet_remove_idxs.append(b_ind)
        if bullet_remove_idxs:
            self.game_state.bullets = [bullet for idx, bullet in enumerate(self.game_state.bullets) if idx not in bullet_remove_idxs]

        # Update mines
        for m in self.game_state.mines:
            m.remaining_time -= DELTA_TIME
            # If the timer is below eps, it'll detonate this timestep
        # Simulate dynamics of asteroids
        # Wrap the asteroid positions in the same operation
        # Between when the asteroids get moved and when the future timesteps gets incremented, these asteroids exist at time (self.initial_timestep + self.future_timesteps + 1) instead of (self.initial_timestep + self.future_timesteps)!
        for a in self.game_state.asteroids:
            a.position = ((a.position[0] + a.velocity[0]*DELTA_TIME) % self.game_state.map_size[0], (a.position[1] + a.velocity[1]*DELTA_TIME) % self.game_state.map_size[1])
        if not wait_out_mines:
            self.forecasted_asteroid_splits = maintain_forecasted_asteroids(self.forecasted_asteroid_splits, self.game_state)
            # Simulate the ship!
            # Bullet firing happens before we turn the ship
            # Check whether we want to shoot a simulated bullet
            if self.ship_state.bullets_remaining != 0:
                if self.fire_first_timestep and self.future_timesteps == 0:
                    # In theory we should be able to hit the target, however if we're in multiagent mode, the other ship could muddle with things in this time making me miss my shot, so let's just confirm that it's going to land before we fire for real!
                    # if len(self.other_ships) != 0:
                    if self.verify_first_shot:
                        actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, ship_was_safe = self.bullet_sim(None, False, 0, True, self.future_timesteps, whole_move_sequence)
                        # I think this assertion doesn't work right after a ship dies, because their bullet can still be travelling in the air
                        # if len(self.other_ships) == 0:
                        #    assert actual_asteroid_hit is not None
                        if actual_asteroid_hit is None:
                            print(f"SURPRISINGLY WE DON'T HIT ANYTHING LIKE WE THOUGHT WE WOULD!")
                            fire_this_timestep = False
                            self.cancel_firing_first_timestep = True
                        else:
                            # print(f"FIRE FIRST TIMESTEP CONFIRMED WILL HIT!!!")
                            fire_this_timestep = True
                    else:
                        fire_this_timestep = True
                elif fire is None:
                    #global update_ts_zero_count
                    # We're able to decide whether we want to fire any convenient shots we can get
                    timesteps_until_can_fire: i64 = max(0, FIRE_COOLDOWN_TS - (self.initial_timestep + self.future_timesteps - self.last_timestep_fired))
                    #if self.sim_id == 88227:
                    #    print(f'HELLO were on future ts {self.future_timesteps} in sim 88227 and fire is None. {self.halt_shooting=} {self.respawn_maneuver_pass_number=}, ts until can fire: {timesteps_until_can_fire}, The prescribed thrust and turn rate are {thrust} {turn_rate}')
                    fire_this_timestep = False
                    ship_heading_rad = radians(self.ship_state.heading)
                    avoid_targeting_this_asteroid: bool
                    if not self.halt_shooting or (self.respawn_maneuver_pass_number == 2 and self.initial_timestep + self.future_timesteps > self.last_timestep_colliding):
                        if timesteps_until_can_fire == 0:
                            #if self.sim_id == 88227:
                            #    print("We can shoot this TS!")
                            #print(f"\ntimesteps_until_can_fire == 0, looping through ALL ASTS!")
                            # We can shoot this timestep! Loop through all asteroids and see which asteroids we can feasibly hit if we shoot at this angle, and take those and simulate with the bullet sim to see which we'll hit
                            # If mines exist, then we can't cull any asteroids since asteroids can be hit by the mine and get flung into the path of my shooting
                            max_interception_time = 0.0
                            culled_targets_for_simulation: list[Asteroid]
                            culled_target_idxs_for_simulation: list[i64] = []
                            feasible_targets_exist: bool = False
                            min_shot_heading_error_rad = inf # Keep track of this so we can begin turning roughly toward our next target
                            second_min_shot_heading_error_rad = inf
                            len_asteroids = len(self.game_state.asteroids)
                            for ast_idx, asteroid in enumerate(chain(self.game_state.asteroids, self.forecasted_asteroid_splits)):
                                # Loop through ALL asteroids and make sure at least one asteroid is a valid target
                                # Get the length of time the longest asteroid would take to hit, and that'll be the upper bound of the bullet sim's timesteps
                                # Avoid shooting my size 1 asteroids that are about to get mined by my mine
                                avoid_targeting_this_asteroid = False
                                if asteroid.size == 1:
                                    for m in self.game_state.mines:
                                        if m.position in self.mine_positions_placed:
                                            # This mine is mine
                                            #project_asteroid_by_timesteps_num = round(m.remaining_time*FPS)
                                            #asteroid_when_mine_explodes = time_travel_asteroid(asteroid, project_asteroid_by_timesteps_num, self.game_state)
                                            asteroid_when_mine_explodes = time_travel_asteroid_s(asteroid, m.remaining_time, self.game_state)
                                            #if check_collision(asteroid_when_mine_explodes.position[0], asteroid_when_mine_explodes.position[1], asteroid_when_mine_explodes.radius, m.position[0], m.position[1], MINE_BLAST_RADIUS):
                                            delta_x = asteroid_when_mine_explodes.position[0] - m.position[0]
                                            delta_y = asteroid_when_mine_explodes.position[1] - m.position[1]
                                            separation = asteroid_when_mine_explodes.radius + MINE_BLAST_RADIUS
                                            if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                                                avoid_targeting_this_asteroid = True
                                                break
                                if avoid_targeting_this_asteroid:
                                    continue
                                if ast_idx < len_asteroids:
                                    #ast_angle = super_fast_atan2(asteroid.position[1] - self.ship_state.position[1], asteroid.position[0] - self.ship_state.position[0])
                                    #if abs(angle_difference_deg(degrees(ast_angle), self.ship_state.heading)) <= MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF:
                                    if heading_diff_within_threshold(ship_heading_rad, asteroid.position[0] - self.ship_state.position[0], asteroid.position[1] - self.ship_state.position[1], MANEUVER_BULLET_SIM_CULLING_CONE_WIDTH_ANGLE_HALF_COSINE):
                                        # We also want to add the surrounding asteroids into the bullet sim, just in case any of them aren't added later in the feasible shots
                                        # The reasons for them not being added later, is that maybe we already shot at it, so we skipped over it.
                                        # We should be including all the asteroids we shot at, but unfortunately even including all the asteroids in a cone doesn't guarantee that, so this system still isn't perfect!!
                                        culled_target_idxs_for_simulation.append(ast_idx)
                                check_next_asteroid = False
                                if check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + 1, self.game_state, asteroid):
                                    for a in unwrap_asteroid(asteroid, self.game_state.map_size[0], self.game_state.map_size[1], UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON, True):
                                        if check_next_asteroid:
                                            break
                                        # Since we need to find the minimum shot heading errors, we can't break out of this loop early. We should just go through them all.
                                        #unwrapped_ast_angle = super_fast_atan2(a.position[1] - self.ship_state.position[1], a.position[0] - self.ship_state.position[0])
                                        #if abs(angle_difference_deg(degrees(unwrapped_ast_angle), self.ship_state.heading)) > MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF:
                                        if not heading_diff_within_threshold(ship_heading_rad, a.position[0] - self.ship_state.position[0], a.position[1] - self.ship_state.position[1], MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF_COSINE):
                                            continue
                                        # feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist_during_interception
                                        feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, _, _, _ = calculate_interception(self.ship_state.position[0], self.ship_state.position[1], a.position[0], a.position[1], a.velocity[0], a.velocity[1], a.radius, self.ship_state.heading, self.game_state)
                                        if feasible:
                                            # Regardless of whether our heading is close enough to shooting this asteroid, keep track of this, just in case no asteroids are within shooting range this timestep, but we can begin to turn toward it next timestep!
                                            if abs(shot_heading_error_rad) < abs(min_shot_heading_error_rad):
                                                second_min_shot_heading_error_rad = min_shot_heading_error_rad
                                                min_shot_heading_error_rad = shot_heading_error_rad
                                            if abs(shot_heading_error_rad) <= shot_heading_tolerance_rad:
                                                # If we shoot at our current heading, this asteroid can be hit!
                                                if ast_idx < len_asteroids:
                                                    # Only add real asteroids to the set of asteroids we simulate! Don't simulate the asteroids that don't exist yet.
                                                    #culled_targets_for_simulation.append(asteroid)
                                                    if not culled_target_idxs_for_simulation or culled_target_idxs_for_simulation[-1] != ast_idx:
                                                        culled_target_idxs_for_simulation.append(ast_idx)
                                                feasible_targets_exist = True
                                                #print(f"Adding feasible target idx {ast_idx}, while the max idx for a real ast is {len(self.game_state.asteroids) - 1}")
                                                if interception_time > max_interception_time:
                                                    max_interception_time = interception_time
                                                check_next_asteroid = True
                                                break
                            if feasible_targets_exist:
                                #if self.sim_id == 88227:
                                #    print(f"Feasible targets exist! {culled_target_idxs_for_simulation=}")
                                # Use the bullet sim to confirm that this will hit something
                                # There's technically a chance for culled_targets_for_simulation to be empty at this point, if we're purely shooting asteroids that haven't come into existence yet. In that case, this will detect that and will avoid doing the culling, and do the full sim. This should be rare.
                                #culled_targets_for_simulation = [asteroid for ast_idx, asteroid in enumerate(self.game_state.asteroids) if ast_idx in culled_target_idxs_for_simulation]
                                #print(f"{len(self.game_state.asteroids)=}, {len(culled_target_idxs_for_simulation)=}, ratio={len(self.game_state.asteroids)/len(culled_target_idxs_for_simulation)}")

                                '''
                                # Convert the list to a set to remove any duplicates
                                unique_set = set(culled_target_idxs_for_simulation)

                                # Convert the set back to a list
                                unique_list = list(unique_set)

                                # Sort both lists in ascending order
                                unique_list.sort()

                                # Verify that both lists are the same and their lengths are the same
                                assert culled_target_idxs_for_simulation == unique_list, "Lists are not identical or contain duplicates"
                                assert len(culled_target_idxs_for_simulation) == len(unique_list), "List lengths are not equal"
                                '''

                                culled_targets_for_simulation = [self.game_state.asteroids[ast_idx] for ast_idx in culled_target_idxs_for_simulation]
                                #if not culled_targets_for_simulation:
                                #    print("WARNING: culled_targets_for_simulation is empty, so I think this means we're purely shooting at forecasted asteroid splits. Doing the full sim with all the bullets without the culling.")
                                    #raise Exception()
                                bullet_sim_timestep_limit = ceil(max_interception_time*FPS) + 1 # TODO: Might not need +1, but maybe it's safer to have it anyway at the cost of a tiny bit of performance
                                actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, ship_was_safe = self.bullet_sim(None, False, 0, True, self.future_timesteps, whole_move_sequence, bullet_sim_timestep_limit, culled_targets_for_simulation if (culled_targets_for_simulation and not self.game_state.mines) else None)
                                if actual_asteroid_hit is not None and ship_was_safe:
                                    # Confirmed that the shot will land
                                    assert timesteps_until_bullet_hit_asteroid is not None
                                    actual_asteroid_hit_at_fire_time = time_travel_asteroid(actual_asteroid_hit, -timesteps_until_bullet_hit_asteroid, self.game_state)
                                    if check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + 1, self.game_state, actual_asteroid_hit_at_fire_time):
                                        #if self.sim_id == 13939:
                                        #    print(f"Checked on ts {self.initial_timestep + self.future_timesteps + 1} that the asteroid {actual_asteroid_hit_at_fire_time} isn't tracked, so we can shoot it!")
                                        fire_this_timestep = True
                                        self.asteroids_shot += 1
                                        self.explanation_messages.append("During the maneuver, I conveniently shot asteroids.")
                                        if actual_asteroid_hit_at_fire_time.size != 1:
                                            #print("Calling forecast ast bull splits from maneuver")
                                            self.forecasted_asteroid_splits.extend(forecast_asteroid_bullet_splits_from_heading(actual_asteroid_hit_at_fire_time, timesteps_until_bullet_hit_asteroid, self.ship_state.heading, self.game_state))
                                        # The reason we add one to the timestep we track on, is that once we updated the asteroids' position in the update loop, it's technically the asteroid positions in the game state of the next timestep that gets passed to the controllers!
                                        # So the asteroid positions at a certain timestep is before their positions get updated. After updating, it's the next timestep.
                                        #if is_close(-10.0, actual_asteroid_hit_at_fire_time.velocity[0]):
                                        #    print(f"\nSHOT AT THE ASTEROID IN UPDATE in sim {self.sim_id} on timestep {self.initial_timestep + self.future_timesteps + 1}, {self.initial_timestep=} {self.future_timesteps=} and this shot will take this many ts: {timesteps_until_bullet_hit_asteroid}")
                                        track_asteroid_we_shot_at(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + 1, self.game_state, timesteps_until_bullet_hit_asteroid, actual_asteroid_hit_at_fire_time)
                                    if not isinf(self.game_state.time_limit) and self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid > floor(FPS*self.game_state.time_limit):
                                        # Added one to the timesteps to prevent off by one :P
                                        #print(f'WITHHOLDING SHOT BECAUSE SCENARIO IS ENDING, self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid + 1 = {self.initial_timestep + self.future_timesteps + timesteps_until_bullet_hit_asteroid + 1}, {self.game_state.time_limit=}, {FPS*self.game_state.time_limit=}')
                                        fire_this_timestep = False
                            #if self.sim_id == 88227:
                            #    print(f"Rip, we didn't lock in this ts. {fire_this_timestep=} {min_shot_heading_error_rad=}, {self.respawn_maneuver_pass_number=}")
                            if self.respawn_maneuver_pass_number == 0 and (self.future_timesteps >= MANEUVER_SIM_DISALLOW_TARGETING_FOR_START_TIMESTEPS_AMOUNT):# or not fire_this_timestep):
                                # Might as well start turning toward our next target!
                                if not fire_this_timestep and not isinf(min_shot_heading_error_rad):
                                    # We didn't fire this timestep, so we can use the min shot heading error rad to turn toward
                                    next_target_heading_error = min_shot_heading_error_rad
                                elif fire_this_timestep and not isinf(second_min_shot_heading_error_rad):
                                    next_target_heading_error = second_min_shot_heading_error_rad
                                else:
                                    next_target_heading_error = nan
                                # The assumption is that the target that was hit wasn't the second smallest heading diff. THIS IS NOT TRUE IN GENERAL. This can be wrong! But whatever, it's not a big deal and probably not worth fixing/taking the extra compute to track this.
                                #if self.sim_id == 88227:
                                #    print(f'HELLO were on future ts {self.future_timesteps} in sim 88227')
                                if not isnan(next_target_heading_error):
                                    min_shot_heading_error_deg = degrees(next_target_heading_error)
                                    if abs(min_shot_heading_error_deg) <= 6.0:
                                        altered_turn_command = min_shot_heading_error_deg*FPS
                                    else:
                                        altered_turn_command = SHIP_MAX_TURN_RATE*sign(min_shot_heading_error_rad)
                                    
                                    turn_rate = altered_turn_command
                                    if whole_move_sequence:
                                        whole_move_sequence[self.future_timesteps].turn_rate = altered_turn_command
                        elif timesteps_until_can_fire <= 3 and self.respawn_maneuver_pass_number == 0 and (self.future_timesteps >= MANEUVER_SIM_DISALLOW_TARGETING_FOR_START_TIMESTEPS_AMOUNT or timesteps_until_can_fire == 1):
                            # if timesteps_until_can_fire == 1, then we're able to fire on the next timestep. We can use this timestep to aim for an asteroid, and we have 6 degrees we're able to turn our ship to do that
                            # On the next timestep, hopefully we'd be aimed at the asteroid and then the above if case will kick in and we will shoot it!
                            # This makes the shot efficiency during maneuvering a lot better because we're not only dodging, but we're also targetting and firing at the same time!
                            # If there's more timesteps to turn before we can fire, then we can still begin to turn toward the closest target
                            locked_in = False
                            #global update_ts_multiple_count
                            asteroid_least_shot_heading_error = inf
                            asteroid_least_shot_heading_tolerance_deg = nan
                            
                            # Roughly predict the ship's position on the next timestep using its current heading. This isn't 100% correct but whatever, it's better than nothing.
                            ship_pred_speed = self.ship_state.speed
                            drag_amount = SHIP_DRAG*DELTA_TIME
                            if drag_amount > abs(ship_pred_speed):
                                ship_pred_speed = 0.0
                            else:
                                ship_pred_speed -= drag_amount*sign(ship_pred_speed)
                            # Apply thrust
                            ship_pred_speed += min(max(-SHIP_MAX_THRUST, thrust), SHIP_MAX_THRUST)*DELTA_TIME
                            if ship_pred_speed > SHIP_MAX_SPEED:
                                ship_pred_speed = SHIP_MAX_SPEED
                            elif ship_pred_speed < -SHIP_MAX_SPEED:
                                ship_pred_speed = -SHIP_MAX_SPEED
                            rad_heading = radians(self.ship_state.heading)
                            ship_speed_ts = DELTA_TIME*float(timesteps_until_can_fire)*ship_pred_speed
                            ship_predicted_pos_x = self.ship_state.position[0] + ship_speed_ts*cos(rad_heading)
                            ship_predicted_pos_y = self.ship_state.position[1] + ship_speed_ts*sin(rad_heading)
                            for asteroid in chain(self.game_state.asteroids, self.forecasted_asteroid_splits):
                                # Avoid shooting my size 1 asteroids that are about to get mined by my mine
                                avoid_targeting_this_asteroid = False
                                if asteroid.size == 1:
                                    for m in self.game_state.mines:
                                        if m.position in self.mine_positions_placed:
                                            # This mine is mine
                                            #project_asteroid_by_timesteps_num = round(m.remaining_time*FPS)
                                            #asteroid_when_mine_explodes = time_travel_asteroid(asteroid, project_asteroid_by_timesteps_num, self.game_state)
                                            asteroid_when_mine_explodes = time_travel_asteroid_s(asteroid, m.remaining_time, self.game_state)
                                            #if check_collision(asteroid_when_mine_explodes.position[0], asteroid_when_mine_explodes.position[1], asteroid_when_mine_explodes.radius, m.position[0], m.position[1], MINE_BLAST_RADIUS):
                                            delta_x = asteroid_when_mine_explodes.position[0] - m.position[0]
                                            delta_y = asteroid_when_mine_explodes.position[1] - m.position[1]
                                            separation = asteroid_when_mine_explodes.radius + MINE_BLAST_RADIUS
                                            if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                                                avoid_targeting_this_asteroid = True
                                                break
                                if avoid_targeting_this_asteroid:
                                    continue
                                if check_whether_this_is_a_new_asteroid_for_which_we_do_not_have_a_pending_shot(self.asteroids_pending_death, self.initial_timestep + self.future_timesteps + 1, self.game_state, asteroid):
                                    for a in unwrap_asteroid(asteroid, self.game_state.map_size[0], self.game_state.map_size[1], UNWRAP_ASTEROID_TARGET_SELECTION_TIME_HORIZON, True):
                                        if locked_in:
                                            break
                                        #unwrapped_ast_angle = super_fast_atan2(a.position[1] - self.ship_state.position[1], a.position[0] - self.ship_state.position[0])
                                        #if abs(angle_difference_deg(degrees(unwrapped_ast_angle), self.ship_state.heading)) > MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF:
                                        if not heading_diff_within_threshold(ship_heading_rad, a.position[0] - self.ship_state.position[0], a.position[1] - self.ship_state.position[1], MANEUVER_CONVENIENT_SHOT_CHECKER_CONE_WIDTH_ANGLE_HALF_COSINE):
                                            continue
                                        # feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, intercept_x, intercept_y, asteroid_dist_during_interception
                                        feasible, shot_heading_error_rad, shot_heading_tolerance_rad, interception_time, _, _, _ = calculate_interception(ship_predicted_pos_x, ship_predicted_pos_y, a.position[0], a.position[1], a.velocity[0], a.velocity[1], a.radius, self.ship_state.heading, self.game_state, timesteps_until_can_fire)
                                        if feasible:
                                            shot_heading_error_deg = degrees(shot_heading_error_rad)
                                            shot_heading_tolerance_deg = degrees(shot_heading_tolerance_rad)
                                            if abs(shot_heading_error_deg) - shot_heading_tolerance_deg < abs(asteroid_least_shot_heading_error):
                                                asteroid_least_shot_heading_error = shot_heading_error_deg
                                                asteroid_least_shot_heading_tolerance_deg = shot_heading_tolerance_deg
                                            if abs(shot_heading_error_deg) - shot_heading_tolerance_deg <= 6.0:  # 6 = DELTA_TIME*SHIP_MAX_TURN_RATE
                                                #update_ts_multiple_count += 1
                                                locked_in = True
                                                if abs(shot_heading_error_deg) <= 6.0:
                                                    # We can turn directly to the target's center without needing the tolerance at all!
                                                    altered_turn_command = shot_heading_error_deg*FPS
                                                else:
                                                    altered_turn_command = SHIP_MAX_TURN_RATE*sign(shot_heading_error_deg)
                                                turn_rate = altered_turn_command
                                                if whole_move_sequence:
                                                    whole_move_sequence[self.future_timesteps].turn_rate = altered_turn_command
                                                break
                            if not locked_in:
                                # We can't turn all the way to something, but we can begin turning that way!
                                if not isinf(asteroid_least_shot_heading_error) and not isnan(asteroid_least_shot_heading_tolerance_deg):
                                    altered_turn_command = SHIP_MAX_TURN_RATE*sign(asteroid_least_shot_heading_error)
                                    turn_rate = altered_turn_command
                                    if whole_move_sequence:
                                        whole_move_sequence[self.future_timesteps].turn_rate = altered_turn_command
                else:
                    #print(f"FIRE IS PRESCRIBED TO BE {fire}")
                    if self.verify_maneuver_shots:
                        if fire:
                            # UNREACHABLE CODE! We never actually execute a prescribed fire.
                            raise Exception("This code should be unreachable currently")
                            actual_asteroid_hit, timesteps_until_bullet_hit_asteroid, ship_was_safe = self.bullet_sim(None, False, 0, True, self.future_timesteps, whole_move_sequence)
                            if actual_asteroid_hit is None:
                                print("\nWe expected to hit stuff, but it's a good thing we checked because we don't hit shit!")
                                fire_this_timestep = False
                            else:
                                print("\nVERIFIED THE SHOT WORKS")
                                fire_this_timestep = True
                        else:
                            fire_this_timestep = False
                    else:
                        fire_this_timestep = fire
            else:
                fire_this_timestep = False

            if fire_this_timestep:
                self.last_timestep_fired = self.initial_timestep + self.future_timesteps
                # Remove respawn cooldown if we were in it
                self.ship_state.is_respawning = False
                self.respawn_timer = 0.0
                # Create new bullets/mines
                if self.ship_state.bullets_remaining != -1:
                    self.ship_state.bullets_remaining -= 1
                rad_heading = radians(self.ship_state.heading)
                cos_heading = cos(rad_heading)
                sin_heading = sin(rad_heading)
                bullet_x = self.ship_state.position[0] + SHIP_RADIUS*cos_heading
                bullet_y = self.ship_state.position[1] + SHIP_RADIUS*sin_heading
                # Make sure the bullet isn't being fired out into the void
                if check_coordinate_bounds(self.game_state, bullet_x, bullet_y):
                    vx = BULLET_SPEED*cos_heading
                    vy = BULLET_SPEED*sin_heading
                    new_bullet = Bullet(
                        position=(bullet_x, bullet_y),
                        velocity=(vx, vy),
                        heading=self.ship_state.heading,
                        mass=BULLET_MASS,
                        tail_delta=(-BULLET_LENGTH*cos_heading, -BULLET_LENGTH*sin_heading)
                    )
                    self.game_state.bullets.append(new_bullet)

            # We can drop a mine once every 30 frames, so make sure we wait at least that long before we try dropping another one
            # It's expensive to check the mine FIS every timestep, so do it periodically as the ship moves around
            if self.ship_state.mines_remaining != 0 and self.last_timestep_mined <= self.initial_timestep + self.future_timesteps - MINE_COOLDOWN_TS - MINE_DROP_COOLDOWN_FUDGE_TS and not self.halt_shooting and self.future_timesteps % MINE_OPPORTUNITY_CHECK_INTERVAL_TS == 0:
                drop_mine_this_timestep = check_mine_opportunity(self.ship_state, self.game_state, self.other_ships)  # Read only access of the ship and game states
            else:
                drop_mine_this_timestep = False
            if drop_mine_this_timestep:
                self.sim_placed_a_mine = True
                self.last_timestep_mined = self.initial_timestep + self.future_timesteps
                # This doesn't check whether it's valid to place a mine! It just does it!
                self.explanation_messages.append("This is a good chance to drop a mine. Bombs away!")
                # Remove respawn cooldown if we were in it
                self.ship_state.is_respawning = False
                self.respawn_timer = 0.0
                # debug_print(f'BOMBS AWAY! Sim ID {self.sim_id}, future timesteps {self.future_timesteps}')
                new_mine = Mine(
                    position=self.ship_state.position,
                    mass=MINE_MASS,
                    fuse_time=MINE_FUSE_TIME,
                    remaining_time=MINE_FUSE_TIME
                )
                self.mine_positions_placed.add(self.ship_state.position) # Track where we placed our mine
                self.game_state.mines.append(new_mine)
                self.ship_state.mines_remaining -= 1

            # Update respawn timer
            if self.respawn_timer <= 0:
                self.respawn_timer = 0.0
            else:
                self.respawn_timer -= DELTA_TIME
            if not self.respawn_timer:
                self.ship_state.is_respawning = False
            # Simulate ship dynamics
            drag_amount = SHIP_DRAG*DELTA_TIME
            if drag_amount > abs(self.ship_state.speed):
                self.ship_state.speed = 0.0
            else:
                self.ship_state.speed -= drag_amount*sign(self.ship_state.speed)
            # Bounds check the thrust
            # TODO: REMOVE BOUNDS CHECKS
            # thrust = min(max(-SHIP_MAX_THRUST, thrust), SHIP_MAX_THRUST)
            # Apply thrust
            self.ship_state.speed += thrust*DELTA_TIME
            # turn_rate = min(max(-SHIP_MAX_TURN_RATE, turn_rate), SHIP_MAX_TURN_RATE)
            # Update the angle based on turning rate
            self.ship_state.heading += turn_rate*DELTA_TIME
            # Keep the angle within (0, 360)
            self.ship_state.heading %= 360.0
            # Use speed magnitude to get velocity vector
            rad_heading = radians(self.ship_state.heading)
            self.ship_state.velocity = (cos(rad_heading)*self.ship_state.speed, sin(rad_heading)*self.ship_state.speed)
            # Update the position based off the velocities
            # Do the wrap in the same operation
            self.ship_state.position = ((self.ship_state.position[0] + self.ship_state.velocity[0]*DELTA_TIME) % self.game_state.map_size[0], (self.ship_state.position[1] + self.ship_state.velocity[1]*DELTA_TIME) % self.game_state.map_size[1])
        
        # Check bullet/asteroid collisions
        bullet_remove_idxs = []
        for b_idx, b in enumerate(self.game_state.bullets):
            b_tail = (b.position[0] + b.tail_delta[0], b.position[1] + b.tail_delta[1])
            for a_idx, a in enumerate(self.game_state.asteroids):
                if a_idx in asteroid_remove_idxs:
                    continue
                # If collision occurs
                if asteroid_bullet_collision(b.position, b_tail, a.position, a.radius):
                    # Mark bullet for removal
                    bullet_remove_idxs.append(b_idx)
                    # Create asteroid splits and mark it for removal
                    if a.size != 1:
                        self.game_state.asteroids.extend(forecast_instantaneous_asteroid_bullet_splits_from_velocity(a, b.velocity, self.game_state))
                    asteroid_remove_idxs.add(a_idx)
                    # Stop checking this bullet
                    break
        # Cull bullets and asteroids that are marked for removal
        if bullet_remove_idxs:
            self.game_state.bullets = [bullet for idx, bullet in enumerate(self.game_state.bullets) if idx not in bullet_remove_idxs]
        if not wait_out_mines:
            self.ship_move_sequence.append(Action(timestep=self.initial_timestep + self.future_timesteps, thrust=thrust, turn_rate=turn_rate, fire=fire_this_timestep, drop_mine=drop_mine_this_timestep))

        # Check mine/asteroid and mine/ship collisions
        mine_remove_idxs: list[i64] = []
        # asteroid_remove_idxs = set() # Use a set, since this is the only case where we may have many asteroids removed at once
        new_asteroids: list[Asteroid] = []
        for idx_mine, mine in enumerate(self.game_state.mines):
            if mine.remaining_time < EPS:
                # Mine is detonating
                mine_remove_idxs.append(idx_mine)
                for a_idx, asteroid in enumerate(self.game_state.asteroids):
                    if a_idx in asteroid_remove_idxs:
                        continue
                    #if check_collision(asteroid.position[0], asteroid.position[1], asteroid.radius, mine.position[0], mine.position[1], MINE_BLAST_RADIUS):
                    delta_x = asteroid.position[0] - mine.position[0]
                    delta_y = asteroid.position[1] - mine.position[1]
                    separation = asteroid.radius + MINE_BLAST_RADIUS
                    if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                        if asteroid.size != 1:
                            new_asteroids.extend(forecast_asteroid_mine_instantaneous_splits(asteroid, mine, self.game_state))
                        asteroid_remove_idxs.add(a_idx)
                if not wait_out_mines:
                    if not self.ship_state.is_respawning:
                        #if check_collision(self.ship_state.position[0], self.ship_state.position[1], SHIP_RADIUS, mine.position[0], mine.position[1], MINE_BLAST_RADIUS):
                        delta_x = self.ship_state.position[0] - mine.position[0]
                        delta_y = self.ship_state.position[1] - mine.position[1]
                        separation = SHIP_RADIUS + MINE_BLAST_RADIUS
                        if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                            # Ship got hit by mine, RIP
                            #print(f"Ship got hit by mine in sim, RIP and the ship respawn state is {self.ship_state.is_respawning} with the respawn timer at {self.respawn_timer}")
                            return_value = False
                            self.ship_crashed = True
                            self.ship_state.lives_remaining -= 1
                            self.ship_state.is_respawning = True
                            self.ship_state.speed = 0.0
                            self.ship_state.velocity = (0.0, 0.0)
                            self.respawn_timer = 3.0
                    elif self.respawn_maneuver_pass_number == 1:
                        # Record the last timestep getting hit by a mine
                        delta_x = self.ship_state.position[0] - mine.position[0]
                        delta_y = self.ship_state.position[1] - mine.position[1]
                        separation = SHIP_RADIUS + MINE_BLAST_RADIUS
                        if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                            self.last_timestep_colliding = self.initial_timestep + self.future_timesteps
        if mine_remove_idxs:
            self.game_state.mines = [mine for idx, mine in enumerate(self.game_state.mines) if idx not in mine_remove_idxs]
        self.game_state.asteroids.extend(new_asteroids)

        if not wait_out_mines:
            # Check ship/asteroid collisions
            # This DOES NOT account for the other ship! Since we can't predict their behavior, I'll just assume the other ship doesn't exist instead of wrongly predicting stuff that doesn't end up happening.
            if not self.ship_state.is_respawning:
                # asteroid_remove_idxs = []
                for a_idx, asteroid in enumerate(self.game_state.asteroids):
                    if a_idx in asteroid_remove_idxs:
                        continue
                    #if check_collision(self.ship_state.position[0], self.ship_state.position[1], SHIP_RADIUS, asteroid.position[0], asteroid.position[1], asteroid.radius):
                    delta_x = self.ship_state.position[0] - asteroid.position[0]
                    delta_y = self.ship_state.position[1] - asteroid.position[1]
                    separation = SHIP_RADIUS + asteroid.radius
                    if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                        if asteroid.size != 1:
                            self.game_state.asteroids.extend(forecast_asteroid_ship_splits(asteroid, 0, self.ship_state.velocity, self.game_state))
                        asteroid_remove_idxs.add(a_idx)
                        # Stop checking this ship's collisions. And also return saying the ship took damage!
                        return_value = False
                        self.ship_crashed = True
                        self.ship_state.lives_remaining -= 1
                        self.ship_state.is_respawning = True
                        self.ship_state.speed = 0.0
                        self.ship_state.velocity = (0.0, 0.0)
                        self.respawn_timer = 3.0
                        break
            elif self.respawn_maneuver_pass_number == 1:
                for a_idx, asteroid in enumerate(self.game_state.asteroids):
                    if a_idx in asteroid_remove_idxs:
                        continue
                    #if check_collision(self.ship_state.position[0], self.ship_state.position[1], SHIP_RADIUS, asteroid.position[0], asteroid.position[1], asteroid.radius):
                    delta_x = self.ship_state.position[0] - asteroid.position[0]
                    delta_y = self.ship_state.position[1] - asteroid.position[1]
                    separation = SHIP_RADIUS + asteroid.radius
                    if abs(delta_x) <= separation and abs(delta_y) <= separation and delta_x*delta_x + delta_y*delta_y <= separation*separation:
                        # The ship and asteroid are overlapping, while the ship is doing a respawn maneuver.
                        # We want to track this, so that we can begin to shoot asteroids right after we're no longer overlapping with any, before we finish the respawn maneuver and get a couple more hits in
                        self.last_timestep_colliding = self.initial_timestep + self.future_timesteps
                        break
        # Cull asteroids marked for removal
        if asteroid_remove_idxs:
            #start_cull_time = time.perf_counter()
            self.game_state.asteroids = [asteroid for idx, asteroid in enumerate(self.game_state.asteroids) if idx not in asteroid_remove_idxs]
            #sim_cull_total_time += time.perf_counter() - start_cull_time
        
        if not wait_out_mines:
            # Checking collisions with the other ship isn't a great idea since this isn't 100% sure. This models the other ship as stationary, which probably won't be true, unless their controller is the null controller.
            '''
            # Check ship/ship collisions
            if not self.ship_state.is_respawning:
                if ENABLE_ASSERTIONS:
                    assert return_value is None
                if self.get_instantaneous_ship_collision():
                    print(f"COLLISION WITH OTHER SHIP!!!")
                    return_value = False
                    self.ship_crashed = True
                    self.ship_state.lives_remaining -= 1
                    self.ship_state.is_respawning = True
                    self.ship_state.speed = 0.0
                    self.ship_state.velocity = (0.0, 0.0)
                    self.respawn_timer = 3.0
            '''
            self.future_timesteps += 1
            self.game_state.sim_frame += 1
        #sim_update_total_time += time.perf_counter() - start_time
        if return_value is None:
            return True
        else:
            return return_value

    def rotate_heading(self, heading_difference_deg: float, shoot_on_first_timestep: bool = False) -> bool:
        target_heading = (self.ship_state.heading + heading_difference_deg) % 360.0
        still_need_to_turn = heading_difference_deg
        while abs(still_need_to_turn) > SHIP_MAX_TURN_RATE*DELTA_TIME + EPS:
            if not self.update(0.0, SHIP_MAX_TURN_RATE*sign(heading_difference_deg), shoot_on_first_timestep):
                return False
            shoot_on_first_timestep = False
            still_need_to_turn -= SHIP_MAX_TURN_RATE*sign(heading_difference_deg)*DELTA_TIME
        if not self.update(0.0, still_need_to_turn*FPS, shoot_on_first_timestep):
            return False
        return True

    def get_rotate_heading_move_sequence(self, heading_difference_deg: float, shoot_on_first_timestep: bool = False) -> list[Action]:
        move_sequence: list[Action] = []
        if abs(heading_difference_deg) < GRAIN:
            # We still need a null sequence here, so that we don't end up with a 0 frame maneuver!
            move_sequence.append(Action(thrust=0.0, turn_rate=0.0, fire=shoot_on_first_timestep))
            return move_sequence
        still_need_to_turn = heading_difference_deg
        while abs(still_need_to_turn) > SHIP_MAX_TURN_RATE*DELTA_TIME:
            move_sequence.append(Action(thrust=0.0, turn_rate=SHIP_MAX_TURN_RATE*sign(heading_difference_deg), fire=shoot_on_first_timestep))
            shoot_on_first_timestep = False
            still_need_to_turn -= SHIP_MAX_TURN_RATE*sign(heading_difference_deg)*DELTA_TIME
        if abs(still_need_to_turn) > EPS:
            move_sequence.append(Action(thrust=0.0, turn_rate=still_need_to_turn*FPS, fire=shoot_on_first_timestep))
        return move_sequence

    def accelerate(self, target_speed: float, turn_rate: float = 0.0) -> bool:
        #print(f"Accelerating to speed {target_speed} while our speed is already {self.ship_state.speed}")
        # Keep in mind speed can be negative
        # Drag will always slow down the ship
        while abs(self.ship_state.speed - target_speed) > EPS:
            drag = -SHIP_DRAG*sign(self.ship_state.speed)
            drag_amount = SHIP_DRAG*DELTA_TIME
            if drag_amount > abs(self.ship_state.speed):
                # The drag amount is reduced if it would make the ship cross 0 speed on its own
                adjust_drag_by = abs((drag_amount - abs(self.ship_state.speed))*FPS)
                drag -= adjust_drag_by*sign(drag)
            delta_speed_to_target = target_speed - self.ship_state.speed
            thrust_amount = delta_speed_to_target*FPS - drag
            #print(f"{thrust_amount=}, clamped thrust: {min(max(-SHIP_MAX_THRUST, thrust_amount), SHIP_MAX_THRUST)}, {self.ship_state.speed=}, {target_speed=}, {delta_speed_to_target=}")
            thrust_amount = min(max(-SHIP_MAX_THRUST, thrust_amount), SHIP_MAX_THRUST)

            if not self.update(thrust_amount, turn_rate):
                return False
        return True

    def cruise(self, cruise_time: i64, cruise_turn_rate: float = 0.0) -> bool:
        # Maintain current speed
        for _ in range(cruise_time):
            if not self.update(sign(self.ship_state.speed)*SHIP_DRAG, cruise_turn_rate):
                return False
        return True

    def get_move_sequence(self) -> list[Action]:
        return self.ship_move_sequence

    def get_intended_move_sequence(self) -> list[Action]:
        if self.intended_move_sequence:
            return self.intended_move_sequence
        else:
            return self.ship_move_sequence

    def get_state_sequence(self) -> list[SimState]:
        if self.state_sequence and self.state_sequence[-1].timestep != self.initial_timestep + self.future_timesteps:
            #print(f"In get state sequence, the final timestep was {self.state_sequence[-1]['timestep']} and we're appending the last state to make it {self.initial_timestep + self.future_timesteps}")
            #self.state_sequence.append(cast(SimState, {'timestep': self.initial_timestep + self.future_timesteps, 'ship_state': copy.copy(self.ship_state), 'game_state': self.get_game_state(), 'asteroids_pending_death': dict(self.asteroids_pending_death), 'forecasted_asteroid_splits': [copy.copy(a) for a in self.forecasted_asteroid_splits]}))
            self.state_sequence.append(SimState(
                timestep=self.initial_timestep + self.future_timesteps,
                ship_state=self.ship_state.copy(),
                game_state=self.get_game_state(),
                asteroids_pending_death=dict(self.asteroids_pending_death),
                forecasted_asteroid_splits=[a.copy() for a in self.forecasted_asteroid_splits]
            ))
        return self.state_sequence

    def get_sequence_length(self) -> i64:
        # debug_print(f"Length of move seq: {len(self.ship_move_sequence)}, length of state seq: {len(self.state_sequence)}")
        return len(self.state_sequence)

    def get_future_timesteps(self) -> i64:
        return self.future_timesteps

    def get_position(self) -> tuple[float, float]:
        return self.ship_state.position

    def get_last_timestep_fired(self) -> i64:
        return self.last_timestep_fired

    def get_last_timestep_mined(self) -> i64:
        return self.last_timestep_mined

    def get_velocity(self) -> tuple[float, float]:
        return self.ship_state.velocity

    def get_heading(self) -> float:
        return float(self.ship_state.heading)


class KesslerController:
    """
     A ship controller class for Kessler. This can be inherited to create custom controllers that can be passed to the
    game to operate within scenarios. A valid controller contains an actions method that takes in a ship object and ass
    game_state dictionary. This action method then sets the thrust, turn_rate, and fire commands on the ship object.
    """

    def actions(self, ship_state: dict[str, Any], game_state: dict[str, Any]) -> tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller.
        """

        raise NotImplementedError('Your derived KesslerController must include an actions method for control input.')


    # Property to store the ID for the ship this controller is attached to during a scenario
    @property
    def ship_id(self) -> int:
        return self._ship_id if self._ship_id else 0

    @ship_id.setter
    def ship_id(self, value: int) -> None:
        self._ship_id = value

    @property
    def name(self) -> str:
        raise NotImplementedError(f"This controller {self.__class__} needs to have a name() property specified.")


class NeoController(KesslerController):
    @property
    def name(self) -> str:
        return "Neo"

    def get_total_sim_ts(self) -> i64:
        return total_sim_timesteps

    def __init__(self, chromosome: Optional[tuple[float, float, float, float, float, float, float, float, float]] = None) -> None:
        self.reset(chromosome)
        #self.ship_id: int = -1 # Dangerous!
        #self._ship_id: int = -1

    def reset(self, chromosome: Optional[tuple[float, float, float, float, float, float, float, float, float]] = None) -> None:
        self.init_done = False
        #self.ship_id = None
        # DO NOT OVERWRITE self.ship_id. That will cause the controller to break, since Kessler manages that itself. If we want to track our ship id, use a different variable name.
        self.ship_id_internal: i64 = -1
        self.current_timestep: i64 = -1
        self.action_queue: deque[tuple[i64, float, float, bool, bool]] = deque()
        self.game_state_plotter: Optional[GameStatePlotter] = None
        self.actioned_timesteps: set[i64] = set()
        self.sims_this_planning_period: list[CompletedSimulation] = []  # The first sim in the list is stationary targetting, and the rest is maneuvers
        self.best_fitness_this_planning_period: float = -inf
        self.best_fitness_this_planning_period_index: Optional[i64] = None
        self.second_best_fitness_this_planning_period: float = -inf
        self.second_best_fitness_this_planning_period_index: Optional[i64] = None
        self.stationary_targetting_sim_index: Optional[i64] = None
        self.game_state_to_base_planning: Optional[BasePlanningGameState] = None
        self.base_gamestate_analysis: Optional[tuple[float, float, float, float, i64, float, i64, i64]] = None
        self.set_of_base_gamestate_timesteps: set[i64] = set()
        self.base_gamestates: dict[i64, Any] = {}  # Key is timestep, value is the state
        self.other_ships_exist: bool = False
        #self.reality_move_sequence: list[dict[str, Any]] = []
        self.simulated_gamestate_history: dict[i64, SimState] = {}
        self.lives_remaining_that_we_did_respawn_maneuver_for: set[i64] = set()
        self.last_timestep_ship_is_respawning = False
        if chromosome is not None:
            global fitness_function_weights
            fitness_function_weights = chromosome
        # For performance controller
        self.outside_controller_time_intervals: list[float] = []
        self.inside_controller_iteration_time_intervals: list[float] = []  # Stores recent iteration times for rolling average of PERFORMANCE_CONTROLLER_ROLLING_AVERAGE_FRAME_INTERVAL frames
        self.last_entrance_time: float = nan
        self.last_exit_time: float = nan
        self.last_iteration_start_time: float = nan
        self.average_iteration_time = DELTA_TIME*0.1

        # Clear globals
        global explanation_messages_with_timestamps
        explanation_messages_with_timestamps.clear()
        #global total_abs_cruise_speed
        #total_abs_cruise_speed = SHIP_MAX_SPEED/2
        #global total_cruise_timesteps
        #total_cruise_timesteps = round(MAX_CRUISE_TIMESTEPS/2)
        #global total_maneuvers_to_learn_from
        #total_maneuvers_to_learn_from = 1
        global abs_cruise_speeds, cruise_timesteps, unwrap_cache, total_sim_timesteps, overall_fitness_record
        abs_cruise_speeds = [SHIP_MAX_SPEED/2]
        cruise_timesteps = [round(MAX_CRUISE_TIMESTEPS/2)]
        overall_fitness_record.clear()
        unwrap_cache.clear()
        total_sim_timesteps = 0


    def finish_init(self, game_state: GameState, ship_state: Ship) -> None:
        # If we need the game state or ship state to finish init, we can use this function to do that
        if self.ship_id_internal == -1:
            self.ship_id_internal = ship_state.id
        if len(get_other_ships(game_state, self.ship_id_internal)) > 0:
            self.other_ships_exist = True
            print_explanation("I've got another ship friend here with me. I'll try coexisting with them, but be careful to avoid them.", self.current_timestep)
        else:
            self.other_ships_exist = False
            print_explanation("I'm alone. I can see into the future perfectly!", self.current_timestep)
        # asteroid_density = control.Antecedent(arange(0, 11, 1), 'asteroid_density')
        # asteroids_entropy = control.Antecedent(arange(0, 11, 1), 'asteroids_entropy')
        # other_ship_lives = control.Antecedent(arange(0, 4, 1), 'other_ship_lives')

        # aggression = control.Consequent(arange(0, 1, 1), 'asteroid_growth_factor')

    def enqueue_action(self, timestep: i64, thrust: float = 0.0, turn_rate: float = 0.0, fire: bool = False, drop_mine: bool = False) -> None:
        self.action_queue.append((timestep, thrust, turn_rate, fire, drop_mine))

    def performance_controller_enter(self) -> None:
        # Called when actions() is called
        current_entrance_time = time.perf_counter()
        if not isnan(self.last_exit_time):
            # Calculate time spent outside since last exit and update the rolling list
            outside_time = current_entrance_time - self.last_exit_time
            self.outside_controller_time_intervals.append(outside_time)
            # Keep only the last 5 elements for rolling average
            self.outside_controller_time_intervals = self.outside_controller_time_intervals[-PERFORMANCE_CONTROLLER_ROLLING_AVERAGE_FRAME_INTERVAL:]
        self.last_entrance_time = current_entrance_time

    def performance_controller_exit(self) -> None:
        # Called at the end of actions()
        exit_time = time.perf_counter()
        self.last_exit_time = exit_time
        # Close out the final iteration, assumed to have been started before this and ended just now
        last_iteration_end_time = time.perf_counter()
        # print(f"In perf controller exit: {last_iteration_end_time=}, {self.last_iteration_start_time=}")
        last_iteration_time_interval = last_iteration_end_time - self.last_iteration_start_time
        self.inside_controller_iteration_time_intervals.append(last_iteration_time_interval)
        # print(f"In perf controller exit. {self.inside_controller_iteration_time_intervals=}, {self.outside_controller_time_intervals=}")
        self.last_iteration_start_time = nan

    def performance_controller_start_iteration(self) -> None:
        # Call this around all iterations. Call it before each iteration, and after the last iteration. If iterations are the fences, this function is the fenceposts around each section of fence.
        if not isnan(self.last_iteration_start_time):
            # This is at least the second iteration run this timestep. Track how long the last iteration took.
            current_iteration_start_time = time.perf_counter()
            last_iteration_time_interval = current_iteration_start_time - self.last_iteration_start_time
            self.last_iteration_start_time = current_iteration_start_time
            self.inside_controller_iteration_time_intervals.append(last_iteration_time_interval)
            # Keep only the last 5 elements for rolling average
            self.inside_controller_iteration_time_intervals = self.inside_controller_iteration_time_intervals[-PERFORMANCE_CONTROLLER_ROLLING_AVERAGE_FRAME_INTERVAL:]
            # Update average iteration time based on the rolling list
            self.average_iteration_time = sum(self.inside_controller_iteration_time_intervals)/len(self.inside_controller_iteration_time_intervals)
        else:
            # This is before the first iteration run within this controller timestep.
            self.last_iteration_start_time = time.perf_counter()

    def performance_controller_check_whether_i_can_do_another_iteration(self) -> bool:
        # Called throughout my code to check whether I can squeeze in another search iteration
        if not isnan(self.last_iteration_start_time):
            # This is at least the second iteration run this timestep
            current_time = time.perf_counter()
            elapsed_time_inside = current_time - self.last_entrance_time
            average_outside_time = sum(self.outside_controller_time_intervals)/len(self.outside_controller_time_intervals) if len(self.outside_controller_time_intervals) > 0 else 0.0
            remaining_time_budget = max(DELTA_TIME*MINIMUM_DELTA_TIME_FRACTION_BUDGET - elapsed_time_inside, DELTA_TIME - average_outside_time - elapsed_time_inside)

            # Check if another iteration can fit within the remaining time budget
            if ENABLE_PERFORMANCE_CONTROLLER:
                if remaining_time_budget >= self.average_iteration_time*PERFORMANCE_CONTROLLER_PUSHING_THE_ENVELOPE_FUDGE_MULTIPLIER:
                    return True
                else:
                    return False
            else:
                # In deterministic mode, just never do additional iterations. This will also test that the minimum iterations are sufficient for a baseline level of strategic performance.
                # return False
                if random.random() < 0.8:
                    return True
                else:
                    return False
        else:
            # This is the first iteration run within this controller timestep. Let the iteration occur
            return True

    def decide_next_action(self, game_state: GameState, ship_state: Ship) -> None:
        assert self.game_state_to_base_planning is not None
        assert self.best_fitness_this_planning_period_index is not None
        #print(f"\nDeciding next action! We're picking out of {len(self.sims_this_planning_period)} total sims")
        # print([x['fitness'] for x in self.sims_this_planning_period])
        
        # all_ship_pos = []
        # all_ship_x = []
        # all_ship_y = []
        # for thing in self.sims_this_planning_period:
        #     state_seq = thing['sim'].get_state_sequence()
        #     ship_line_x = []
        #     ship_line_y = []
        #     for state in state_seq:
        #         ship_pos = state.ship_state.position
        #         all_ship_pos.append(ship_pos)
        #         ship_line_x.append(ship_pos[0])
        #         ship_line_y.append(ship_pos[1])
        #     all_ship_x.append(ship_line_x)
        #     all_ship_y.append(ship_line_y)
        # if len(all_ship_x) > 30:
        #     for i in range(len(all_ship_x)):
        #         plt.scatter(all_ship_x[i], all_ship_y[i], linewidths=1.0, label=f"Maneuver {i}")
        #     plt.xlim(0, 1000)
        #     plt.ylim(0, 800)
        #     plt.show()
        
        #time.sleep(10)
        # print(f"Deciding next action, Respawn maneuver status is: {self.game_state_to_base_planning['respawning']}")
        # Go through the list of planned maneuvers and pick the one with the best fitness function score
        # Update the state to base planning off of, so Neo can get to work on planning the next set of moves while this current set of moves executes
        # print('Going through sorted sims list to pick the best action')
        best_action_sim: Matrix
        if self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['state_type'] == 'predicted':
            # Since the game is non-deterministic, we need to apply our simulated moves onto the actual corrected state, so errors don't build up
            best_action_sim_predicted: Matrix = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['sim']
            # debug_print(f"\nPredicted best action sim first state:", best_action_sim_predicted.get_state_sequence()[0])
            best_action_fitness_predicted = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness']
            best_action_maneuver_tuple = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['maneuver_tuple']
            best_predicted_sim_fire_next_timestep_flag = best_action_sim_predicted.get_fire_next_timestep_flag()
            # debug_print(f"best_predicted_sim_fire_next_timestep_flag: {best_predicted_sim_fire_next_timestep_flag}, self.game_state_to_base_planning['fire_next_timestep_flag']: {self.game_state_to_base_planning['fire_next_timestep_flag']}")
            # self.game_state_to_base_planning['fire_next_timestep_flag'] is whether we fire at the BEGINNING of the period, while best_action_sim_predicted.get_fire_next_timestep_flag() is whether we fire AFTER this period
            # debug_print('DECIDE NEXT ACTION REDO sim ast pending death:')
            # debug_print(self.game_state_to_base_planning['asteroids_pending_death'])
            #print("Doing a best action sim")
            best_action_sim = Matrix(game_state=game_state,
                                     ship_state=ship_state,
                                     initial_timestep=self.current_timestep,
                                     respawn_timer=self.game_state_to_base_planning['ship_respawn_timer'],
                                     asteroids_pending_death=self.game_state_to_base_planning['asteroids_pending_death'],
                                     forecasted_asteroid_splits=self.game_state_to_base_planning['forecasted_asteroid_splits'],
                                     last_timestep_fired=self.game_state_to_base_planning['last_timestep_fired'],
                                     last_timestep_mined=self.game_state_to_base_planning['last_timestep_mined'],
                                     mine_positions_placed=self.game_state_to_base_planning['mine_positions_placed'],
                                     halt_shooting=self.game_state_to_base_planning['respawning'],
                                     fire_first_timestep=self.game_state_to_base_planning['fire_next_timestep_flag'],
                                     verify_first_shot=True,
                                     verify_maneuver_shots=True,
                                     last_timestep_colliding=best_action_sim_predicted.get_last_timestep_colliding() if self.game_state_to_base_planning['respawning'] else -1,
                                     game_state_plotter=self.game_state_plotter)
            best_action_sim_predicted_move_sequence = best_action_sim_predicted.get_intended_move_sequence()
            #print(f"\nIntended move sequence:")
            #print(best_action_sim_predicted_move_sequence)
            #print(f"Actual move sequence:")
            #print(best_action_sim_predicted.get_move_sequence())
            #print(f"Applying move sequence for maneuver. Seq is {best_action_sim_predicted_move_sequence}")
            if self.game_state_to_base_planning['respawning']:
                #print("Applying respawning move sequence")
                best_action_sim.apply_move_sequence(best_action_sim_predicted_move_sequence, True)
            else:
                # This can only be maneuvering, and this gives the freedom to aim and shoot during the maneuver
                #print(f'\ncalling sim maneuver in decide best with intended move sequence of {best_action_sim_predicted_move_sequence} and actual move sequence of {best_action_sim_predicted.get_move_sequence()}')
                best_action_sim.simulate_maneuver(best_action_sim_predicted_move_sequence, True) # TODO: Investigate. This completely takes out the shooting that we planned before!
            best_action_sim.set_fire_next_timestep_flag(best_predicted_sim_fire_next_timestep_flag)
            best_action_fitness = best_action_sim.get_fitness()
            best_action_fitness_breakdown = best_action_sim.get_fitness_breakdown()
            # debug_print(f"\nActual best action first state:", best_action_sim.get_state_sequence()[0])
            # debug_print(f"\nUpdated simmed state. Old predicted fitness: {best_action_fitness_predicted}, new predicted fitness: {best_action_fitness}")
            if best_action_fitness < best_action_fitness_predicted - 0.05:
                # debug_print(f"\n\n\n\nDANGERRRRR!!!!! Updated simmed state. Old predicted fitness: {best_action_fitness_predicted}, new predicted fitness IS MUCH WORSE!!!!!!!: {best_action_fitness}")
                if self.second_best_fitness_this_planning_period_index is not None:
                    # The best action sim's reality is worse than expected. Try our second best as a backup and hopefully this will be better, and go according to plan!
                    if self.sims_this_planning_period[self.second_best_fitness_this_planning_period_index]['state_type'] == 'predicted':
                        # The second best sim also uses a predicted state
                        second_best_action_sim_predicted: Matrix = self.sims_this_planning_period[self.second_best_fitness_this_planning_period_index]['sim']
                        second_best_action_fitness_predicted = self.sims_this_planning_period[self.second_best_fitness_this_planning_period_index]['fitness']
                        second_best_predicted_sim_fire_next_timestep_flag = second_best_action_sim_predicted.get_fire_next_timestep_flag()
                        #print(f"Doing second best action sim. Ship respawn timer: {self.game_state_to_base_planning['ship_respawn_timer']}, asts pending death: {self.game_state_to_base_planning['asteroids_pending_death']}, forecasted splits: {self.game_state_to_base_planning['forecasted_asteroid_splits']}, is respawning: {self.game_state_to_base_planning['respawning']}, fire next ts flag: {self.game_state_to_base_planning['fire_next_timestep_flag']}")
                        second_best_action_sim = Matrix(game_state=game_state,
                                                        ship_state=ship_state,
                                                        initial_timestep=self.current_timestep,
                                                        respawn_timer=self.game_state_to_base_planning['ship_respawn_timer'],
                                                        asteroids_pending_death=self.game_state_to_base_planning['asteroids_pending_death'],
                                                        forecasted_asteroid_splits=self.game_state_to_base_planning['forecasted_asteroid_splits'],
                                                        last_timestep_fired=self.game_state_to_base_planning['last_timestep_fired'],
                                                        last_timestep_mined=self.game_state_to_base_planning['last_timestep_mined'],
                                                        mine_positions_placed=self.game_state_to_base_planning['mine_positions_placed'],
                                                        halt_shooting=self.game_state_to_base_planning['respawning'],
                                                        fire_first_timestep=self.game_state_to_base_planning['fire_next_timestep_flag'],
                                                        verify_first_shot=True,
                                                        verify_maneuver_shots=True,
                                                        last_timestep_colliding=second_best_action_sim_predicted.get_last_timestep_colliding() if self.game_state_to_base_planning['respawning'] else -1,
                                                        game_state_plotter=self.game_state_plotter)
                        second_best_action_sim_predicted_move_sequence = second_best_action_sim_predicted.get_intended_move_sequence()
                        # print(f"Applying move sequence for maneuver #2, seq is {second_best_action_sim_predicted_move_sequence}")
                        if self.game_state_to_base_planning['respawning']:
                            second_best_action_sim.apply_move_sequence(second_best_action_sim_predicted_move_sequence, True)
                        else:
                            # This can only be maneuvering, and this gives the freedom to aim and shoot during the maneuver
                            print('calling sim maneuver in decide second best')
                            second_best_action_sim.simulate_maneuver(second_best_action_sim_predicted_move_sequence, True)
                        second_best_action_sim.set_fire_next_timestep_flag(second_best_predicted_sim_fire_next_timestep_flag)
                        second_best_action_fitness = second_best_action_sim.get_fitness()
                        second_best_action_fitness_breakdown = second_best_action_sim.get_fitness_breakdown()
                    else:
                        # The second best sim uses an exact state
                        second_best_action_sim = self.sims_this_planning_period[self.second_best_fitness_this_planning_period_index]['sim']
                        second_best_action_fitness = self.sims_this_planning_period[self.second_best_fitness_this_planning_period_index]['fitness']
                        second_best_action_fitness_breakdown = self.sims_this_planning_period[self.second_best_fitness_this_planning_period_index]['fitness_breakdown']
                    second_best_action_maneuver_tuple = self.sims_this_planning_period[self.second_best_fitness_this_planning_period_index]['maneuver_tuple']
                    if second_best_action_fitness > best_action_fitness:
                        # debug_print(f"HOORAY, the second best action's real fitness of {second_best_action_fitness} and predicted fitness of {second_best_action_fitness_predicted} is better than the best!")
                        best_action_fitness = second_best_action_fitness
                        best_action_sim = second_best_action_sim
                        best_action_fitness_breakdown = second_best_action_fitness_breakdown
                        best_action_maneuver_tuple = second_best_action_maneuver_tuple
                    #else:
                        # debug_print(f"CRAP, even the second best action's real fitness of {second_best_action_fitness} and predicted fitness of {second_best_action_fitness_predicted} isn't better than the first, so we'll just have to go with what we have and maybe get screwed.")

            if self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['action_type'] == 'targetting':
                # The targetting sim was done with the true state, so this should be the exact same and redundant
                raise Exception("WHY THE HECK ARE WE IN HERE?!!")
        else:
            # The state we based planning off of is exact
            if self.game_state_to_base_planning['respawning']:
                # If we did a respawn maneuver, we still have to run a second pass of it so we can get more shots in at the end, and hopefully eek out a bit more fitness score
                best_action_sim_respawn_first_pass: Matrix = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['sim']
                best_action_sim_respawn_first_pass_fitness: float = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness']
                best_respawn_first_pass_sim_fire_next_timestep_flag = best_action_sim_respawn_first_pass.get_fire_next_timestep_flag()
                #print(f"RUNNING SECOND PASS OF RESPAWN MANEUVER. {best_action_sim_respawn_first_pass.get_last_timestep_colliding_with_asteroid()=}")
                best_action_sim = Matrix(game_state=game_state,
                                         ship_state=ship_state,
                                         initial_timestep=self.current_timestep,
                                         respawn_timer=self.game_state_to_base_planning['ship_respawn_timer'],
                                         asteroids_pending_death=self.game_state_to_base_planning['asteroids_pending_death'],
                                         forecasted_asteroid_splits=self.game_state_to_base_planning['forecasted_asteroid_splits'],
                                         last_timestep_fired=self.game_state_to_base_planning['last_timestep_fired'],
                                         last_timestep_mined=self.game_state_to_base_planning['last_timestep_mined'],
                                         mine_positions_placed=self.game_state_to_base_planning['mine_positions_placed'],
                                         halt_shooting=self.game_state_to_base_planning['respawning'],
                                         fire_first_timestep=self.game_state_to_base_planning['fire_next_timestep_flag'],
                                         verify_first_shot=True,
                                         verify_maneuver_shots=True,
                                         last_timestep_colliding=best_action_sim_respawn_first_pass.get_last_timestep_colliding(), # This is the secret sauce!
                                         game_state_plotter=self.game_state_plotter)
                best_action_sim_respawn_first_pass_move_sequence = best_action_sim_respawn_first_pass.get_intended_move_sequence()
                best_action_sim.apply_move_sequence(best_action_sim_respawn_first_pass_move_sequence, True)
                best_action_sim.set_fire_next_timestep_flag(best_respawn_first_pass_sim_fire_next_timestep_flag)
                best_action_fitness = best_action_sim.get_fitness()
                best_action_fitness_breakdown = best_action_sim.get_fitness_breakdown()
                best_action_maneuver_tuple = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['maneuver_tuple']
                print(f"First pass fitness: {best_action_sim_respawn_first_pass_fitness}, second pass fitness: {best_action_fitness}, first pass breakdown: {self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness_breakdown']}, second pass breakdown: {best_action_fitness_breakdown}")
                if best_action_sim_respawn_first_pass_fitness > best_action_fitness + 0.015:
                    print("REVERTING TO FIRST PASS. SECOND PASS DIDN'T HELP!")
                    # The additional shots didn't actually help our fitness. Reverting to just using the first pass sim which is totally valid still
                    best_action_sim = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['sim']
                    best_action_fitness = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness']
                    best_action_fitness_breakdown = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness_breakdown']
                    best_action_maneuver_tuple = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['maneuver_tuple']
            else:
                # Exact planning, and this isn't a respawn maneuver so we just do the one-pass simulation method and call it a day
                best_action_sim = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['sim']
                best_action_fitness = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness']
                best_action_fitness_breakdown = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness_breakdown']
                best_action_maneuver_tuple = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['maneuver_tuple']
        if best_action_maneuver_tuple is not None and not self.game_state_to_base_planning['respawning'] and best_action_fitness_breakdown[5] != 0.0:
            # This is either a heuristic or random maneuver
            #global total_abs_cruise_speed, total_cruise_timesteps, total_maneuvers_to_learn_from
            #total_abs_cruise_speed += abs(best_action_maneuver_tuple[1])
            #total_cruise_timesteps += best_action_maneuver_tuple[3]
            #total_maneuvers_to_learn_from += 1
            global abs_cruise_speeds, cruise_timesteps
            abs_cruise_speeds.append(abs(best_action_maneuver_tuple[1]))
            cruise_timesteps.append(best_action_maneuver_tuple[3])
            if len(abs_cruise_speeds) > MANEUVER_TUPLE_LEARNING_ROLLING_AVERAGE_PERIOD:
                abs_cruise_speeds = abs_cruise_speeds[-MANEUVER_TUPLE_LEARNING_ROLLING_AVERAGE_PERIOD:]
            if len(cruise_timesteps) > MANEUVER_TUPLE_LEARNING_ROLLING_AVERAGE_PERIOD:
                cruise_timesteps = cruise_timesteps[-MANEUVER_TUPLE_LEARNING_ROLLING_AVERAGE_PERIOD:]
            #print(f"{best_action_maneuver_tuple=}, and the avg best cruise speed is now {weighted_average(abs_cruise_speeds)} and avg cruise timesteps is {weighted_average(cruise_timesteps)}")
        # Maintain a rolling average of the overall fitnesses, so we know how well we're doing
        global overall_fitness_record
        overall_fitness_record.append(best_action_fitness)
        if len(overall_fitness_record) > OVERALL_FITNESS_ROLLING_AVERAGE_PERIOD:
            overall_fitness_record = overall_fitness_record[-OVERALL_FITNESS_ROLLING_AVERAGE_PERIOD:]
        
        # Print out the explanation messages that were stored within the sim
        if self.stationary_targetting_sim_index is not None:
            stationary_safety_messages: list[str] = self.sims_this_planning_period[self.stationary_targetting_sim_index]['sim'].get_safety_messages()
            for message in stationary_safety_messages:
                print_explanation(message, self.current_timestep)

        # if best_action_fitness <= 0.1:
        if best_action_fitness_breakdown[5] == 0.0:
            # We're gonna die. Force select the one where I stay put and accept my fate, and don't even begin a maneuver.
            print_explanation("RIP, I'm gonna die", self.current_timestep)
            #print('IT LOOKS LIKE THIS NEW ACTION WE ARE DOING ENDS IN DEATH!!!')
            # if self.stationary_targetting_sim_index:
            #    self.best_fitness_this_planning_period_index = self.stationary_targetting_sim_index
            #    best_action_sim: Simulation = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['sim']
            #    best_action_fitness = self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness']
        if self.game_state_to_base_planning['respawning']:
            print_explanation("Doing a respawn maneuver to get to a safe spot using my respawn invincibility", self.current_timestep)
        if self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['action_type'] in ['random_maneuver', 'heuristic_maneuver']:
            # [asteroid_safe_time_fitness, mine_safe_time_fitness, asteroids_fitness, sequence_length_fitness, other_ship_proximity_fitness, crash_fitness, asteroid_aiming_cone_fitness]
            #if self.stationary_targetting_sim_index is None:
                #print(f"WARNING: There are no stationary targetting sims!")
            if self.stationary_targetting_sim_index is not None:
                stationary_fitness_breakdown = self.sims_this_planning_period[self.stationary_targetting_sim_index]['fitness_breakdown']
                # debug_print('stationary fitneses', stationary_fitness_breakdown)
                #print(f"Stationary breakdown: {stationary_fitness_breakdown}, best sim breakdown: {best_action_fitness_breakdown}")
                if best_action_fitness_breakdown[1] == 1.0 and stationary_fitness_breakdown[1] == 1.0:
                    # No mines are threatening us whether we stay put or move
                    if best_action_fitness_breakdown[0] > stationary_fitness_breakdown[0]:
                        print_explanation("Doing a maneuver to dodge asteroids!", self.current_timestep)
                elif best_action_fitness_breakdown[1] > stationary_fitness_breakdown[1]:
                    print_explanation("Doing a maneuver to dodge a mine!", self.current_timestep)
                if best_action_fitness_breakdown[4] > stationary_fitness_breakdown[4] + 0.05:
                    print_explanation("Doing a maneuver to get away from the other ship!", self.current_timestep)
        best_move_sequence = best_action_sim.get_move_sequence()
        #print(f"Best sim ID: {best_action_sim.get_sim_id()}, with index {self.best_fitness_this_planning_period_index} and fitness {best_action_fitness} breakdown: {best_action_fitness_breakdown} and length {len(best_move_sequence)}")#, move seq: {best_move_sequence}")
        #print(f"Current average overall fitness is {weighted_average(overall_fitness_record)}")
        # debug_print(f"Respawn maneuver status is: {self.game_state_to_base_planning['respawning']}, Move type: {self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['action_type']}, state type: {self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['state_type']}, Best move seq with fitness {best_action_fitness}: {best_move_sequence}")
        best_action_sim_state_sequence = best_action_sim.get_state_sequence()
        # debug_print(f"The action we're taking is from timestep {best_action_sim_state_sequence[0]['timestep']} to {best_action_sim_state_sequence[-1]['timestep']}")
        explanation_messages = best_action_sim.get_explanations()
        for explanation in explanation_messages:
            print_explanation(explanation, self.current_timestep)
        # end_state = sim_ship.get_state_sequence()[-1]
        # debug_print(f"Maneuver fitness: {best_action_fitness}, stationary fitness: {self.sims_this_planning_period[self.stationary_targetting_sim_index]['fitness']}")
        # print('state seq:', best_action_sim_state_sequence)
        # debug_print('Best move seq:', best_move_sequence)
        # debug_print(f"Best sim index: {self.best_fitness_this_planning_period_index}")
        # debug_print(f"Choosing action: {self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['action_type']} with fitness {best_action_fitness} {best_action_fitness_breakdown}")
        # print('all sims this planning period:')
        # print(self.sims_this_planning_period)
        if not best_action_sim_state_sequence:
            raise Exception("Why in the world is this state sequence empty?")
        best_action_sim_last_state = best_action_sim_state_sequence[-1]
        # Prune out the list of asteroids we shot at if the timestep (key) is in the past
        asteroids_pending_death = best_action_sim.get_asteroids_pending_death()
        # debug_print(f"Timesteps in asteroids pending death: {[timestep for timestep in asteroids_pending_death.keys()]}")
        #print(f"Size of asts pending death: {sys.getsizeof()}")
        #global asteroids_pending_death_total_cull_time
        #start_time = time.perf_counter()
        #asteroids_pending_death = {timestep: asteroids for timestep, asteroids in asteroids_pending_death.items() if timestep >= best_action_sim_last_state.timestep}
        for timestep in range(self.current_timestep, best_action_sim_last_state.timestep):
            if timestep in asteroids_pending_death:
                del asteroids_pending_death[timestep]
        #asteroids_pending_death_total_cull_time += time.perf_counter() - start_time
        forecasted_asteroid_splits = best_action_sim.get_forecasted_asteroid_splits()
        next_base_game_state = best_action_sim.get_game_state()
        #print(f"Ast pending death keys: {asteroids_pending_death.keys()}")
        # Made this change, because if we're waiting out mines, that'll mess up the game state. But the state sequence still has the last actual game state, so we'll use that!
        # next_base_game_state = best_action_sim_last_state['game_state']
        # print(f'\nNext base game state for timestep {best_action_sim_last_state["timestep"]}:')
        self.set_of_base_gamestate_timesteps.add(best_action_sim_last_state.timestep)
        new_ship_state = best_action_sim.get_ship_state()
        new_fire_next_timestep_flag = best_action_sim.get_fire_next_timestep_flag()
        # print(f"Firing next ts status is: {new_fire_next_timestep_flag}")
        if new_ship_state.is_respawning and new_fire_next_timestep_flag and new_ship_state.lives_remaining not in self.lives_remaining_that_we_did_respawn_maneuver_for:
            # print(f"Forcing off the fire next timestep, because we just took damage")
            new_fire_next_timestep_flag = False
        # print(f"{new_ship_state.lives_remaining=}, {str(self.lives_remaining_that_we_did_respawn_maneuver_for)=}, {new_ship_state.is_respawning=}")
        # debug_print(f"Deciding next action on ts {self.current_timestep}! The new planning ship speed is {new_ship_state.speed} and the move sequence we're executing is REDACTED best_move_sequence, type is {self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['action_type']}")
        self.game_state_to_base_planning = {
            'timestep': best_action_sim_last_state.timestep,
            'respawning': new_ship_state.lives_remaining not in self.lives_remaining_that_we_did_respawn_maneuver_for and new_ship_state.is_respawning,
            'ship_state': new_ship_state,
            'game_state': next_base_game_state,
            'ship_respawn_timer': best_action_sim.get_respawn_timer(),
            'asteroids_pending_death': asteroids_pending_death,
            'forecasted_asteroid_splits': forecasted_asteroid_splits,
            'last_timestep_fired': best_action_sim.get_last_timestep_fired(),
            'last_timestep_mined': best_action_sim.get_last_timestep_mined(),
            'mine_positions_placed': best_action_sim.get_mine_positions_placed(),
            'fire_next_timestep_flag': new_fire_next_timestep_flag,
        }
        if self.game_state_to_base_planning['respawning']:
            # print(f"Adding to lives remaining that we did respawn for, in decide next action: {new_ship_state.lives_remaining}")
            self.lives_remaining_that_we_did_respawn_maneuver_for.add(new_ship_state.lives_remaining)
        # debug_print(f"The next base state's respawning state is {self.game_state_to_base_planning['respawning']}")
        # debug_print("The new ship state is", new_ship_state)
        # debug_print(f"The fire next timestep flag is: {new_fire_next_timestep_flag}")
        # debug_print(f"\nNext base state ts: {self.game_state_to_base_planning['timestep']}, respawn maneuver: {self.game_state_to_base_planning['respawning']}, respawn timer: {self.game_state_to_base_planning['ship_respawn_timer']}, ship state: {new_ship_state}")
        self.base_gamestates[best_action_sim_last_state.timestep] = self.game_state_to_base_planning
        state_dump_dict = {
            'timestep': self.game_state_to_base_planning['timestep'],
            'ship_state': self.game_state_to_base_planning['ship_state'],
            'asteroids': self.game_state_to_base_planning['game_state'].asteroids,
            'bullets': self.game_state_to_base_planning['game_state'].bullets,
        }

        # print(f"Best move sequence:", best_move_sequence)
        for move in best_move_sequence:
            self.enqueue_action(move.timestep, move.thrust, move.turn_rate, move.fire, move.drop_mine)
        self.sims_this_planning_period.clear()
        self.best_fitness_this_planning_period = -inf
        self.best_fitness_this_planning_period_index = None  # TODO: Change this to -1 so mypyc is faster!
        self.second_best_fitness_this_planning_period = -inf
        self.second_best_fitness_this_planning_period_index = None
        self.stationary_targetting_sim_index = None
        self.base_gamestate_analysis = None
        #if len(best_move_sequence) > 5:
        #if random.random() > 0.5:
        global unwrap_cache
        unwrap_cache.clear()

    def plan_action(self, other_ships_exist: bool, base_state_is_exact: bool, iterations_boost: bool = False, plan_stationary: bool = False) -> None:
        #print("Calling plan action")
        # gc.disable()
        # Simulate and look for a good move
        # We have two options. Stay put and focus on targetting asteroids, or we can come up with an avoidance maneuver and target asteroids along the way if convenient
        # We simulate both options, and take the one with the higher fitness score
        # If we stay still, we can potentially keep shooting asteroids that are on collision course with us without having to move
        # But if we're overwhelmed, it may be a lot better to move to a safer spot
        # The third scenario is that even if we're safe where we are, we may be able to be on the offensive and seek out asteroids to lay mines, so that can also increase the fitness function of moving, making it better than staying still
        # Our number one priority is to stay alive. Second priority is to shoot as much as possible. And if we can, lay mines without putting ourselves in danger.
        assert self.game_state_to_base_planning is not None
        state_type = 'exact' if base_state_is_exact else 'predicted'
        #index_according_to_lives_remaining = min(3, self.game_state_to_base_planning['ship_state'].lives_remaining)
        if self.game_state_to_base_planning['respawning']:
            #print("Planning respawn maneuver")
            # Simulate and look for a good move
            # print(f"Checking for imminent danger. We're currently at position {ship_state.position[0]} {ship_state.position[1]}")
            # print(f"Current ship location: {ship_state.position[0]}, {ship_state.position[1]}, ship heading: {ship_state.heading}")

            # Check for danger
            MAX_CRUISE_SECONDS = 1.0 + 26.0*DELTA_TIME
            # ship_random_range, ship_random_max_maneuver_length = get_simulated_ship_max_range(max_cruise_seconds)
            # print(f"Respawn maneuver max length: {ship_random_max_maneuver_length}s")

            #print("Looking for a respawn maneuver")
            # Run a simulation and find a course of action to put me to safety
            search_iterations_count = 0

            # while search_iterations_count < min_search_iterations or (not safe_maneuver_found and search_iterations_count < max_search_iterations):
            # for _ in range(search_iterations):
            while (search_iterations_count < get_min_respawn_per_timestep_search_iterations(self.game_state_to_base_planning['ship_state'].lives_remaining, weighted_average(overall_fitness_record)) or self.performance_controller_check_whether_i_can_do_another_iteration()) and not search_iterations_count >= MAX_RESPAWN_PER_TIMESTEP_SEARCH_ITERATIONS:
                self.performance_controller_start_iteration()
                search_iterations_count += 1
                if search_iterations_count % 1 == 0:
                    # print(f"Respawn search iteration {search_iterations_count}")
                    pass
                num_sims_this_planning_period = len(self.sims_this_planning_period)
                if num_sims_this_planning_period == 0:
                    # On the first iteration, try the null action. For ring scenarios, it may be best to stay at the center of the ring.
                    # TODO: RESTORE NULL ACTION
                    random_ship_heading_angle = 0.0
                    ship_accel_turn_rate = 0.0
                    ship_cruise_speed = 0.0
                    ship_cruise_turn_rate = 0.0
                    ship_cruise_timesteps = 0
                elif num_sims_this_planning_period == 1:
                    # TODO: Use this opportunity to aim at an asteroid! But we need to do something to get it to shoot, so maybe this won't work with our current framework rip.
                    # On the second iteration, try staying still for 1 second (and just turn a little bit so we can use the same framework to do this null movement with a wait)
                    random_ship_heading_angle = 180.0
                    ship_accel_turn_rate = 180.0
                    ship_cruise_speed = 0.0
                    ship_cruise_turn_rate = 0.0
                    ship_cruise_timesteps = 0
                elif num_sims_this_planning_period == 2:
                    # TODO: Use this opportunity to aim at an asteroid!
                    # On the third iteration, try staying still for 2 seconds (and just turn a little bit so we can use the same framework to do this null movement with a wait)
                    random_ship_heading_angle = 180.0
                    ship_accel_turn_rate = 90.0
                    ship_cruise_speed = 0.0
                    ship_cruise_turn_rate = 0.0
                    ship_cruise_timesteps = 0
                else:
                    random_ship_heading_angle = random.uniform(-20.0, 20.0)
                    ship_accel_turn_rate = random.uniform(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)
                    ship_cruise_speed = SHIP_MAX_SPEED*random.choice([-1, 1])
                    ship_cruise_turn_rate = 0.0
                    ship_cruise_timesteps = random.randint(0, round(MAX_CRUISE_SECONDS*FPS))
                # TODO: There's a hardcoded false in the arguments to the following sim. Investigate!!!
                #print(f"Doing respawn maneuver with {random_ship_heading_angle=} {ship_accel_turn_rate=} {ship_cruise_speed=} {ship_cruise_turn_rate=} {ship_cruise_timesteps=}")
                maneuver_sim = Matrix(game_state=self.game_state_to_base_planning['game_state'],
                                      ship_state=self.game_state_to_base_planning['ship_state'],
                                      initial_timestep=self.game_state_to_base_planning['timestep'],
                                      respawn_timer=self.game_state_to_base_planning['ship_respawn_timer'],
                                      asteroids_pending_death=self.game_state_to_base_planning['asteroids_pending_death'],
                                      forecasted_asteroid_splits=self.game_state_to_base_planning['forecasted_asteroid_splits'],
                                      last_timestep_fired=self.game_state_to_base_planning['last_timestep_fired'],
                                      last_timestep_mined=self.game_state_to_base_planning['last_timestep_mined'],
                                      mine_positions_placed=self.game_state_to_base_planning['mine_positions_placed'],
                                      halt_shooting=True,
                                      fire_first_timestep=False and self.game_state_to_base_planning['fire_next_timestep_flag'],
                                      verify_first_shot=False,
                                      verify_maneuver_shots=False,
                                      game_state_plotter=self.game_state_plotter)
                # This statement's a doozy. We evaluate left to right, and once it returns false, we stop going.
                (maneuver_sim.rotate_heading(random_ship_heading_angle) and maneuver_sim.accelerate(ship_cruise_speed, ship_accel_turn_rate) and maneuver_sim.cruise(ship_cruise_timesteps, ship_cruise_turn_rate) and maneuver_sim.accelerate(0))
                    # The ship went through all the steps without colliding
                    #debug_print("The ship went through all the steps without colliding")
                    # maneuver_complete_without_crash = True
                #else:
                    # The ship crashed somewhere before reaching the final resting spot
                    #debug_print("The ship crashed somewhere before reaching the final resting spot")
                #print(f"Move seq: {maneuver_sim.get_move_sequence()}")
                maneuver_fitness = maneuver_sim.get_fitness()
                #print(f"Respawn maneuver fitness: {maneuver_fitness} {maneuver_sim.get_fitness_breakdown()}, move seq length was {len(maneuver_sim.get_move_sequence())}")

                self.sims_this_planning_period.append({
                    'sim': maneuver_sim,
                    'fitness': maneuver_fitness,
                    'fitness_breakdown': maneuver_sim.get_fitness_breakdown(),
                    'action_type': 'respawn',
                    'state_type': state_type,
                    'maneuver_tuple': (random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate, ship_cruise_timesteps, ship_cruise_turn_rate)
                })
                if maneuver_fitness > self.best_fitness_this_planning_period:
                    self.second_best_fitness_this_planning_period = self.best_fitness_this_planning_period
                    self.second_best_fitness_this_planning_period_index = self.best_fitness_this_planning_period_index

                    self.best_fitness_this_planning_period = maneuver_fitness
                    self.best_fitness_this_planning_period_index = len(self.sims_this_planning_period) - 1
        else:
            # Non-respawn move

            # Stationary targetting simulation
            #assert self.game_state_to_base_planning is not None
            if self.base_gamestate_analysis is None:
                self.base_gamestate_analysis = analyze_gamestate_for_heuristic_maneuver(self.game_state_to_base_planning['game_state'], self.game_state_to_base_planning['ship_state'])
            ship_is_stationary = True
            if plan_stationary and self.game_state_to_base_planning['ship_state'].bullets_remaining != 0 and (ship_is_stationary := is_close_to_zero(self.game_state_to_base_planning['ship_state'].speed)):
                # No need to check whether this is allowed, because we need to do this iteration at minimum
                self.performance_controller_start_iteration()
                # The first list element is the stationary targetting
                # print('game state to base planning:')
                # print(self.game_state_to_base_planning)
                # debug_print('Stationary sim ast pending death:')
                # debug_print(self.game_state_to_base_planning['asteroids_pending_death'])
                stationary_targetting_sim = Matrix(game_state=self.game_state_to_base_planning['game_state'],
                                                   ship_state=self.game_state_to_base_planning['ship_state'],
                                                   initial_timestep=self.game_state_to_base_planning['timestep'],
                                                   respawn_timer=self.game_state_to_base_planning['ship_respawn_timer'],
                                                   asteroids_pending_death=self.game_state_to_base_planning['asteroids_pending_death'],
                                                   forecasted_asteroid_splits=self.game_state_to_base_planning['forecasted_asteroid_splits'],
                                                   last_timestep_fired=self.game_state_to_base_planning['last_timestep_fired'],
                                                   last_timestep_mined=self.game_state_to_base_planning['last_timestep_mined'],
                                                   mine_positions_placed=self.game_state_to_base_planning['mine_positions_placed'],
                                                   halt_shooting=False,
                                                   fire_first_timestep=self.game_state_to_base_planning['fire_next_timestep_flag'],
                                                   verify_first_shot=True if len(self.sims_this_planning_period) == 0 and other_ships_exist else False,
                                                   verify_maneuver_shots=False,
                                                   game_state_plotter=self.game_state_plotter)
                stationary_targetting_sim.target_selection()
                #print('\nstationary targetting sim move seq')
                #print(stationary_targetting_sim.get_move_sequence())

                best_stationary_targetting_fitness = stationary_targetting_sim.get_fitness()
                if len(self.sims_this_planning_period) == 0:
                    if stationary_targetting_sim.get_cancel_firing_first_timestep():
                        # The plan was to first at the first timestep this planning period. However, due to non-determinism caused by the existence of another ship, this shot would actually miss. We checked and caught this, so we're going to just nix the idea of shooting on the first timestep.
                        self.game_state_to_base_planning['fire_next_timestep_flag'] = False

                self.sims_this_planning_period.append({
                    'sim': stationary_targetting_sim,
                    'fitness': best_stationary_targetting_fitness,
                    'fitness_breakdown': stationary_targetting_sim.get_fitness_breakdown(),
                    'action_type': 'targetting',
                    'state_type': state_type,
                    'maneuver_tuple': None,
                })
                self.stationary_targetting_sim_index = len(self.sims_this_planning_period) - 1
                if best_stationary_targetting_fitness > self.best_fitness_this_planning_period:
                    self.second_best_fitness_this_planning_period = self.best_fitness_this_planning_period
                    self.second_best_fitness_this_planning_period_index = self.best_fitness_this_planning_period_index

                    self.best_fitness_this_planning_period = best_stationary_targetting_fitness
                    self.best_fitness_this_planning_period_index = self.stationary_targetting_sim_index

                # debug_print(f"Planning targetting, and got fitness {best_stationary_targetting_fitness}")
            if plan_stationary and not ship_is_stationary:
                print(f"\nWARNING: The ship wasn't stationary after the last maneuver, so we're skipping stationary targeting! Our planning period starts on ts {self.game_state_to_base_planning['timestep']}")
            # Try moving! Run a simulation and find a course of action to put me to safety
            '''
            if other_ships_exist:
                if isinf(self.best_fitness_this_planning_period):
                    # This is the first timestep we're planning for this period, so we don't really know how many iterations to use. Don't go all out on this first one in case it's an easy one.
                    search_iterations = 2
                elif self.best_fitness_this_planning_period > 0.9:
                    search_iterations = 1
                elif self.best_fitness_this_planning_period > 0.8:
                    search_iterations = 1
                elif self.best_fitness_this_planning_period > 0.7:
                    search_iterations = 2
                elif self.best_fitness_this_planning_period > 0.6:
                    search_iterations = 3
                elif self.best_fitness_this_planning_period > 0.5:
                    search_iterations = 4
                elif self.best_fitness_this_planning_period > 0.4:
                    search_iterations = 5
                else:
                    search_iterations = 6
            else:
                if isinf(self.best_fitness_this_planning_period) and self.game_state_to_base_planning['ship_state'].bullets_remaining != 0:
                    raise Exception("If there's no ships, why don't we have any sims this planning period yet? We should have done stationary first.")
                    # This is the first timestep we're planning for this period, so we don't really know how many iterations to use. Don't go all out on this first one in case it's an easy one.
                    search_iterations = 2
                elif self.best_fitness_this_planning_period > 0.9:
                    search_iterations = 1
                elif self.best_fitness_this_planning_period > 0.8:
                    search_iterations = 1
                elif self.best_fitness_this_planning_period > 0.7:
                    search_iterations = 2
                elif self.best_fitness_this_planning_period > 0.6:
                    search_iterations = 3
                elif self.best_fitness_this_planning_period > 0.5:
                    search_iterations = 4
                elif self.best_fitness_this_planning_period > 0.4:
                    search_iterations = 5
                else:
                    search_iterations = 6
            '''

            '''
            if iterations_boost:
                search_iterations = min(80, (search_iterations + 1)*10)

            if self.game_state_to_base_planning['ship_state'].lives_remaining == 1:
                # When down to our last life, try twice as hard to survive
                search_iterations *= 2
            elif self.game_state_to_base_planning['ship_state'].lives_remaining == 2:
                search_iterations = floor(search_iterations*1.5)
            '''
            if len(self.sims_this_planning_period) == 0 or (len(self.sims_this_planning_period) == 1 and self.sims_this_planning_period[0]['action_type'] != 'heuristic_maneuver'):
                heuristic_maneuver = True
            else:
                heuristic_maneuver = False

            imminent_asteroid_speed, imminent_asteroid_relative_heading, largest_gap_relative_heading, nearby_asteroid_average_speed, nearby_asteroid_count, average_directional_speed, total_asteroids_count, current_asteroids_count = self.base_gamestate_analysis

            # Let's just pretend the following is a fuzzy system lol
            # For performance and simplicity, I'll just use a bunch of if statements
            if average_directional_speed > 80 and current_asteroids_count > 5 and total_asteroids_count >= 100:
                # This is probably a wall scenario! We have many asteroids all travelling in basically the same direction
                print_explanation(f"Wall scenario detected! Preferring trying longer cruise lengths", self.current_timestep)
                ship_cruise_speed_mode = SHIP_MAX_SPEED
                ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS
                max_pre_maneuver_turn_timesteps = 6.0
            elif any(m.position in self.game_state_to_base_planning['mine_positions_placed'] for m in self.game_state_to_base_planning['game_state'].mines):
                # We're probably within the radius of a mine we placed
                print_explanation("We're probably within the radius of a mine we placed! Biasing faster/longer moves to be more likely to escape the mine.", self.current_timestep)
                ship_cruise_speed_mode = SHIP_MAX_SPEED
                ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.75
                max_pre_maneuver_turn_timesteps = 10.0
            else:
                max_pre_maneuver_turn_timesteps = 15.0
                #ship_cruise_speed_mode = nan
                #ship_cruise_timesteps_mode = nan
                #global total_abs_cruise_speed, total_cruise_timesteps, total_maneuvers_to_learn_from
                #ship_cruise_speed_mode = total_abs_cruise_speed/total_maneuvers_to_learn_from
                #ship_cruise_timesteps_mode = total_cruise_timesteps/total_maneuvers_to_learn_from
                global abs_cruise_speeds, cruise_timesteps
                ship_cruise_speed_mode = weighted_average(abs_cruise_speeds)
                ship_cruise_timesteps_mode = weighted_average(cruise_timesteps)
            '''
            if nearby_asteroid_count > 15:
                # Many nearby asteroids
                if nearby_asteroid_average_speed > 100:
                    # Fast asteroids
                    ship_cruise_speed_mode = SHIP_MAX_SPEED*0.25
                    ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.15
                elif nearby_asteroid_average_speed > 50:
                    # Medium asteroids
                    ship_cruise_speed_mode = SHIP_MAX_SPEED*0.18
                    ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.1
                else:
                    # Slow asteroids
                    ship_cruise_speed_mode = SHIP_MAX_SPEED*0.0
                    ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.0
            elif nearby_asteroid_count > 5:
                # Some nearby asteroids
                if nearby_asteroid_average_speed > 100:
                    # Fast asteroids
                    ship_cruise_speed_mode = SHIP_MAX_SPEED*0.5
                    ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.33
                elif nearby_asteroid_average_speed > 50:
                    # Medium asteroids
                    ship_cruise_speed_mode = SHIP_MAX_SPEED*0.5
                    ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.28
                else:
                    # Slow asteroids
                    ship_cruise_speed_mode = SHIP_MAX_SPEED*0.5
                    ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.25
            else:
                # Few nearby asteroids
                if nearby_asteroid_average_speed > 100:
                    # Fast asteroids
                    ship_cruise_speed_mode = SHIP_MAX_SPEED*0.5
                    ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.33
                elif nearby_asteroid_average_speed > 50:
                    # Medium asteroids
                    ship_cruise_speed_mode = SHIP_MAX_SPEED*0.37
                    ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.33
                else:
                    # Slow asteroids
                    ship_cruise_speed_mode = SHIP_MAX_SPEED*0.25
                    ship_cruise_timesteps_mode = MAX_CRUISE_TIMESTEPS*0.33
            '''
            # print(f"Nearby asteroids count is {nearby_asteroid_count}, average speed of asts is {nearby_asteroid_average_speed}, avg directional speed is {average_directional_speed}, so therefore I'm picking ship cruise timesteps mode to be {ship_cruise_timesteps_mode} and ship speed mode of {ship_cruise_speed_mode}")
            search_iterations_count = 0
            while (search_iterations_count < get_min_maneuver_per_timestep_search_iterations(self.game_state_to_base_planning['ship_state'].lives_remaining, weighted_average(overall_fitness_record)) or self.performance_controller_check_whether_i_can_do_another_iteration()) and not search_iterations_count >= MAX_MANEUVER_PER_TIMESTEP_SEARCH_ITERATIONS:
                self.performance_controller_start_iteration()
                search_iterations_count += 1
                if heuristic_maneuver:
                    random_ship_heading_angle = 0.0
                    ship_accel_turn_rate, ship_cruise_speed, ship_cruise_turn_rate, ship_cruise_timesteps_float, thrust_direction = maneuver_heuristic_fis(imminent_asteroid_speed, imminent_asteroid_relative_heading, largest_gap_relative_heading, nearby_asteroid_average_speed, nearby_asteroid_count)
                    # print(ship_accel_turn_rate, ship_cruise_speed, ship_cruise_turn_rate, ship_cruise_timesteps, thrust_direction)
                    ship_cruise_timesteps = round(ship_cruise_timesteps_float)
                    if thrust_direction < -GRAIN:
                        ship_cruise_speed = -ship_cruise_speed
                    elif thrust_direction < GRAIN:
                        # The FIS couldn't decide which way to thrust, so we'll just skip the heuristic maneuver altogether
                        heuristic_maneuver = False
                if not heuristic_maneuver:
                    random_ship_heading_angle = random.triangular(-6.0*max_pre_maneuver_turn_timesteps, 6.0*max_pre_maneuver_turn_timesteps, 0)
                    ship_accel_turn_rate = random.triangular(0, SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)*(2.0*float(random.getrandbits(1)) - 1.0)
                    # random_ship_cruise_speed = random.uniform(-ship_max_speed, ship_max_speed)
                    if isnan(ship_cruise_speed_mode):
                        ship_cruise_speed = random.uniform(-SHIP_MAX_SPEED, SHIP_MAX_SPEED)
                    else:
                        ship_cruise_speed = random.triangular(0, SHIP_MAX_SPEED, ship_cruise_speed_mode)*(2.0*float(random.getrandbits(1)) - 1.0)  # random.triangular(0, SHIP_MAX_SPEED, SHIP_MAX_SPEED)*(2.0*float(random.getrandbits(1)) - 1.0)
                    ship_cruise_turn_rate = random.triangular(0, SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)*(2.0*float(random.getrandbits(1)) - 1.0)  # random.uniform(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)
                    # TODO: For denser asteroid fields, decrease the max cruise seconds to encourage shorter maneuvers!
                    # ship_cruise_timesteps = random.randint(1, round(max_cruise_seconds*FPS))
                    if isnan(ship_cruise_timesteps_mode):
                        ship_cruise_timesteps = random.randint(0, round(MAX_CRUISE_TIMESTEPS))
                    else:
                        ship_cruise_timesteps = floor(random.triangular(0.0, MAX_CRUISE_TIMESTEPS, ship_cruise_timesteps_mode))

                    '''
                    random_ship_heading_angle = random.triangular(-6.0*10.0, 6.0*10.0, 0)
                    ship_accel_turn_rate = random.triangular(0, SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)*(2.0*float(random.getrandbits(1)) - 1.0)
                    #random_ship_cruise_speed = random.uniform(-ship_max_speed, ship_max_speed)
                    ship_cruise_speed = random.uniform(-SHIP_MAX_SPEED, SHIP_MAX_SPEED)#random.triangular(0, SHIP_MAX_SPEED, SHIP_MAX_SPEED)*(2.0*float(random.getrandbits(1)) - 1.0)
                    ship_cruise_turn_rate = random.triangular(0, SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)*(2.0*float(random.getrandbits(1)) - 1.0)#random.uniform(-SHIP_MAX_TURN_RATE, SHIP_MAX_TURN_RATE)
                    # TODO: For denser asteroid fields, decrease the max cruise seconds to encourage shorter maneuvers!
                    #ship_cruise_timesteps = random.randint(1, round(max_cruise_seconds*FPS))
                    ship_cruise_timesteps = floor(random.triangular(0.0, max_cruise_seconds*FPS, 0.0))
                    '''

                # First do a dummy simulation just to go through the motion, so we have the list of moves
                #print(f"\nDoing the move shenanigans")
                preview_move_sequence = get_ship_maneuver_move_sequence(random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate, ship_cruise_timesteps, ship_cruise_turn_rate, self.game_state_to_base_planning['ship_state'].speed)
                maneuver_sim = Matrix(game_state=self.game_state_to_base_planning['game_state'],
                                      ship_state=self.game_state_to_base_planning['ship_state'],
                                      initial_timestep=self.game_state_to_base_planning['timestep'],
                                      respawn_timer=self.game_state_to_base_planning['ship_respawn_timer'],
                                      asteroids_pending_death=self.game_state_to_base_planning['asteroids_pending_death'],
                                      forecasted_asteroid_splits=self.game_state_to_base_planning['forecasted_asteroid_splits'],
                                      last_timestep_fired=self.game_state_to_base_planning['last_timestep_fired'],
                                      last_timestep_mined=self.game_state_to_base_planning['last_timestep_mined'],
                                      mine_positions_placed=self.game_state_to_base_planning['mine_positions_placed'],
                                      halt_shooting=False,
                                      fire_first_timestep=self.game_state_to_base_planning['fire_next_timestep_flag'],
                                      verify_first_shot=True if len(self.sims_this_planning_period) == 0 and other_ships_exist else False,
                                      verify_maneuver_shots=False,
                                      game_state_plotter=self.game_state_plotter)
                # While evaluating, the simulation is advancing, and if it crashes, then it'll evaluate to false and stop the sim.
                #print(preview_move_sequence)
                #if maneuver_sim.get_sim_id() == 333:
                #    print('\ncalling sim maneuver in plan maneuver')
                if maneuver_sim.simulate_maneuver(preview_move_sequence, True):
                    # The ship went through all the steps without colliding
                    #pre_fitness = maneuver_sim.get_fitness()
                    #pre_fitness_breakdown = maneuver_sim.get_fitness_breakdown()
                    #if maneuver_sim.get_sim_id() == 333:
                    #    print(f"After sim maneuver in plan maneuver, The ship went through all the steps without colliding and lasted {len(maneuver_sim.get_state_sequence())}")
                    # maneuver_complete_without_crash = True
                    pass
                else:
                    # The ship crashed somewhere before reaching the final resting spot
                    #pre_fitness = maneuver_sim.get_fitness()
                    #pre_fitness_breakdown = maneuver_sim.get_fitness_breakdown()
                    #if maneuver_sim.get_sim_id() == 333:
                    #    print(f"After sim maneuver in plan maneuver, The ship crashed somewhere before reaching the final resting spot and only lasted {len(maneuver_sim.get_state_sequence())}")
                    # maneuver_complete_without_crash = False
                    pass
                # print(f"Maneuver completed without crash: {maneuver_complete_without_crash}")
                maneuver_fitness = maneuver_sim.get_fitness()
                #print(f"Maneuver fitness: {maneuver_fitness}, maneuver tuple: {random_ship_heading_angle=} {ship_cruise_speed=} {ship_accel_turn_rate=} {ship_cruise_timesteps=} {ship_cruise_turn_rate=}")
                maneuver_fitness_breakdown = maneuver_sim.get_fitness_breakdown()
                #post_fitness = maneuver_sim.get_fitness()
                #print(f"{pre_fitness=} {pre_fitness_breakdown=}, {maneuver_fitness=} {maneuver_fitness_breakdown=}, {post_fitness=}")
                if len(self.sims_this_planning_period) == 0:
                    if maneuver_sim.get_cancel_firing_first_timestep():
                        # The plan was to first at the first timestep this planning period. However, due to non-determinism caused by the existence of another ship, this shot would actually miss. We checked and caught this, so we're going to just nix the idea of shooting on the first timestep.
                        self.game_state_to_base_planning['fire_next_timestep_flag'] = False
                '''
                global heuristic_fis_iterations
                global heuristic_fis_total_fitness
                global random_search_iterations
                global random_search_total_fitness
                if heuristic_maneuver:
                    heuristic_fis_iterations += 1
                    heuristic_fis_total_fitness += maneuver_fitness
                else:
                    random_search_iterations += 1
                    random_search_total_fitness += maneuver_fitness
                '''

                self.sims_this_planning_period.append({
                    'sim': maneuver_sim,
                    'fitness': maneuver_fitness,
                    'fitness_breakdown': maneuver_fitness_breakdown,
                    'action_type': 'heuristic_maneuver' if heuristic_maneuver else 'random_maneuver',
                    'state_type': 'exact' if base_state_is_exact else 'predicted',
                    'maneuver_tuple': (random_ship_heading_angle, ship_cruise_speed, ship_accel_turn_rate, ship_cruise_timesteps, ship_cruise_turn_rate)
                })
                # if heuristic_maneuver:
                #    debug_print(f"Heuristic maneuver got fitness of {maneuver_fitness}")
                # debug_print(f"Planning random maneuver, and got fitness {maneuver_fitness}")
                if maneuver_fitness > self.best_fitness_this_planning_period:
                    self.second_best_fitness_this_planning_period = self.best_fitness_this_planning_period
                    self.second_best_fitness_this_planning_period_index = self.best_fitness_this_planning_period_index

                    self.best_fitness_this_planning_period = maneuver_fitness
                    self.best_fitness_this_planning_period_index = len(self.sims_this_planning_period) - 1
                if heuristic_maneuver:
                    heuristic_maneuver = False
        # gc.enable()
        # gc.collect()

    def actions(self, ship_state_dict: dict[str, Any], game_state_dict: dict[str, Any]) -> tuple[float, float, bool, bool]:
        #global unwrap_cache_misses
        #global unwrap_cache_hits
        #if self.current_timestep % 3 == 0:
        #    global unwrap_cache
        #    unwrap_cache.clear()
        #print(f"Cache hits: {unwrap_cache_hits}, misses: {unwrap_cache_misses}")
        # Method processed each time step by this controller.
        self.current_timestep += 1
        recovering_from_crash = False
        #print(f"Calling Neo's actions() on timestep {game_state_dict['sim_frame']}, and Neo thinks it's timestep {self.current_timestep}")
        #print(self.action_queue)
        ship_state = create_ship_from_dict(cast(ShipDict, ship_state_dict))
        game_state = create_game_state_from_dict(cast(GameStateDict, game_state_dict))

        if CLEAN_UP_STATE_FOR_SUBSEQUENT_SCENARIO_RUNS or STATE_CONSISTENCY_CHECK_AND_RECOVERY:
            timestep_mismatch: bool = not game_state.sim_frame == self.current_timestep
            # Amid running the scenario, the action queue is desynced with our timestep. This may be caused by an exception that was raised in Neo which was caught by Kessler, so the actions for this timestep were never consumed.
            action_queue_desync: bool = len(self.action_queue) > 0 and self.action_queue[0][0] != self.current_timestep
            planning_base_state_outdated: bool = self.game_state_to_base_planning is not None and self.game_state_to_base_planning['timestep'] < self.current_timestep
            if timestep_mismatch or (STATE_CONSISTENCY_CHECK_AND_RECOVERY and (action_queue_desync or planning_base_state_outdated)):
                if timestep_mismatch and not (action_queue_desync or planning_base_state_outdated):
                    print("This was not a fresh run of the controller! I'll try cleaning up the previous run and reset the state.")
                elif timestep_mismatch:
                    print(f"Neo didn't start from time 0. Was there a controller exception? Setting timestep to match the passed-in game state's nonzero starting timestep of: {game_state.sim_frame}")
                self.reset()
                self.current_timestep += 1
                if STATE_CONSISTENCY_CHECK_AND_RECOVERY and (action_queue_desync or planning_base_state_outdated):
                    print("Neo probably crashed or something because the internal state is all messed up. Welp, let's try this again.")
                    recovering_from_crash = True
                if timestep_mismatch:
                    self.current_timestep = game_state.sim_frame

        if self.current_timestep == 0:
            # Only do these on the first timestep
            inspect_scenario(game_state, ship_state)
        # debug_print(f"\n\nTimestep {self.current_timestep}, ship id {ship_state.id} is at {ship_state.position[0]} {ship_state.position[1]}")

        if not self.init_done:
            self.finish_init(game_state, ship_state)
            self.init_done = True
        self.performance_controller_enter()
        iterations_boost = False
        if self.current_timestep == 0:
            iterations_boost = True
        if self.other_ships_exist:
            # We cannot use deterministic mode to plan ahead
            # We can still try to plan ahead, but we need to compare the predicted state with the actual state
            # Note that if the other ship dies, then we will switch from this case to the case where other ships don't exist

            # Since other ships exist right now and the game isn't deterministic, we can die at any time even during the middle of a planned maneuver where we SHOULD survive.
            # Or maybe we planned to die at the end of the maneuver, but we died in the middle instead. That's a sneaky case that's possible too. Handle all of these!
            # Check for that case:
            unexpected_death = False
            # If we're dead/respawning but we didn't plan a respawn maneuver for it, OR if we do expect to die at the end of the maneuver, however we actually died mid-maneuver
            #print(f"{ship_state.is_respawning=}, ts: {self.current_timestep}, Action queue length: {len(self.action_queue)}")
            # Originally I thought it'd be a necessary condition to check (not self.last_timestep_ship_is_respawning and ship_state.is_respawning and ship_state.lives_remaining not in self.lives_remaining_that_we_did_respawn_maneuver_for) however WE DO NOT want to check that the last timestep we weren't respawning!
            # Because a sneaky edge case is, what if we did a respawn maneuver, and then we began to shoot in the middle of the respawn maneuver RIGHT AS the other ship is inside of us? Then we stay in the respawning state without ever getting out of it, but we just lose a life. Losing a life is the main thing we need to check for! And yes, this is an edge case I experienced and spent an hour tracking down.
            if (ship_state.is_respawning and ship_state.lives_remaining not in self.lives_remaining_that_we_did_respawn_maneuver_for) or (self.action_queue and not self.last_timestep_ship_is_respawning and ship_state.is_respawning and ship_state.lives_remaining in self.lives_remaining_that_we_did_respawn_maneuver_for):
                print(f"Ouch, I died in the middle of a maneuver where I expected to survive, due to other ships being present! We have {ship_state.lives_remaining} lives left, and here's the set of lives left we did respawn maneuvers for: {self.lives_remaining_that_we_did_respawn_maneuver_for}")
                # Clear the move queue, since previous moves have been invalidated by us taking damage
                self.action_queue.clear()
                self.actioned_timesteps.clear()  # If we don't clear it, we'll have duplicated moves since we have to overwrite our planned moves to get to safety, which means enqueuing moves on timesteps we already enqueued moves for.
                self.fire_next_timestep_flag = False  # If we were planning on shooting this timestep but we unexpectedly got hit, DO NOT SHOOT! Actually even if we didn't reset this variable here, we'd only shoot after the respawn maneuver is done and then we'd miss a shot. And yes that was a bug that I fixed lmao
                # self.game_state_to_base_planning = None
                self.sims_this_planning_period.clear()
                self.best_fitness_this_planning_period_index = INT_NEG_INF
                self.best_fitness_this_planning_period = -inf
                self.second_best_fitness_this_planning_period_index = INT_NEG_INF
                self.second_best_fitness_this_planning_period = -inf
                self.base_gamestate_analysis = None
                unexpected_death = True
                iterations_boost = True
                if ship_state.lives_remaining in self.lives_remaining_that_we_did_respawn_maneuver_for:
                    # We expected to die at the end of the maneuver, however we actually died mid-maneuver, so we have to revoke the respawn maneuver we had planned, and plan a new one.
                    # Removing the life remaining number from this set will allow us to plan a new maneuver for this number of lives remaining
                    print("GOTCHA, this life remaining shouldn't be in here! Yoink!")
                    self.lives_remaining_that_we_did_respawn_maneuver_for.remove(ship_state.lives_remaining)
            unexpected_survival = False
            # If we're alive at the end of a maneuver but we're expecting to be dead at the end of the maneuver and we've planned a respawn maneuver
            #if self.game_state_to_base_planning is not None:
            #    print(f"Checking for unexpected survival: {ship_state.is_respawning=} {self.game_state_to_base_planning['ship_state'].is_respawning=} {self.game_state_to_base_planning['respawning']=}")
            if not self.action_queue and self.game_state_to_base_planning is not None and not ship_state.is_respawning and self.game_state_to_base_planning['ship_state'].is_respawning and self.game_state_to_base_planning['respawning']:
                # We thought this maneuver would end in us dying, with the next move being a respawn maneuver. However this is not the case. We're alive at the end of the maneuver! This must be because the other ship saved us by shooting an asteroid that was going to hit us, or something.
                # This assertion isn't true because we could be doing a respawn maneuver, dying, and doing another respawn maneuver!
                # assert not self.last_timestep_ship_is_respawning
                print_explanation(f"\nI thought I would die, but the other ship saved me!!!", self.current_timestep)
                # Clear the move queue, since previous moves have been invalidated by us taking damage
                self.action_queue.clear()
                self.actioned_timesteps.clear()  # If we don't clear it, we'll have duplicated moves since we have to overwrite our planned moves to get to safety, which means enqueuing moves on timesteps we already enqueued moves for.
                self.fire_next_timestep_flag = False  # This should be false anyway!
                # self.game_state_to_base_planning = None
                self.sims_this_planning_period.clear()
                self.best_fitness_this_planning_period_index = INT_NEG_INF
                self.best_fitness_this_planning_period = -inf
                self.second_best_fitness_this_planning_period_index = INT_NEG_INF
                self.second_best_fitness_this_planning_period = -inf
                self.base_gamestate_analysis = None
                iterations_boost = True
                unexpected_survival = True
                # Yoink this life remaining from the respawn maneuvers, since we no longer are doing one
                if (ship_state.lives_remaining - 1) in self.lives_remaining_that_we_did_respawn_maneuver_for:
                    # We need to subtract one from the lives remaining, because when we added it, it was from a simulated ship that had one fewer life. In reality we never lost that life, so we subtract one from our actual lives.
                    self.lives_remaining_that_we_did_respawn_maneuver_for.remove(ship_state.lives_remaining - 1)
            # set up the actions planning
            if unexpected_death:
                # We need to refresh the state if we died unexpectedly
                print_explanation(f"\nOuch! Due to the other ship, I unexpectedly died!", self.current_timestep)
                assert self.game_state_to_base_planning is not None
                self.game_state_to_base_planning = {
                    'timestep': self.current_timestep,
                    'respawning': ship_state.is_respawning and ship_state.lives_remaining not in self.lives_remaining_that_we_did_respawn_maneuver_for,
                    'ship_state': ship_state,
                    'game_state': game_state,
                    'ship_respawn_timer': 3.0,
                    'asteroids_pending_death': {},
                    'forecasted_asteroid_splits': [],
                    'last_timestep_fired': self.game_state_to_base_planning['last_timestep_fired'],
                    'last_timestep_mined': self.game_state_to_base_planning['last_timestep_mined'],
                    'mine_positions_placed': self.game_state_to_base_planning['mine_positions_placed'],
                    'fire_next_timestep_flag': False,
                }

                if self.game_state_to_base_planning['respawning']:
                    print(f"Adding to lives remaining that we did respawn for, in the unexpected death: {ship_state.lives_remaining}")
                    self.lives_remaining_that_we_did_respawn_maneuver_for.add(ship_state.lives_remaining)
            elif unexpected_survival:
                print(f"Unexpected survival, the ship state is {ship_state}")
                # We need to refresh the state if we survived unexpectedly. Technically if we still had the remainder of the maneuver from before we could use that, but it's easier to just make a new maneuver from this starting point.
                assert self.game_state_to_base_planning is not None
                self.game_state_to_base_planning = {
                    'timestep': self.current_timestep,
                    'respawning': ship_state.is_respawning and ship_state.lives_remaining not in self.lives_remaining_that_we_did_respawn_maneuver_for,
                    'ship_state': ship_state,
                    'game_state': game_state,
                    'ship_respawn_timer': 0.0,
                    'asteroids_pending_death': {},
                    'forecasted_asteroid_splits': [],
                    'last_timestep_fired': self.game_state_to_base_planning['last_timestep_fired'],
                    'last_timestep_mined': self.game_state_to_base_planning['last_timestep_mined'],
                    'mine_positions_placed': self.game_state_to_base_planning['mine_positions_placed'],
                    'fire_next_timestep_flag': False,
                }
            elif not self.game_state_to_base_planning:
                self.game_state_to_base_planning = {
                    'timestep': self.current_timestep,
                    'respawning': ship_state.is_respawning and ship_state.lives_remaining not in self.lives_remaining_that_we_did_respawn_maneuver_for,
                    'ship_state': ship_state,
                    'game_state': game_state,
                    'ship_respawn_timer': 0.0,
                    'asteroids_pending_death': {},
                    'forecasted_asteroid_splits': [],
                    'last_timestep_fired': INT_NEG_INF,
                    'last_timestep_mined': INT_NEG_INF,
                    'mine_positions_placed': set(),
                    'fire_next_timestep_flag': False,
                }
                if self.game_state_to_base_planning['respawning']:
                    # print(f"Adding to lives remaining that we did respawn for, in actions: {ship_state.lives_remaining}")
                    self.lives_remaining_that_we_did_respawn_maneuver_for.add(ship_state.lives_remaining)

            if self.action_queue:
                self.plan_action(self.other_ships_exist, False, iterations_boost, False)
            else:
                # Refresh the base state now that we have the true base state!
                # debug_print('REFRESHING BASE STATE FOR STATIONARY ON TS', self.current_timestep)
                self.game_state_to_base_planning['ship_state'] = ship_state
                self.game_state_to_base_planning['game_state'] = game_state
                if not self.game_state_to_base_planning['ship_state'].is_respawning and bool(self.game_state_to_base_planning['ship_respawn_timer']):
                    # We're not respawning but the ship respawn timer is non-zero, so we're gonna fix this and make it consistent!
                    self.game_state_to_base_planning['ship_respawn_timer'] = 0.0
                # When there's other ships, stationary targetting is the LAST thing done, just so it can be based off of the reality state
                # The base state is exact on the final planning timestep, since the base state is the state we're on right now
                # if len(get_other_ships(game_state, self.ship_id_internal)) == 0:
                #    debug_print("\n\nWe're alone already. Injecting the following game state:")
                #    debug_print(game_state)
                self.plan_action(self.other_ships_exist, True, iterations_boost, True)
                assert self.best_fitness_this_planning_period_index is not None
                #index_according_to_lives_remaining = min(3, self.game_state_to_base_planning['ship_state'].lives_remaining)
                while len(self.sims_this_planning_period) < (get_min_maneuver_per_period_search_iterations_if_will_die(self.game_state_to_base_planning['ship_state'].lives_remaining, weighted_average(overall_fitness_record)) if self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness_breakdown'][5] == 0.0 else (get_min_respawn_per_period_search_iterations(self.game_state_to_base_planning['ship_state'].lives_remaining, weighted_average(overall_fitness_record)) if self.game_state_to_base_planning['respawning'] else get_min_maneuver_per_period_search_iterations(self.game_state_to_base_planning['ship_state'].lives_remaining, weighted_average(overall_fitness_record)))):
                    # Planning extra iterations to reach minimum threshold!
                    # print(f"Planning extra iterations to reach minimum threshold! {len(self.sims_this_planning_period)}")
                    self.plan_action(self.other_ships_exist, True, False, False)
                self.decide_next_action(game_state, ship_state)  # Since other ships exist and this is non-deterministic, we constantly feed in the updated reality
                if len(get_other_ships(game_state, self.ship_id_internal)) == 0:
                    # The other ship just died. I'm now alone!
                    print_explanation("I'm alone. I can see into the future perfectly now!", self.current_timestep)
                    self.simulated_gamestate_history.clear()
                    self.set_of_base_gamestate_timesteps.clear()
                    self.other_ships_exist = False
        else:
            # No other ships exist, we're deterministically planning the future
            if not self.game_state_to_base_planning:
                # set up the actions planning
                if self.current_timestep == 0 or recovering_from_crash:
                    iterations_boost = True
                self.game_state_to_base_planning = {
                    'timestep': self.current_timestep,
                    'respawning': False,  # On the first timestep 0, the is_respawning flag is ALWAYS false, even if we spawn inside asteroids.
                    'ship_state': ship_state,
                    'game_state': game_state,
                    'ship_respawn_timer': 0,
                    'asteroids_pending_death': {},
                    'forecasted_asteroid_splits': [],
                    'last_timestep_fired': INT_NEG_INF if not recovering_from_crash else self.current_timestep - 1,  # self.current_timestep - 1, # TODO: CHECK EDGECASE, may need to restore to larger number to be safe
                    'last_timestep_mined': INT_NEG_INF if not recovering_from_crash else self.current_timestep - 1,
                    'mine_positions_placed': set(),
                    'fire_next_timestep_flag': False,
                }
                if recovering_from_crash:
                    print(f"Recovering from crash! Setting the base gamestate. The timestep is {self.current_timestep}")
            # No matter what, spend some time evaluating the best action from the next predicted state
            # When no ships are around, the stationary targetting is the first thing done
            if not self.sims_this_planning_period:
                self.plan_action(self.other_ships_exist, True, iterations_boost, True)
            else:
                self.plan_action(self.other_ships_exist, True, iterations_boost, False)
            if not self.action_queue:
                assert self.best_fitness_this_planning_period_index is not None
                #index_according_to_lives_remaining = min(3, self.game_state_to_base_planning['ship_state'].lives_remaining)
                while len(self.sims_this_planning_period) < (get_min_maneuver_per_period_search_iterations_if_will_die(self.game_state_to_base_planning['ship_state'].lives_remaining, weighted_average(overall_fitness_record)) if self.sims_this_planning_period[self.best_fitness_this_planning_period_index]['fitness_breakdown'][5] == 0.0 else (get_min_respawn_per_period_search_iterations(self.game_state_to_base_planning['ship_state'].lives_remaining, weighted_average(overall_fitness_record)) if self.game_state_to_base_planning['respawning'] else get_min_maneuver_per_period_search_iterations(self.game_state_to_base_planning['ship_state'].lives_remaining, weighted_average(overall_fitness_record)))):
                    # Planning extra iterations to reach minimum threshold!
                    # print("Planning extra iterations to reach minimum threshold!")
                    self.plan_action(self.other_ships_exist, True, False, False)
                # Nothing's in the action queue. Evaluate the current situation and figure out the best course of action
                if not self.current_timestep == self.game_state_to_base_planning['timestep'] and not recovering_from_crash:
                    raise Exception(f"The actions queue is empty, however the base state's timestep {self.game_state_to_base_planning['timestep']} doesn't match the current timestep {self.current_timestep}! That's weird.")
                # debug_print("Decide the next action.")
                self.decide_next_action(game_state, ship_state)

        # Execute the actions in the queue for this timestep
        if self.action_queue and self.action_queue[0][0] == self.current_timestep:
            _, thrust, turn_rate, fire, drop_mine = self.action_queue.popleft()
        else:
            raise Exception(f"Sequence error on timestep {self.current_timestep}!")
            thrust, turn_rate, fire, drop_mine = 0.0, 0.0, False, False

        # The next action in the queue is for a future timestep. All actions for this timestep are processed.


        self.performance_controller_exit()
        self.last_timestep_ship_is_respawning = ship_state.is_respawning
        return thrust, turn_rate, fire, drop_mine


if __name__ == '__main__':
    print("This is a Kessler controller meant to be imported, and not run directly!")

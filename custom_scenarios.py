from src.kesslergame import Scenario
import random
import numpy as np

width, height = (1920, 1080)
target_priority_optimization1 = Scenario(name='Target priority optimization 1',
                            asteroid_states=[{'position': (width*5//100, height*51//100), 'speed': 200, 'angle': 180, 'size': 1},
                                             {'position': (width*5//100, height*49//100), 'speed': 100, 'angle': 0, 'size': 1}],
                            ship_states=[
                                {'position': (width//2, height//2), 'angle': 0, 'lives': 1, 'team': 1, "mines_remaining": 3},
                            ],
                            map_size=(width, height),
                            #seed=2,
                            time_limit=30,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Parameters for the ring of asteroids
R_initial = 200  # Initial radius of the ring, large enough to enclose the ship
num_asteroids = 20  # Number of asteroids in the ring
speed = 40  # Speed at which asteroids close in

# Ship's initial position (center of the screen)
ship_position = (500, 400)

# Calculating initial positions and angles for the asteroids
theta = np.linspace(0, 2 * np.pi, num_asteroids, endpoint=False)
ast_x = [R_initial * np.cos(angle) + ship_position[0] for angle in theta]
ast_y = [R_initial * np.sin(angle) + ship_position[1] for angle in theta]
init_angle = [np.rad2deg(np.arctan2(ship_position[1] - y, ship_position[0] - x)) for x, y in zip(ast_x, ast_y)]

# Creating asteroid states
asteroid_states = [{"position": (x, y), "angle": angle, "speed": speed} for x, y, angle in zip(ast_x, ast_y, init_angle)]

# Creating the scenario
closing_ring_scenario = Scenario(
    name="closing_ring_scenario",
    asteroid_states=asteroid_states,
    ship_states=[{"position": ship_position, 'lives': 10, 'team': 1, "mines_remaining": 3}],
    seed=0
)









# Parameters for the ring of asteroids
R_initial = 200  # Initial radius of the ring, large enough to enclose the ship
num_asteroids = 10  # Number of asteroids in the ring
speed = 40  # Speed at which asteroids close in

# Ship's initial position (center of the screen)
ship_position = (500, 400)

# Calculating initial positions and angles for the asteroids
theta = np.linspace(0, 2 * np.pi, num_asteroids, endpoint=False)
ast_x = [R_initial * np.cos(angle) + ship_position[0] for angle in theta]
ast_y = [R_initial * np.sin(angle) + ship_position[1] for angle in theta]
init_angle = [np.rad2deg(np.arctan2(ship_position[1] - y, ship_position[0] - x)) for x, y in zip(ast_x, ast_y)]

# Creating asteroid states
asteroid_states = [{"position": (x, y), "angle": angle, "speed": speed, 'size': 3} for x, y, angle in zip(ast_x, ast_y, init_angle)]

# Creating the scenario
easy_closing_ring_scenario = Scenario(
    name="easy_closing_ring_scenario",
    asteroid_states=asteroid_states,
    ship_states=[{"position": ship_position, 'lives': 5, 'team': 1, "mines_remaining": 1}],
    seed=0
)







# Parameters for the dense and fast-closing ring of asteroids
R_initial = 400  # Increased initial radius of the ring
num_asteroids = 40  # More asteroids for a denser ring
speed = 60  # Increased speed for faster closing

# Ship's initial position (center of the screen)
ship_position = (500, 400)

# Calculating initial positions and angles for the asteroids
theta = np.linspace(0, 2 * np.pi, num_asteroids, endpoint=False)
ast_x = [R_initial * np.cos(angle) + ship_position[0] for angle in theta]
ast_y = [R_initial * np.sin(angle) + ship_position[1] for angle in theta]
init_angle = [np.rad2deg(np.arctan2(ship_position[1] - y, ship_position[0] - x)) for x, y in zip(ast_x, ast_y)]

# Creating asteroid states
asteroid_states = [{"position": (x, y), "angle": angle, "speed": speed} for x, y, angle in zip(ast_x, ast_y, init_angle)]

# Creating the scenario with additional ship states
more_intense_closing_ring_scenario = Scenario(
    name="more_intense_closing_ring_scenario",
    asteroid_states=asteroid_states,
    ship_states=[{
        "position": ship_position,
        "lives": 10, 
        "team": 1, 
        "mines_remaining": 3
    }],
    seed=0
)





def calculate_angle(from_pos, to_pos):
    """Calculate the angle for movement from from_pos to to_pos."""
    dx, dy = np.array(to_pos) - np.array(from_pos)
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad) % 360
    return angle_deg

# Parameters for the rotating square/diamond
center = (500, 400)  # Center of the screen
size = 200  # Side length of the square
speed = 30  # Speed of the asteroids

# Calculate corner positions of the square
corners = [
    (center[0] - size / 2, center[1] - size / 2),
    (center[0] + size / 2, center[1] - size / 2),
    (center[0] + size / 2, center[1] + size / 2),
    (center[0] - size / 2, center[1] + size / 2)
]

# Create asteroid states with initial positions and angles
asteroid_states = []
for i in range(len(corners)):
    next_corner = corners[(i + 1) % len(corners)]
    angle = calculate_angle(corners[i], next_corner)
    asteroid_states.append({
        "position": corners[i],
        "angle": angle,
        "speed": speed
    })

# Create the scenario
rotating_square_scenario = Scenario(
    name="rotating_square_scenario",
    asteroid_states=asteroid_states,
    ship_states=[{"position": center, 'lives': 3, 'mines_remaining': 1}],  # Add additional ship states as needed
    seed=0
)











def calculate_angle(from_pos, to_pos):
    """Calculate the angle for movement from from_pos to to_pos."""
    dx, dy = np.array(to_pos) - np.array(from_pos)
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad) % 360
    return angle_deg

# Parameters for the rotating square/diamond
center = (500, 400)  # Center of the screen
size = 180  # Side length of the square
speed = 20  # Speed of the asteroids

# Calculate corner positions of the square
corners = [
    (center[0] - size / 2, center[1] - size / 2),
    (center[0] + size / 2, center[1] - size / 2),
    (center[0] + size / 2, center[1] + size / 2),
    (center[0] - size / 2, center[1] + size / 2)
]

# Create asteroid states with initial positions and angles
asteroid_states = []
for i in range(len(corners)):
    next_corner = corners[(i + 1) % len(corners)]
    angle = calculate_angle(corners[i], next_corner)
    asteroid_states.append({
        "position": corners[i],
        "angle": angle,
        "speed": speed
    })
asteroid_states *= 3
# Create the scenario
rotating_square_2_overlap = Scenario(
    name="rotating_square_2_overlap",
    asteroid_states=asteroid_states,
    ship_states=[{"position": center, 'lives': 3, 'mines_remaining': 2}],  # Add additional ship states as needed
    seed=0
)











def calculate_diagonal_angle(direction):
    """Calculate the angle for diagonal movement based on the given direction."""
    if direction == 'left':
        return 225  # Diagonal left (down and left)
    else:  # direction == 'right'
        return 135  # Diagonal right (down and right)

# Parameters for the Falling Leaves scenario
screen_width = 1000
start_y = 0  # Starting height (top of the screen)
speed = 100  # Speed of the asteroids
spacing = 100  # Horizontal spacing between asteroids
num_asteroids = screen_width // spacing

# Create asteroid states
asteroid_states = []
for i in range(num_asteroids):
    start_x = i * spacing
    direction = 'left' if i % 2 == 0 else 'right'
    angle = calculate_diagonal_angle(direction)
    asteroid_states.append({
        "position": (start_x, start_y),
        "angle": angle,
        "speed": speed
    })

# Create the scenario
falling_leaves_scenario = Scenario(
    name="falling_leaves_scenario",
    asteroid_states=asteroid_states,
    ship_states=[{"position": (500, 400)}],  # Update with your ship's initial position
    seed=0
)








def zigzag_angle(row):
    """Determine the angle for the asteroid's zigzag motion based on its row."""
    if row % 2 == 0:
        return 45  # Moving diagonally down and to the right
    else:
        return 135  # Moving diagonally down and to the left

# Parameters for the Zigzag Motion scenario
screen_height = 800
speed = 100  # Speed of the asteroids
spacing = 50  # Vertical spacing between rows of asteroids
num_rows = screen_height // spacing

# Create asteroid states
asteroid_states = []
for row in range(num_rows):
    y_position = row * spacing
    angle = zigzag_angle(row)
    asteroid_states.append({
        "position": (0, y_position),  # Starting from the left edge
        "angle": angle,
        "speed": speed
    })

# Create the scenario
zigzag_motion_scenario = Scenario(
    name="zigzag_motion_scenario",
    asteroid_states=asteroid_states,
    ship_states=[{"position": (500, 400), 'lives': 5, 'mines_remaining': 3}],  # Update with your ship's initial position
    seed=0
)










def calculate_speed_and_angle(y_position, base_speed, speed_increment, center_y):
    """Determine the speed and angle based on the asteroid's vertical position."""
    distance_from_center = abs(y_position - center_y)
    speed = base_speed + speed_increment * (distance_from_center / vertical_spacing)
    
    if y_position < center_y:  # Above the center row
        return speed, 180  # Moving to the left
    elif y_position > center_y:  # Below the center row
        return speed, 0    # Moving to the right
    else:  # Center row
        return 0, 0  # Stationary

# Parameters for the revised Shearing Pattern scenario
screen_width, screen_height = 1000, 800
center_y = screen_height / 2
base_speed = 0  # Base speed for asteroids closest to the center
speed_increment = 18  # Additional speed for each row away from the center
vertical_spacing = 180  # Vertical spacing between rows
horizontal_spacing = 100  # Horizontal spacing within each row

# Create asteroid states
asteroid_states = []
for y_position in range(0, screen_height, vertical_spacing):
    speed, angle = calculate_speed_and_angle(y_position, base_speed, speed_increment, center_y)
    for x_position in range(0, screen_width, horizontal_spacing):
        asteroid_states.append({
            "position": (x_position, y_position),
            "angle": angle,
            "speed": speed
        })

# Create the scenario
shearing_pattern_scenario = Scenario(
    name="shearing_pattern_scenario",
    asteroid_states=asteroid_states,
    ship_states=[{"position": (500, 400), "lives": 3, "mines_remaining": 3}],
    seed=0
)

















num_asteroids = 18
side_padding = 60
asteroid_spacing = (width - 2*side_padding) // num_asteroids  # Spacing between asteroids
asteroid_speed = 90  # Constant speed of the asteroids
asteroid_start_y = 100  # Starting height of the wall

# Create asteroid states
asteroid_states = []
for i in range(num_asteroids):
    x_position = side_padding + i * asteroid_spacing + asteroid_spacing / 2  # Centering each asteroid
    asteroid_states.append({
        'position': (x_position, asteroid_start_y),
        'angle': -90,  # Moving straight down
        'speed': asteroid_speed,
        'size': 3,
    })

# Create the scenario
super_hard_wrap = Scenario(
    name='adv_multi_wall_bottom_hard_1',
    asteroid_states=asteroid_states,
    ship_states=[{'position': (width//4, height*9//10), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 0}],
    map_size=(width, height),
    time_limit=30,
)




super_fast_asteroid = Scenario(name='Test Scenario',
                            #num_asteroids=200,
                            asteroid_states=[{'position': (width/2, height*2/3), 'speed': 100+width*30, 'angle': 0, 'size': 3}],
                            #                {'position': (width*2//3, height*40//100), 'speed': 100, 'angle': -91, 'size': 4},
                            #                 {'position': (width*1//3, height*40//100), 'speed': 100, 'angle': -91, 'size': 4}],
                            ship_states=[
                                {'position': (width//2, height//2), 'angle': 0, 'lives': 5, 'team': 1, "mines_remaining": 0},
                                #{'position': (width*2//3, height//2), 'angle': 90, 'lives': 10, 'team': 2, "mines_remaining": 10},
                            ],
                            map_size=(width, height),
                            #seed=2,
                            time_limit=np.inf,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)




edge_asteroid = Scenario(name='Test Scenario',
                            #num_asteroids=200,
                            asteroid_states=[{'position': (0, 0), 'speed': 0, 'angle': 0, 'size': 1}],
                            #                {'position': (width*2//3, height*40//100), 'speed': 100, 'angle': -91, 'size': 4},
                            #                 {'position': (width*1//3, height*40//100), 'speed': 100, 'angle': -91, 'size': 4}],
                            ship_states=[
                                {'position': (width//2, height//2), 'angle': 0, 'lives': 5, 'team': 1, "mines_remaining": 0},
                                #{'position': (width*2//3, height//2), 'angle': 90, 'lives': 10, 'team': 2, "mines_remaining": 10},
                            ],
                            map_size=(width, height),
                            #seed=2,
                            time_limit=np.inf,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)






# Parameters for the ring of asteroids
R_initial = 200  # Initial radius of the ring, large enough to enclose the ship
num_asteroids = 20  # Number of asteroids in the ring
speed = 20  # Speed at which asteroids close in

# Ship's initial position (center of the screen)
ship_position = (500, 400)

# Calculating initial positions and angles for the asteroids
theta = np.linspace(0, 2 * np.pi, num_asteroids, endpoint=False)
ast_x = [R_initial * np.cos(angle) + ship_position[0] for angle in theta]
ast_y = [R_initial * np.sin(angle) + ship_position[1] for angle in theta]
init_angle = [np.rad2deg(np.arctan2(ship_position[1] - y, ship_position[0] - x)) + random.triangular(-90, 90, 0) for x, y in zip(ast_x, ast_y)]

# Creating asteroid states
asteroid_states = [{"position": (x, y), "angle": angle, "speed": speed, 'size': 3} for x, y, angle in zip(ast_x, ast_y, init_angle)]

# Creating the scenario
wonky_ring = Scenario(
    name="wonky_ring",
    asteroid_states=asteroid_states,
    ship_states=[{"position": ship_position, 'lives': 3, 'team': 1, "mines_remaining": 3}],
    seed=0
)












def polar_to_cartesian(angle, speed):
    """Convert polar coordinates (angle, speed) to Cartesian (vx, vy)."""
    angle_rad = np.radians(angle)
    vx = speed * np.cos(angle_rad)
    vy = speed * np.sin(angle_rad)
    return vx, vy

def cartesian_to_polar(vx, vy):
    """Convert Cartesian coordinates (vx, vy) to polar (angle, speed)."""
    speed = np.sqrt(vx**2 + vy**2)
    angle = np.degrees(np.arctan2(vy, vx)) % 360
    return angle, speed

# Parameters for the moving ring scenario
R_initial = 400
num_asteroids = 20
closing_speed = -40
horizontal_speed = 100
ship_position = (960, 540)

# Calculate initial positions and angles for the asteroids
theta = np.linspace(0, 2 * np.pi, num_asteroids, endpoint=False)
asteroid_states = []
for angle in theta:
    x = R_initial * np.cos(angle) + ship_position[0]
    y = R_initial * np.sin(angle) + ship_position[1]
    init_angle = np.rad2deg(angle) % 360

    # Convert to Cartesian, adjust vx, and convert back to polar
    vx, vy = polar_to_cartesian(init_angle, closing_speed)
    vx += horizontal_speed  # Adding horizontal movement
    angle, speed = cartesian_to_polar(vx, vy)

    asteroid_states.append({'position': (x, y), 'angle': angle, 'speed': speed, 'size': 3})

# Create the Moving Ring scenario
moving_ring_scenario = Scenario(
    name='moving_ring_scenario',
    asteroid_states=asteroid_states,
    ship_states=[{'position': ship_position, 'angle': 0, 'lives': 3, 'mines_remaining': 3}],
    map_size=(1920, 1080),
    seed=0
)











def cartesian_to_polar(vx, vy):
    """Convert Cartesian coordinates (vx, vy) to polar (angle, speed)."""
    speed = np.sqrt(vx**2 + vy**2)
    angle = np.degrees(np.arctan2(vy, vx)) % 360
    return angle, speed

# Parameters for the shifting square scenario
side_length = 600
num_asteroids_per_side = 10
closing_speed = 20
horizontal_speed = 100
ship_position = (960, 540)

# Calculate initial positions for the asteroids
asteroid_states = []
for i in range(num_asteroids_per_side):
    # Top side
    x_top = ship_position[0] - side_length / 2 + i * (side_length / (num_asteroids_per_side - 1))
    asteroid_states.append({'position': (x_top, ship_position[1] - side_length / 2),
                            'angle': 0, 'speed': horizontal_speed})
    
    # Bottom side
    x_bottom = ship_position[0] - side_length / 2 + i * (side_length / (num_asteroids_per_side - 1))
    asteroid_states.append({'position': (x_bottom, ship_position[1] + side_length / 2),
                            'angle': 180, 'speed': horizontal_speed})
    
    # Left side (excluding corners)
    if i != 0 and i != num_asteroids_per_side - 1:
        y_left = ship_position[1] - side_length / 2 + i * (side_length / (num_asteroids_per_side - 1))
        asteroid_states.append({'position': (ship_position[0] - side_length / 2, y_left),
                                'angle': 270, 'speed': closing_speed})
        
    # Right side (excluding corners)
    if i != 0 and i != num_asteroids_per_side - 1:
        y_right = ship_position[1] - side_length / 2 + i * (side_length / (num_asteroids_per_side - 1))
        asteroid_states.append({'position': (ship_position[0] + side_length / 2, y_right),
                                'angle': 90, 'speed': closing_speed})

# Create the Shifting Square scenario
shifting_square_scenario = Scenario(
    name='shifting_square_scenario',
    asteroid_states=asteroid_states,
    ship_states=[{'position': ship_position, 'angle': 0, 'lives': 5, 'mines_remaining': 3}],
    map_size=(1920, 1080),
    seed=0
)








# Parameters for the ring of asteroids
R_initial = 800  # Initial radius of the ring, large enough to enclose the ship
num_asteroids = 20  # Number of asteroids in the ring
speed = 40  # Speed at which asteroids close in

# Ship's initial position (center of the screen)
ship_position = (500, 400)

# Calculating initial positions and angles for the asteroids
theta = np.linspace(0, 2 * np.pi, num_asteroids, endpoint=False)
ast_x = [R_initial * np.cos(angle) + ship_position[0] for angle in theta]
ast_y = [R_initial * np.sin(angle) + ship_position[1] for angle in theta]
init_angle = [np.rad2deg(np.arctan2(ship_position[1] - y, ship_position[0] - x)) for x, y in zip(ast_x, ast_y)]

# Creating asteroid states
asteroid_states = [{"position": (x, y), "angle": angle, "speed": speed, 'size': 4} for x, y, angle in zip(ast_x, ast_y, init_angle)]

# Creating the scenario
delayed_closing_ring_scenario = Scenario(
    name="delayed_closing_ring_scenario",
    asteroid_states=asteroid_states,
    ship_states=[{"position": ship_position, 'lives': 3, 'team': 1, "mines_remaining": 3}],
)









def spiral_position(center, radius, angle):
    """Calculate the x, y position for a given angle along a spiral."""
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    return x, y

def calculate_spiral_movement(angle, tightness):
    """Calculate the movement angle (tangent to the spiral) at a given point."""
    movement_angle = np.degrees(angle + np.pi/2) % 360  # Perpendicular to radius
    return movement_angle

# Parameters for the Spiral Assault scenario
center = (960, 540)
initial_radius = 600
num_asteroids = 50
spiral_tightness = 0.5  # Adjust for desired spiral tightness
asteroid_speed = 20  # Can be constant or variable

# Calculate initial positions for the asteroids
asteroid_states = []
for i in range(num_asteroids - 3):
    angle = i * spiral_tightness
    radius = initial_radius - i * (initial_radius / num_asteroids)
    position = spiral_position(center, radius, angle)
    movement_angle = calculate_spiral_movement(angle, spiral_tightness)
    asteroid_states.append({'position': position, 'angle': movement_angle, 'speed': asteroid_speed, 'size': 3})

# Create the Spiral Assault scenario
spiral_assault_scenario = Scenario(
    name='spiral_assault_scenario',
    asteroid_states=asteroid_states,
    ship_states=[{'position': center, 'angle': 0, 'lives': 3, 'mines_remaining': 2}],
    map_size=(1920, 1080),
)












# Parameters for the ring of asteroids
R_initial = 400  # Initial radius of the ring, large enough to enclose the ship
num_asteroids = 20  # Number of asteroids in the ring
speed = 40  # Speed at which asteroids close in

# Ship's initial position (center of the screen)
ship_position = (500, 400)

# Calculating initial positions and angles for the asteroids
theta = np.linspace(0, 2 * np.pi, num_asteroids, endpoint=False)
ast_x = [R_initial * np.cos(angle) + ship_position[0] for angle in theta]
ast_y = [R_initial * np.sin(angle) + ship_position[1] for angle in theta]
init_angle = [np.rad2deg(np.arctan2(ship_position[1] - y, ship_position[0] - x)) + 30 for x, y in zip(ast_x, ast_y)]

# Creating asteroid states
asteroid_states = [{"position": (x, y), "angle": angle, "speed": speed} for x, y, angle in zip(ast_x, ast_y, init_angle)]

# Creating the scenario
dancing_ring = Scenario(
    name="dancing_ring",
    asteroid_states=asteroid_states,
    ship_states=[{"position": ship_position, 'lives': 10, 'team': 1, "mines_remaining": 1}],
    seed=0
)





# Parameters for the ring of asteroids
R_initial = 400  # Initial radius of the ring, large enough to enclose the ship
num_asteroids = 20  # Number of asteroids in the ring
speed = 40  # Speed at which asteroids close in

# Ship's initial position (center of the screen)
ship_position = (500, 400)

# Calculating initial positions and angles for the asteroids
theta = np.linspace(0, 2 * np.pi, num_asteroids, endpoint=False)
ast_x = [R_initial * np.cos(angle) + ship_position[0] for angle in theta]
ast_y = [R_initial * np.sin(angle) + ship_position[1] for angle in theta]
offset = [len(theta)//2 - i for i in range(len(theta))]
init_angle = [np.rad2deg(np.arctan2(ship_position[1] - y, ship_position[0] - x)) + o*5 for x, y, o in zip(ast_x, ast_y, offset)]

# Creating asteroid states
asteroid_states = [{"position": (x, y), "angle": angle, "speed": speed} for x, y, angle in zip(ast_x, ast_y, init_angle)]

# Creating the scenario
dancing_ring_2 = Scenario(
    name="dancing_ring_2",
    asteroid_states=asteroid_states,
    ship_states=[{"position": ship_position, 'lives': 5, 'team': 1, "mines_remaining": 0}],
    seed=0
)








def create_diagonal_asteroids(start_pos, end_pos, num_asteroids, direction, speed):
    """Generate states for asteroids along a diagonal line."""
    positions = np.linspace(start_pos, end_pos, num_asteroids)
    #angle = np.degrees(np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])) % 360
    return [{'position': pos, 'angle': direction, 'speed': speed, 'size': 3} for pos in positions]

# Parameters for the Intersecting Lines scenario
width, height = 1920, 1080
num_asteroids = 20
speed = 200  # pixels per second

# Create asteroid lines
line1 = create_diagonal_asteroids((0, 0), (width, height), num_asteroids, 0, speed)  # Top-left to bottom-right
line2 = create_diagonal_asteroids((width, 0), (0, height), num_asteroids, 180, speed)  # Top-right to bottom-left

# Combine asteroid states
asteroid_states = line1 + line2

# Create the Intersecting Lines scenario
intersecting_lines_scenario = Scenario(
    name='intersecting_lines_scenario',
    asteroid_states=asteroid_states,
    ship_states=[{'position': (width/4, height/2), 'angle': 0, 'lives': 3, 'mines_remaining': 1}],#
                 #{'position': (width*3/4, height/2), 'angle': 0, 'lives': 3, 'mines_remaining': 1}],
    map_size=(width, height),
)




# Can cause a runtime warning due to divide by 0 if you drop a mine on the first timestep and get out of the way, since the distance between the asteroid and mine is 0 when the mine is exploding
minecrash = Scenario(name='Mine Crash',
                            asteroid_states=[{'position': (1920//2+30*3+1, 1080//2), 'speed': 30, 'angle': 180, 'size': 4}, {'position': (1920//2+30*3+1, 1000//2), 'speed': 30, 'angle': 180, 'size': 4}],
                            ship_states=[
                                {'position': (1920//2, 1080//2), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 1},
                                #{'position': (1920*2//3, 1080//2), 'angle': 90, 'lives': 10, 'team': 2, "mines_remaining": 10},
                            ],
                            map_size=(width, height),
                            #seed=2,
                            time_limit=500,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)














width, height = 1920, 1080
center = (width // 2, height // 2)
grid_size = 12
distance_factor = 0.5  # This factor will determine how speed increases with distance

def calculate_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def calculate_angle(center, position):
    dx, dy = position[0] - center[0], position[1] - center[1]
    return np.degrees(np.arctan2(dy, dx)) % 360

asteroid_states = []
grid_step = min(width, height) // (grid_size + 1)

for i in range(grid_size):
    for j in range(grid_size):
        asteroid_x = grid_step * (i + 1)
        asteroid_y = grid_step * (j + 1)
        position = (asteroid_x, asteroid_y)
        distance = calculate_distance(center, position)
        speed = distance_factor * distance
        angle = calculate_angle(center, position)

        asteroid_states.append({
            'position': position,
            'speed': speed,
            'angle': angle,
            'size': 1
        })

# Ship state
ship_state = [{'position': center, 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}]

# Creating the Exploding Grid scenario
exploding_grid_scenario = Scenario(
    name='exploding_grid_scenario',
    asteroid_states=asteroid_states,
    ship_states=ship_state,
    map_size=(width, height),
    seed=0
)

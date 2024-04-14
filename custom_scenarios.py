from src.kesslergame import Scenario
import random
import numpy as np
import math

width, height = (1000, 800)
target_priority_optimization1 = Scenario(name='Target priority optimization 1',
                            asteroid_states=[{'position': (width*5//100, height*51//100), 'speed': 200, 'angle': 180, 'size': 1},
                                             {'position': (width*5//100, height*49//100), 'speed': 100, 'angle': 0, 'size': 1}],
                            ship_states=[
                                {'position': (width//2, height//2), 'angle': 0, 'lives': 1, 'team': 1, "mines_remaining": 3},
                            ],
                            map_size=(width, height),
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
    ship_states=[{"position": ship_position, 'lives': 3, 'team': 1, "mines_remaining": 3}],
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
    ship_states=[{"position": ship_position, 'lives': 2, 'team': 1, "mines_remaining": 2}],
)







# Parameters for the dense and fast-closing ring of asteroids
R_initial = 400  # Increased initial radius of the ring
num_asteroids = 40  # More asteroids for a denser ring
speed = 60  # Increased speed for faster closing

# Ship's initial position (center of the screen)
ship_position_1 = (400, 400)
ship_position_2 = (600, 400)

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
        "position": ship_position_1,
        "lives": 10, 
        "team": 1, 
        "mines_remaining": 6
    },
    {
        "position": ship_position_2,
        "lives": 10, 
        "team": 2, 
        "mines_remaining": 6
    }],
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
    ship_states=[{"position": (500, 600), 'lives': 5, 'mines_remaining': 3, 'team': 1},
                 {"position": (500, 200), 'lives': 5, 'mines_remaining': 3, 'team': 2}],  # Update with your ship's initial position
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
speed_increment = 30  # Additional speed for each row away from the center
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
    ship_states=[{"position": (250, 400), "lives": 3, 'team': 1, "mines_remaining": 3},
                 {"position": (750, 400), "lives": 3, 'team': 2, "mines_remaining": 3}],
)

















num_asteroids = 18
side_padding = 0
asteroid_spacing = (width - 2*side_padding) // num_asteroids  # Spacing between asteroids
asteroid_speed = 160  # Constant speed of the asteroids
asteroid_start_y = 0  # Starting height of the wall

# Create asteroid states
asteroid_states = []
for i in range(num_asteroids):
    x_position = side_padding + i * asteroid_spacing + asteroid_spacing / 2  # Centering each asteroid
    asteroid_states.append({
        'position': (x_position, asteroid_start_y),
        'angle': -90,  # Moving straight down
        'speed': asteroid_speed,
        'size': 4,
    })

# Create the scenario
super_hard_wrap = Scenario(
    name='super_hard_wrap',
    asteroid_states=asteroid_states,
    ship_states=[{'position': (width/2, height*7//10), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 3}],
    map_size=(width, height),
    time_limit=240,
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
    ship_states=[{'position': ship_position, 'angle': 0, 'lives': 3, 'mines_remaining': 0}],
    map_size=(1000, 800),
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
    map_size=(1000, 800),
)








# Parameters for the ring of asteroids
R_initial = 1200  # Initial radius of the ring, large enough to enclose the ship
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
center = (500, 400)
initial_radius = 400
num_asteroids = 50
spiral_tightness = 0.5  # Adjust for desired spiral tightness
asteroid_speed = 50  # Can be constant or variable

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
    map_size=(1000, 800),
)












# Parameters for the ring of asteroids
R_initial = 400  # Initial radius of the ring, large enough to enclose the ship
num_asteroids = 20  # Number of asteroids in the ring
speed = 40  # Speed at which asteroids close in

# Ship's initial position (center of the screen)
#ship_position = (500, 400)

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
    ship_states=[{"position": (450, 400), 'lives': 3, 'team': 1, "mines_remaining": 1},
                 {"position": (550, 400), 'lives': 3, 'team': 2, "mines_remaining": 1}],
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
)








def create_diagonal_asteroids(start_pos, end_pos, num_asteroids, direction, speed):
    """Generate states for asteroids along a diagonal line."""
    positions = np.linspace(start_pos, end_pos, num_asteroids)
    #angle = np.degrees(np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])) % 360
    return [{'position': pos, 'angle': direction, 'speed': speed, 'size': 3} for pos in positions]

# Parameters for the Intersecting Lines scenario
width, height = 1000, 800
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
    ship_states=[{'position': (width/4, height/2), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 1},
                 {'position': (width*3/4, height/2), 'angle': 0, 'lives': 3, 'team': 2, 'mines_remaining': 1}],
    map_size=(width, height),
)












# Can cause a runtime warning due to divide by 0 if you drop a mine on the first timestep and get out of the way, since the distance between the asteroid and mine is 0 when the mine is exploding
minecrash = Scenario(name='Mine Crash',
                            asteroid_states=[{'position': (1000//2+30*3+1, 800//2), 'speed': 30, 'angle': 180, 'size': 4}, {'position': (1000//2+30*3+1, 1000//2), 'speed': 30, 'angle': 180, 'size': 4}],
                            ship_states=[
                                {'position': (1000//2, 800//2), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 1},
                                #{'position': (1000*2//3, 800//2), 'angle': 90, 'lives': 10, 'team': 2, "mines_remaining": 10},
                            ],
                            map_size=(width, height),
                            time_limit=500,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)














width, height = 1000, 800
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
)























width, height = 1000, 800
center = (width // 2, height // 2)
grid_size = 14
time_to_form_grid = 5  # Time in seconds after which the grid is formed
grid_spacing = 100  # Spacing between asteroids in the grid

asteroid_states = []

# Calculate the initial velocity for each asteroid to form a grid
for i in range(grid_size):
    for j in range(grid_size):
        # Target position in the grid
        target_x = center[0] + (i - grid_size // 2) * grid_spacing
        target_y = center[1] + (j - grid_size // 2) * grid_spacing
        target_position = (target_x, target_y)

        # Calculate velocity to reach the target position in the specified time
        velocity_x = (target_position[0] - center[0]) / time_to_form_grid
        velocity_y = (target_position[1] - center[1]) / time_to_form_grid

        # Convert velocity to polar coordinates (angle and speed)
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        angle = np.degrees(np.arctan2(velocity_y, velocity_x)) % 360

        asteroid_states.append({
            'position': center,
            'speed': speed,
            'angle': angle,
            'size': 1
        })

# Ship state
ship_state = [{'position': (0, 0), 'angle': 45, 'lives': 3, 'team': 1, 'mines_remaining': 3}]

# Creating the Grid Formation Explosion scenario
grid_formation_explosion_scenario = Scenario(
    name='grid_formation_explosion_scenario',
    asteroid_states=asteroid_states,
    ship_states=ship_state,
    map_size=(width, height),
)
















width, height = 1000, 800
center = (width // 2, height // 2)
time_to_form_grid = 4  # Time in seconds after which the grid is formed

# Calculate the number of asteroids based on aspect ratio
aspect_ratio = width / height
num_asteroids_width = 14  # Define the number of asteroids in width
num_asteroids_height = int(num_asteroids_width / aspect_ratio)  # Calculate the number in height based on aspect ratio

# Calculate grid spacing
grid_spacing_x = width / num_asteroids_width
grid_spacing_y = height / num_asteroids_height

asteroid_states = []

# Calculate the initial velocity for each asteroid to form a grid
for i in range(num_asteroids_width):
    for j in range(num_asteroids_height):
        # Target position in the grid
        target_x = (i * grid_spacing_x) + (grid_spacing_x / 2)
        target_y = (j * grid_spacing_y) + (grid_spacing_y / 2)
        target_position = (target_x, target_y)

        # Calculate velocity to reach the target position in the specified time
        velocity_x = (target_position[0] - center[0]) / time_to_form_grid
        velocity_y = (target_position[1] - center[1]) / time_to_form_grid

        # Convert velocity to polar coordinates (angle and speed)
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        angle = np.degrees(np.arctan2(velocity_y, velocity_x)) % 360

        asteroid_states.append({
            'position': center,
            'speed': speed,
            'angle': angle,
            'size': 2
        })

# Ship state
ship_state = [{'position': center, 'angle': 0, 'lives': 40, 'team': 1, 'mines_remaining': 3}]

# Creating the Aspect Ratio Grid Formation scenario
aspect_ratio_grid_formation_scenario = Scenario(
    name='aspect_ratio_grid_formation_scenario',
    asteroid_states=asteroid_states,
    ship_states=ship_state,
    map_size=(width, height),
)











width, height = 1000, 800

adv_asteroid_stealing = Scenario(
    name='adv_asteroid_stealing',
    asteroid_states=[{'position': (width*0.15, height*0.25), 'angle': 90.0, 'speed': 100, 'size': 3},
                     {'position': (width*0.85, height*0.75), 'angle': 270.0, 'speed': 100, 'size': 3},],
    ship_states=[{'position': (width*0.35, height*0.75), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 1},
                 {'position': (width*0.65, height*0.25), 'angle': 180, 'lives': 3, 'team': 2, "mines_remaining": 1}],
    time_limit=20,
    map_size=(width, height),
)










widths = np.linspace(0, width, 10)
heights = np.linspace(0, height, 10)

wrapping_nightmare = Scenario(
    name='wrapping_nightmare',
    asteroid_states=[{'position': (w, 0), 'angle': 180.0, 'speed': 100, 'size': 3} for w in widths] + 
                    [{'position': (w, height), 'angle': 0.0, 'speed': 100, 'size': 3} for w in widths] +
                    [{'position': (0, h), 'angle': 90.0, 'speed': 100, 'size': 3} for h in heights] +
                    [{'position': (width, h), 'angle': -90.0, 'speed': 100, 'size': 3} for h in heights],
    ship_states=[{'position': (width*0.35, height*0.75), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 1},
                 {'position': (width*0.65, height*0.25), 'angle': 180, 'lives': 3, 'team': 2, "mines_remaining": 1}],
    time_limit=60,
    map_size=(width, height),
)













widths = np.linspace(0, width, 10)
heights = np.linspace(0, height, 10)

wrapping_nightmare_fast = Scenario(
    name='wrapping_nightmare_fast',
    asteroid_states=[{'position': (w, 0), 'angle': 180.0, 'speed': 800, 'size': 2} for w in widths] + 
                    [{'position': (w, height), 'angle': 0.0, 'speed': 800, 'size': 2} for w in widths] +
                    [{'position': (0, h), 'angle': 90.0, 'speed': 800, 'size': 2} for h in heights] +
                    [{'position': (width, h), 'angle': -90.0, 'speed': 800, 'size': 2} for h in heights] + 
                    [{'position': (w, 0), 'angle': 180.0, 'speed': -800, 'size': 2} for w in widths] + 
                    [{'position': (w, height), 'angle': 0.0, 'speed': -800, 'size': 2} for w in widths] +
                    [{'position': (0, h), 'angle': 90.0, 'speed': -800, 'size': 2} for h in heights] +
                    [{'position': (width, h), 'angle': -90.0, 'speed': -800, 'size': 2} for h in heights],
    ship_states=[{'position': (width*0.35, height*0.75), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 1},
                 {'position': (width*0.65, height*0.25), 'angle': 180, 'lives': 3, 'team': 2, "mines_remaining": 1}
                 ],
    time_limit=60,
    map_size=(width, height),
)














purgatory = Scenario(
    name='purgatory',
    asteroid_states=[{'position': (0, 0), 'angle': 0.0, 'speed': 0, 'size': 1}],
    ship_states=[{'position': (0, 0), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 1}],
    time_limit=60,
    map_size=(width, height),
)
















cross = Scenario(name='Cross',
    asteroid_states=[{'position': (width/2, height/2), 'speed': 100, 'angle': np.degrees(np.arctan2(height, width)), 'size': 4},
                     {'position': (width/2, height/2), 'speed': 100, 'angle': -np.degrees(np.arctan2(height, width)), 'size': 4},
                     {'position': (width/2, height/2), 'speed': 100, 'angle': 180 - np.degrees(np.arctan2(height, width)), 'size': 4},
                     {'position': (width/2, height/2), 'speed': 100, 'angle': 180 + np.degrees(np.arctan2(height, width)), 'size': 4},
                     {'position': (0, 0), 'speed': 100, 'angle': np.degrees(np.arctan2(height, width)), 'size': 4},
                     {'position': (0, height), 'speed': 100, 'angle': -np.degrees(np.arctan2(height, width)), 'size': 4},
                     {'position': (width, 0), 'speed': 100, 'angle': 180 - np.degrees(np.arctan2(height, width)), 'size': 4},
                     {'position': (width, height), 'speed': 100, 'angle': 180 + np.degrees(np.arctan2(height, width)), 'size': 4}],
    ship_states=[
        {'position': (width//3, height//2), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 2},
        {'position': (width*2//3, height//2), 'angle': 180, 'lives': 3, 'team': 2, "mines_remaining": 2},
    ],
    map_size=(width, height),
    time_limit=np.inf,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False
)







fight_for_asteroid = Scenario(name='Fight For Asteroid',
    asteroid_states=[{'position': (width/2, height/2), 'speed': 0, 'angle': np.degrees(np.arctan2(height, width)), 'size': 1}],
    ship_states=[
        {'position': (1, height//2), 'angle': 20, 'lives': 3, 'team': 1, "mines_remaining": 2},
        {'position': (width - 1, height//2), 'angle': 180, 'lives': 3, 'team': 2, "mines_remaining": 2},
    ],
    map_size=(width, height),
    time_limit=np.inf,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False
)
















shot_pred_test = Scenario(name='shot_pred_test',
    asteroid_states=[{'position': (width, height*6/7), 'speed': 1000, 'angle': 180, 'size': 2}],
    ship_states=[
        {'position': (width/4, 1), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 0},
    ],
    map_size=(width, height),
    time_limit=np.inf,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False
)



















num_asteroids = 6
side_padding = 60
asteroid_spacing = (height - 2*side_padding)/num_asteroids  # Spacing between asteroids
asteroid_speed = 350  # Constant speed of the asteroids
asteroid_start_x = width*4/5  # Starting height at the top of the screen

# Create asteroid states
asteroid_states = []
for i in range(num_asteroids):
    y_position = side_padding + i * asteroid_spacing + asteroid_spacing/2  # Centering each asteroid
    asteroid_states.append({
        'position': (asteroid_start_x, y_position),
        'angle': 180 + (30 if i%2 == 0 else -30),
        'speed': asteroid_speed,
        'size': 4,
    })

# Create the scenario
shredder = Scenario(
    name='shredder',
    asteroid_states=asteroid_states,
    ship_states=[{'position': (width/5, height*3/4), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 1},
                 {'position': (width/5, height/4), 'angle': 90, 'lives': 3, 'team': 2, 'mines_remaining': 1}],
    map_size=(width, height),
    time_limit=45,
)








num_asteroids = 12
asteroid_y_spacing = height/num_asteroids  # Spacing between asteroids
asteroid_x_spacing = width/num_asteroids
asteroid_speed_x = (height+width)*200/height  # Constant speed of the asteroids
asteroid_speed_y = (height+width)*200/width
asteroid_start_x = width*4/5  # Starting height at the top of the screen

# Create asteroid states
asteroid_states = []
for i in range(num_asteroids):
    y_position = i * asteroid_y_spacing + asteroid_y_spacing/2  # Centering each asteroid
    x_position = i * asteroid_x_spacing + asteroid_x_spacing/2
    asteroid_states.append({
        'position': (x_position, y_position),
        'angle': 0 if i%2 == 0 else 270,
        'speed': asteroid_speed_x if i%2 == 0 else asteroid_speed_y,
        'size': 3,
    })

# Create the scenario
diagonal_shredder = Scenario(
    name='diagonal_shredder',
    asteroid_states=asteroid_states,
    ship_states=[{'position': (width/5, height*3/4), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 1},
                 {'position': (width/5, height/4), 'angle': 90, 'lives': 3, 'team': 2, 'mines_remaining': 1}],
    map_size=(width, height),
    time_limit=45,
)













out_of_bound_mine = Scenario(
    name='out_of_bound_mine',
    asteroid_states=[{'position': (width/100, height*42/100), 'speed': 0, 'angle': 0, 'size': 1}],
    ship_states=[{'position': (width/2, height/2), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 2}],
    map_size=(width, height),
    time_limit=60,
)








explainability_1 = Scenario(
    name='explainability_1',
    asteroid_states=[{'position': (width/100, height*99/100), 'speed': 350, 'angle': -math.degrees(math.atan(height/width)), 'size': 1},
                     {'position': (width*5/100, height*5/100), 'speed': 0, 'angle': 0, 'size': 4}],
    ship_states=[{'position': (width/2, height/2), 'angle': 180+90+0*math.degrees(math.atan(height/width)), 'lives': 3, 'team': 1, 'mines_remaining': 3}],
    map_size=(width, height),
    time_limit=15,
)

explainability_2 = Scenario(
    name='explainability_2',
    asteroid_states=[{'position': (width/100, height*99/100), 'speed': 350, 'angle': -math.degrees(math.atan(height/width)), 'size': 1},
                     {'position': (width*5/100, height*5/100), 'speed': 0, 'angle': 0, 'size': 4}],
    ship_states=[{'position': (width/2, height/2), 'angle': 180+90+0*math.degrees(math.atan(height/width)), 'lives': 3, 'team': 1, 'mines_remaining': 3}],
    map_size=(width, height),
    time_limit=15,
)






split_forecasting = Scenario(
    name='split_forecasting',
    asteroid_states=[{'position': (width*0.9, height*0.7), 'speed': 300, 'angle': 180, 'size': 2}],
    ship_states=[{'position': (width*0.5, height*0.15), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 3}],
    map_size=(width, height),
    time_limit=15,
)














# Maze parameters
maze_rows, maze_columns = 8, 12
asteroid_size = 3
spacing_x = width // (maze_columns + 1)
spacing_y = height // (maze_rows + 1)

# Generate maze asteroid positions
asteroid_states = []
for row in range(1, maze_rows + 1):
    for col in range(1, maze_columns + 1):
        # Skip positions to create paths
        if (row + col) % 3 != 0:  # Simple condition to create paths, adjust as needed
            x_position = col * spacing_x
            y_position = row * spacing_y
            asteroid_states.append({
                'position': (x_position, y_position),
                'angle': 0,
                'speed': random.uniform(0, 40),
                'size': asteroid_size,
            })

# Player ship state
ship_state = [{
    'position': (width // 2, height - 100),  # Start position near the bottom center
    'angle': 0,
    'lives': 3,
    'team': 1,
    'mines_remaining': 5,  # Limited number of mines to clear the path
}]

# Creating the Minefield Maze scenario
minefield_maze_scenario = Scenario(
    name='minefield_maze_scenario',
    asteroid_states=asteroid_states,
    ship_states=ship_state,
    map_size=(width, height),
    time_limit=3600,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False,
)





wrap_collision_test = Scenario(name='Wrap Collision Test',
                            asteroid_states=[{'position': (50, 800-50), 'speed': 100, 'angle': 90+45+0.000001, 'size': 4}, {'position': (1000-50, 50), 'speed': 100, 'angle': 180+90+45-0.01, 'size': 4}],
                            ship_states=[
                                {'position': (0.01, 0.01), 'angle': 80, 'lives': 3, 'team': 1, "mines_remaining": 1},
                            ],
                            map_size=(width, height),
                            time_limit=500,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)


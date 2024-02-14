from math import floor, sin, cos
from random import randint, uniform
import random
import time
#from scalene import profile

MINE_BLAST_RADIUS = 150

random.seed(1)

class AsteroidSpatialHash:
    def __init__(self, cell_size: int = 100):
        self.cell_size = cell_size
        self.d = {}

    def show_d(self):
        """Show the current state of the spatial hash dictionary."""
        return self.d

    def _cell_coords_for_object(self, x1, y1, x2, y2):
        """Calculate the cell coordinates for an object based on its bounding box."""
        cy_start = floor(y1) // self.cell_size
        cy_end = floor(y2) // self.cell_size
        cx_start = floor(x1) // self.cell_size
        cx_end = floor(x2) // self.cell_size
        return [(cx, cy) for cy in range(cy_start, cy_end + 1) for cx in range(cx_start, cx_end + 1)]

    def add_asteroid(self, asteroid_index, asteroid):
        """Add an asteroid by index."""
        position, radius = asteroid['position'], asteroid['radius']
        x1, y1, x2, y2 = position[0] - radius, position[1] - radius, position[0] + radius, position[1] + radius
        cells = self._cell_coords_for_object(x1, y1, x2, y2)
        for c in cells:
            self.d.setdefault(c, set()).add(asteroid_index)

    def remove_asteroid(self, asteroid_index, asteroid):
        """Remove an asteroid by index."""
        position, radius = asteroid['position'], asteroid['radius']
        x1, y1, x2, y2 = position[0] - radius, position[1] - radius, position[0] + radius, position[1] + radius
        cells = self._cell_coords_for_object(x1, y1, x2, y2)
        for c in cells:
            if c in self.d:
                self.d[c].discard(asteroid_index)

    def _potential_collisions_for_object(self, x1, y1, x2, y2):
        cells = self._cell_coords_for_object(x1, y1, x2, y2)
        potentials = set()
        for c in cells:
            if c in self.d:
                potentials.update(self.d[c])
        return potentials

    def potential_collisions_for_circle(self, circle):
        """Get a set of all asteroid indices that potentially intersect the given object."""
        position, radius = circle['position'], circle['radius']
        x1, y1, x2, y2 = position[0] - radius, position[1] - radius, position[0] + radius, position[1] + radius
        return self._potential_collisions_for_object(x1, y1, x2, y2)

    def potential_collisions_for_bullet(self, bullet):
        """Get a set of all asteroid indices that potentially intersect the given object."""
        position, tail_delta = bullet['position'], bullet['tail_delta']
        x1, y1 = position[0] + (tail_delta[0] if tail_delta[0] < 0 else 0), position[1] + (tail_delta[1] if tail_delta[1] < 0 else 0)
        x2, y2 = position[0] + (tail_delta[0] if tail_delta[0] > 0 else 0), position[1] + (tail_delta[1] if tail_delta[1] > 0 else 0)
        return self._potential_collisions_for_object(x1, y1, x2, y2)




class SpatialHashFast:
    def __init__(self, cell_size=10.0):
        self.cell_size = cell_size
        self._cells = {}
        self._current_query_id = 0
        self._query_ids = {}

    def _get_cell_index(self, position):
        x_index = int(position[0] // self.cell_size)
        y_index = int(position[1] // self.cell_size)
        return x_index, y_index

    def _cell_coords_for_object(self, x1, y1, x2, y2):
        cells = []
        cx_start, cy_start = self._get_cell_index((x1, y1))
        cx_end, cy_end = self._get_cell_index((x2, y2))
        for cx in range(cx_start, cx_end + 1):
            for cy in range(cy_start, cy_end + 1):
                cells.append((cx, cy))
        return cells

    def _bbox_for_circle(self, position, radius):
        return position[0] - radius, position[1] - radius, position[0] + radius, position[1] + radius

    def _bbox_for_bullet(self, position, tail_delta):
        x1, y1 = position
        x2, y2 = position[0] + tail_delta[0], position[1] + tail_delta[1]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def add_asteroid(self, asteroid_index, asteroid):
        x1, y1, x2, y2 = self._bbox_for_circle(asteroid['position'], asteroid['radius'])
        cells = self._cell_coords_for_object(x1, y1, x2, y2)
        for c in cells:
            if c not in self._cells:
                self._cells[c] = set()
            self._cells[c].add(asteroid_index)

    def potential_collisions_for_circle(self, circle):
        x1, y1, x2, y2 = self._bbox_for_circle(circle['position'], circle['radius'])
        return self._potential_collisions_for_object(x1, y1, x2, y2)

    def potential_collisions_for_bullet(self, bullet):
        x1, y1, x2, y2 = self._bbox_for_bullet(bullet['position'], bullet['tail_delta'])
        return self._potential_collisions_for_object(x1, y1, x2, y2)

    def _potential_collisions_for_object(self, x1, y1, x2, y2):
        self._current_query_id += 1
        potentials = set()
        cells = self._cell_coords_for_object(x1, y1, x2, y2)
        for c in cells:
            if c in self._cells:
                for asteroid_index in self._cells[c]:
                    if self._query_ids.get(asteroid_index, -1) != self._current_query_id:
                        potentials.add(asteroid_index)
                        self._query_ids[asteroid_index] = self._current_query_id
        return potentials




# Example usage remains the same


ast = {
    "position": (580.8532337851002, 249.24164155498283),
    "velocity": (42.9834991427168, -179.24032258197764),
    "size": 3,
    "mass": 452.3893421169302,
    "radius": 24.0,
}

ship = {
    "position": (580.8532337851002, 249.24164155498283),
    "velocity": (42.9834991427168, -179.24032258197764),
    "size": 3,
    "mass": 452.3893421169302,
    "radius": 24.0,
}

# Benchmark Setup
num_asteroids = 6000000
num_bullets = 10
asteroids = []
bullets = []
width = 1000
height = 800

# Generate random asteroids
for _ in range(num_asteroids):
    asteroid = {
        "position": (uniform(0, width), uniform(0, height)),
        "radius": [8.0, 16.0, 24.0, 32.0][random.randint(0, 3)]
    }
    asteroids.append(asteroid)

# Generate random bullets
for _ in range(num_bullets):
    bullet_position = (uniform(0, width), uniform(0, height))
    angle = uniform(0, 360)
    dx = 12 * cos(angle)
    dy = 12 * sin(angle)
    bullet = {
        "position": bullet_position,
        "tail_delta": (dx, dy)
    }
    bullets.append(bullet)

# Initialize spatial hash
ash = AsteroidSpatialHash(cell_size=100)

# Add asteroids to spatial hash and benchmark
start_time = time.time()
for i, asteroid in enumerate(asteroids):
    ash.add_asteroid(i, asteroid)
add_asteroids_time = time.time() - start_time

# Benchmark potential collision checks for bullets
start_time = time.time()
for bullet in bullets:
    _ = ash.potential_collisions_for_bullet(bullet)
check_collisions_time = time.time() - start_time

print(f"add_asteroids_time: {add_asteroids_time}, check_collisions_time: {check_collisions_time}")

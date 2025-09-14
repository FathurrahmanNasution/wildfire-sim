# world.py
import random
import numpy as np
from config import GRID_SIZE, TILE_WORLD_SIZE, BURNING_DURATION, FIRE_SPREAD_BASE
from states import TreeState
import math

TYPE_RESISTANCE = {
    'city': 35,
    'river': 100,
    'forest': 20,
    'dry_forest': 5,
    'brush': 20,
    'dry_brush': 0,
    'grass': 20,
    'dry_grass': 0,
    'stone': 100,
    'swamp': 60
}

# Height ranges for different terrain types
HEIGHT_RANGES = {
    'city': (6.0, 20.0),
    'river': (0.1, 0.3),
    'forest': (8.0, 14.0),
    'dry_forest': (6.0, 12.0),
    'brush': (2.0, 4.0),
    'dry_brush': (1.5, 3.5),
    'grass': (0.5, 1.5),
    'dry_grass': (0.3, 1.0),
    'stone': (3.0, 8.0),
    'swamp': (1.0, 3.0)
}

class WildfireWorld:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.cells = {}  # key: (gx, gz) -> dict {type, state, height, burning_time, position}
        self.wind_dir = np.array([1.0, 0.0], dtype=np.float32)
        self.wind_strength = 0.35
        self.init_world()

    def get_height_for_type(self, terrain_type):
        """Get appropriate height for terrain type"""
        height_range = HEIGHT_RANGES.get(terrain_type, (1.0, 3.0))
        return np.random.uniform(height_range[0], height_range[1])

    def init_world(self):
        self.cells.clear()
        half = self.grid_size // 2
        
        # Define terrain distribution patterns
        terrain_weights = {
            'forest': 30,
            'dry_forest': 15,
            'grass': 20,
            'dry_grass': 10,
            'brush': 10,
            'dry_brush': 5,
            'city': 3,
            'river': 2,
            'stone': 3,
            'swamp': 2
        }
        
        # Create weighted list for random selection
        terrain_list = []
        for terrain, weight in terrain_weights.items():
            terrain_list.extend([terrain] * weight)
        
        for gx in range(-half, half):
            for gz in range(-half, half):
                # Add some clustering for more realistic terrain
                distance_from_center = math.sqrt(gx*gx + gz*gz)
                
                # Cities more likely near center
                if distance_from_center < 3 and random.random() < 0.3:
                    ctype = "city"
                # Rivers in lines (simplified)
                elif abs(gx) == 2 and random.random() < 0.6:
                    ctype = "river"
                # Stone in clusters
                elif distance_from_center > 8 and random.random() < 0.2:
                    ctype = "stone"
                else:
                    ctype = random.choice(terrain_list)
                
                height = self.get_height_for_type(ctype)
                pos_world = np.array([gx * TILE_WORLD_SIZE, 0.0, gz * TILE_WORLD_SIZE], dtype=np.float32)
                
                # Determine initial state based on resistance
                initial_state = TreeState.HEALTHY
                if TYPE_RESISTANCE[ctype] >= 100:  # Non-flammable
                    initial_state = TreeState.BURNT  # Use BURNT to indicate non-flammable
                
                self.cells[(gx, gz)] = {
                    'type': ctype,
                    'state': initial_state,
                    'height': height,
                    'burning_time': 0,
                    'position': pos_world
                }

    def reset(self):
        self.init_world()

    def random_ignite(self, count=1):
        flammable = [pos for pos, c in self.cells.items() 
                     if c['state'] == TreeState.HEALTHY and TYPE_RESISTANCE[c['type']] < 100]
        if not flammable:
            return
        for _ in range(count):
            if not flammable:
                break
            pos = random.choice(flammable)
            self.cells[pos]['state'] = TreeState.BURNING
            self.cells[pos]['burning_time'] = 0
            flammable.remove(pos)

    def spread_fire(self):
        new_fires = []
        for (gx, gz), c in list(self.cells.items()):
            if c['state'] == TreeState.BURNING:
                c['burning_time'] += 1
                resistance = TYPE_RESISTANCE.get(c['type'], 20)
                duration = max(1, BURNING_DURATION + resistance // 5)

                # Finished burning
                if c['burning_time'] > duration:
                    c['state'] = TreeState.BURNT
                    c['burning_time'] = 0
                else:
                    # Spread fire only during early burning phase
                    if c['burning_time'] < (duration // 3):
                        for dx in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                if dx == 0 and dz == 0:
                                    continue
                                npos = (gx + dx, gz + dz)
                                if npos in self.cells:
                                    neighbor = self.cells[npos]
                                    if neighbor['state'] == TreeState.HEALTHY:
                                        nres = TYPE_RESISTANCE.get(neighbor['type'], 20)
                                        if nres >= 100:
                                            continue  # Cannot burn
                                        
                                        distance = math.sqrt(dx*dx + dz*dz)
                                        wind_effect = max(0.0, np.dot(self.wind_dir, np.array([dx, dz], dtype=np.float32))) * self.wind_strength
                                        
                                        # Different spread rates for different terrain
                                        base_spread = FIRE_SPREAD_BASE
                                        if neighbor['type'].startswith('dry_'):
                                            base_spread *= 2.0  # Dry terrain burns faster
                                        elif neighbor['type'] in ['city', 'stone']:
                                            base_spread *= 0.5  # Urban/rocky areas spread slower
                                        elif neighbor['type'] == 'swamp':
                                            base_spread *= 0.3  # Wet areas spread much slower
                                        
                                        prob = (base_spread * (1.0 + wind_effect) / distance) * (1.0 - nres/100.0)
                                        if random.random() < prob:
                                            new_fires.append(npos)
        
        for p in new_fires:
            self.cells[p]['state'] = TreeState.BURNING
            self.cells[p]['burning_time'] = 0

    def save_map(self, filename):
        """Save current terrain layout to file"""
        try:
            with open(filename, 'w') as f:
                half = self.grid_size // 2
                for gz in range(-half, half):
                    row = []
                    for gx in range(-half, half):
                        if (gx, gz) in self.cells:
                            row.append(self.cells[(gx, gz)]['type'])
                        else:
                            row.append('grass')
                    f.write(' '.join(row) + '\n')
            print(f"Map saved to {filename}")
        except Exception as e:
            print(f"Error saving map: {e}")

    def load_map(self, filename):
        """Load terrain layout from file"""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            self.cells.clear()
            half = len(lines) // 2
            
            for gz_idx, line in enumerate(lines):
                terrain_types = line.strip().split()
                gz = gz_idx - half
                for gx_idx, terrain_type in enumerate(terrain_types):
                    gx = gx_idx - half
                    height = self.get_height_for_type(terrain_type)
                    pos_world = np.array([gx * TILE_WORLD_SIZE, 0.0, gz * TILE_WORLD_SIZE], dtype=np.float32)
                    
                    initial_state = TreeState.HEALTHY
                    if TYPE_RESISTANCE.get(terrain_type, 20) >= 100:
                        initial_state = TreeState.BURNT
                    
                    self.cells[(gx, gz)] = {
                        'type': terrain_type,
                        'state': initial_state,
                        'height': height,
                        'burning_time': 0,
                        'position': pos_world
                    }
            print(f"Map loaded from {filename}")
        except Exception as e:
            print(f"Error loading map: {e}")
            self.init_world()  # Fallback to default generation
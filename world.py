# world.py
import random
import numpy as np
import math
from config import GRID_SIZE, TILE_WORLD_SIZE, TYPE_RESISTANCE, HEIGHT_RANGES, FIRE_SPREAD_BASE, BURNING_DURATION, WIND_DIR, WIND_STRENGTH
from states import TreeState

class WildfireWorld:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.half = grid_size // 2
        self.cells = {}
        self.wind_dir = np.array(WIND_DIR, dtype=float)
        self.wind_strength = WIND_STRENGTH
        self.init_world()

    def get_height_for_type(self, ttype):
        r = HEIGHT_RANGES.get(ttype, (1.0, 3.0))
        return random.uniform(r[0], r[1])

    def init_world(self):
        self.cells.clear()
        terrain_weights = {
            'forest': 30, 'dry_forest': 10, 'grass': 20, 'dry_grass': 8,
            'brush': 10, 'dry_brush': 5, 'city': 3, 'river': 2, 'stone': 3, 'swamp': 4
        }
        terrain_list = []
        for t, w in terrain_weights.items():
            terrain_list.extend([t] * w)
        for gx in range(-self.half, self.half):
            for gz in range(-self.half, self.half):
                d = math.hypot(gx, gz)
                if d < 3 and random.random() < 0.25:
                    ttype = 'city'
                elif abs(gx) == 2 and random.random() < 0.5:
                    ttype = 'river'
                elif d > 8 and random.random() < 0.15:
                    ttype = 'stone'
                else:
                    ttype = random.choice(terrain_list)
                height = self.get_height_for_type(ttype)
                pos = np.array([gx * TILE_WORLD_SIZE, 0.0, gz * TILE_WORLD_SIZE], dtype=float)
                state = TreeState.HEALTHY
                if TYPE_RESISTANCE.get(ttype, 20) >= 100:
                    state = TreeState.BURNT
                self.cells[(gx, gz)] = {
                    'type': ttype,
                    'state': state,
                    'height': height,
                    'burning_time': 0,
                    'position': pos,
                    # morph dictionary holds prev/target/progress
                    'morph': None
                }

    def reset(self):
        self.init_world()

    def start_morph(self, cell_key, new_type):
        """Begin a morph from current type -> new_type on a cell."""
        if cell_key not in self.cells:
            return
        cell = self.cells[cell_key]
        if cell['type'] == new_type:
            return
        cell['morph'] = {
            'prev_type': cell['type'],
            'target_type': new_type,
            'progress': 0.0,
            'duration': 0.8  # seconds to complete morph
        }
        # keep state healthy while morphing
        cell['state'] = TreeState.HEALTHY
        cell['burning_time'] = 0

    def random_ignite(self, count=1):
        flammable = [pos for pos,c in self.cells.items() if c['state']==TreeState.HEALTHY and TYPE_RESISTANCE.get(c['type'],20)<100]
        if not flammable:
            return
        for _ in range(count):
            p = random.choice(flammable)
            self.cells[p]['state'] = TreeState.BURNING
            self.cells[p]['burning_time'] = 0
            flammable.remove(p)

    def update(self, dt):
        """Advance burning timers, spread fires, and morph transitions. dt in seconds."""
        newfires = []
        for (gx,gz), c in list(self.cells.items()):
            # update morph
            m = c.get('morph')
            if m:
                m['progress'] += dt / max(1e-9, m['duration'])
                if m['progress'] >= 1.0:
                    # finalize
                    c['type'] = m['target_type']
                    c['height'] = self.get_height_for_type(c['type'])
                    c['morph'] = None

            # burning behavior
            if c['state'] == TreeState.BURNING:
                c['burning_time'] += dt
                res = TYPE_RESISTANCE.get(c['type'], 20)
                duration = max(0.5, (BURNING_DURATION + res // 5) / 30.0)  # seconds
                if c['burning_time'] > duration:
                    c['state'] = TreeState.BURNT
                    c['burning_time'] = 0
                else:
                    # spread only occasionally
                    # use ticks per second ~ 4 spread attempts per second
                    # compute integer ticks from time
                    if int(c['burning_time'] * 4) != int((c['burning_time'] - dt) * 4):
                        for dx in (-1,0,1):
                            for dz in (-1,0,1):
                                if dx == 0 and dz == 0:
                                    continue
                                npos = (gx+dx, gz+dz)
                                if npos not in self.cells:
                                    continue
                                neighbor = self.cells[npos]
                                if neighbor['state'] != TreeState.HEALTHY:
                                    continue
                                nres = TYPE_RESISTANCE.get(neighbor['type'], 20)
                                if nres >= 100:
                                    continue
                                distance = math.hypot(dx, dz)
                                wind_effect = max(0.0, np.dot(self.wind_dir, np.array([dx, dz], dtype=float))) * self.wind_strength
                                base = FIRE_SPREAD_BASE * 0.35  # reduce base for slower spread
                                if neighbor['type'].startswith('dry_'):
                                    base *= 1.6
                                elif neighbor['type'] in ['city', 'stone']:
                                    base *= 0.5
                                elif neighbor['type'] == 'swamp':
                                    base *= 0.25
                                prob = (base * (1.0 + wind_effect) / max(0.001, distance)) * (1.0 - nres/100.0)
                                if random.random() < prob:
                                    newfires.append(npos)
        for p in newfires:
            if self.cells[p]['state'] == TreeState.HEALTHY:
                self.cells[p]['state'] = TreeState.BURNING
                self.cells[p]['burning_time'] = 0

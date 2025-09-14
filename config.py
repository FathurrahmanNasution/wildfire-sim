# config.py
SCREEN_W = 1200
SCREEN_H = 800
FPS = 60

# World / grid
GRID_SIZE = 20
TILE_WORLD_SIZE = 5.0

# Fire
BURNING_DURATION = 60        # base frames for burning (will be adjusted by resistance)
FIRE_SPREAD_BASE = 0.20
WIND_DIR = (1.0, 0.0)
WIND_STRENGTH = 0.35

# Particles
MAX_PARTICLES = 800

# Colors (normalized floats 0..1) for terrain types (distinct)
TYPE_COLOR = {
    "grass":      (0.30, 0.90, 0.30),
    "dry_grass":  (0.90, 0.90, 0.30),
    "brush":      (0.10, 0.50, 0.10),
    "dry_brush":  (0.60, 0.30, 0.10),
    "forest":     (0.00, 0.40, 0.00),
    "dry_forest": (0.82, 0.41, 0.12),  # chocolate-like
    "city":       (0.50, 0.50, 0.50),
    "river":      (0.10, 0.30, 0.90),
    "stone":      (0.60, 0.60, 0.60),
    "swamp":      (0.00, 0.30, 0.20)
}

# Resistance (0..100). 100 = non-flammable
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

# Height ranges
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

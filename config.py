# config.py
SCREEN_W = 1200
SCREEN_H = 800
BACKGROUND_SKY = (135, 206, 235)
FPS = 30

GRID_SIZE = 20           # NxN world grid
TILE_WORLD_SIZE = 5.0    # spacing between trees (world units)
TREE_DENSITY = 0.75      # chance to place tree at a grid cell

# fire params (you said will tune later)
BURNING_DURATION = 15     # frames until tree becomes burnt
FIRE_SPREAD_BASE = 0.07   # base spread chance per neighbor per spread call

# Particle params
MAX_PARTICLES = 800

# Colors
COLOR_TRUNK = (101, 67, 33)
COLOR_ASH = (64, 64, 64)

# config.py
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

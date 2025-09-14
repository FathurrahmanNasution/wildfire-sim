# terrain.py
from config import HEIGHT_RANGES
import random

class TerrainTile:
    def __init__(self, ttype='grass'):
        self.type = ttype
        self.set_params(ttype)

    def set_params(self, ttype):
        self.type = ttype
        rng = HEIGHT_RANGES.get(ttype, (1.0,3.0))
        self.height = random.uniform(rng[0], rng[1])

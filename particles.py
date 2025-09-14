# particles.py
import random
import numpy as np
from config import MAX_PARTICLES

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def spawn_tree_particles(self, world_pos, height, count_range=(0,3)):
        for _ in range(random.randint(*count_range)):
            if len(self.particles) > MAX_PARTICLES:
                break
            p = {
                'pos': world_pos + np.array([random.uniform(-0.6,0.6), height*0.6 + random.uniform(0,1.6), random.uniform(-0.6,0.6)], dtype=np.float32),
                'vel': np.array([random.uniform(-0.2,0.2), random.uniform(0.6, 2.2), random.uniform(-0.2,0.2)], dtype=np.float32),
                'life': random.randint(30, 90),
                'size': random.uniform(1.0, 3.0),
                'type': 'ember' if random.random() < 0.6 else 'smoke'
            }
            self.particles.append(p)

    def update(self):
        new_particles = []
        for p in self.particles:
            p['pos'] = p['pos'] + p['vel'] * 0.2
            p['vel'][1] -= 0.06
            p['vel'] *= 0.995
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

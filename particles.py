# particles.py
import random
import numpy as np
from config import MAX_PARTICLES
from OpenGL.GL import (
    glDisable, glEnable, glBegin, glEnd, glColor4f,
    glVertex3f, glPointSize, GL_LIGHTING, GL_POINTS
)

class ParticleSystem:
    def __init__(self):
        self.particles = []  # {pos, vel, life, color}

    def spawn_tree_particles(self, world_pos, height, count=2):
        """Spawn embers and smoke near top of object."""
        for _ in range(count):
            if len(self.particles) >= MAX_PARTICLES:
                break
            top = np.array(world_pos, dtype=float) + np.array([0.0, height * 0.75, 0.0])
            p = {
                "pos": top + np.array([random.uniform(-0.4,0.4), random.uniform(0.0,0.4), random.uniform(-0.4,0.4)]),
                "vel": np.array([random.uniform(-0.06,0.06), random.uniform(0.4,1.0), random.uniform(-0.06,0.06)]),
                "life": random.uniform(0.8, 1.6),
                "color": (1.0, random.uniform(0.3,0.8), random.uniform(0.0,0.2)) if random.random() < 0.7 else (0.5,0.5,0.5)
            }
            self.particles.append(p)

    def update(self, dt):
        new = []
        for p in self.particles:
            p["pos"] = p["pos"] + p["vel"] * dt
            # slow gravity effect so embers rise then fade
            p["vel"][1] -= 0.4 * dt
            p["vel"] *= 0.995
            p["life"] -= dt
            if p["life"] > 0:
                new.append(p)
        self.particles = new

    def render_gl(self):
        if not self.particles:
            return
        glDisable(GL_LIGHTING)
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for p in self.particles:
            life = max(0.0, min(1.0, p["life"]))
            glColor4f(p["color"][0], p["color"][1], p["color"][2], life)
            glVertex3f(float(p["pos"][0]), float(p["pos"][1]), float(p["pos"][2]))
        glEnd()
        glEnable(GL_LIGHTING)

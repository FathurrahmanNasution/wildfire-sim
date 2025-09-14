# main.py
import sys, math, pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from config import SCREEN_W, SCREEN_H, FPS, TILE_WORLD_SIZE, GRID_SIZE
from world import WildfireWorld
from particles import ParticleSystem
from renderer_opengl import GLRenderer
from states import TreeState

TERRAIN_TYPES = [
    "forest", "dry_forest",
    "grass", "dry_grass",
    "brush", "dry_brush",
    "city", "river", "stone", "swamp"
]

def init_pygame():
    pygame.init()
    pygame.display.set_mode((SCREEN_W, SCREEN_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Wildfire Forest Simulator (OpenGL)")

def draw_ui_opengl(font, paused, current_terrain, stats):
    """Render a small UI panel using pygame font to an OpenGL raster."""
    lines = [
        "CONTROLS:",
        "WASD: Move camera",
        "Arrow keys: Rotate camera",
        "Q/E: Move up/down",
        "F: Random Fire",
        "Left click: Ignite + Highlight",
        "Right click: Cycle terrain (morph)",
        f"Place type: {current_terrain}",
        f"Status: {'PAUSED' if paused else 'RUNNING'}",
        f"Healthy:{stats[0]} Burning:{stats[1]} Burnt:{stats[2]}"
    ]
    surf_w = 320
    surf_h = 20 * len(lines) + 10
    surf = pygame.Surface((surf_w, surf_h), flags=SRCALPHA)
    surf.fill((0, 0, 0, 150))
    for i, line in enumerate(lines):
        col = (255, 255, 0) if i == 0 else (255, 255, 255)
        txt = font.render(line, True, col)
        surf.blit(txt, (6, 6 + i * 20))
    data = pygame.image.tostring(surf, "RGBA", True)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity()
    glOrtho(0, SCREEN_W, 0, SCREEN_H, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glRasterPos2i(10, SCREEN_H - surf_h - 10)
    glDrawPixels(surf_w, surf_h, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def main():
    init_pygame()
    world = WildfireWorld(grid_size=GRID_SIZE)
    particles = ParticleSystem()
    renderer = GLRenderer(SCREEN_W, SCREEN_H)

    cam_pos = [0.0, 28.0, 80.0]
    cam_yaw = 0.0
    cam_pitch = -20.0
    move_speed = 3.0
    rot_speed = 2.0

    font = pygame.font.Font(None, 20)
    current_terrain_index = 0

    clock = pygame.time.Clock()
    running = True
    paused = False
    
    # barrier box = world boundary (in grid coordinates)
    half = GRID_SIZE // 2
    barriers = [
        ((-half, -half), (half, -half)),
        ((half, -half), (half, half)),
        ((half, half), (-half, half)),
        ((-half, half), (-half, -half))
    ]

    while running:
        dt = clock.tick(FPS) / 1000.0

        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    running = False
                elif ev.key == K_SPACE:
                    paused = not paused
                elif ev.key == K_r:
                    world.reset()
                    particles.particles.clear()
                elif ev.key == K_f:
                    world.random_ignite(3)
            elif ev.type == MOUSEBUTTONDOWN:
                # ensure camera matrix loaded for accurate picking
                tx = cam_pos[0] + math.sin(math.radians(cam_yaw))
                ty = cam_pos[1] + math.tan(math.radians(cam_pitch))
                tz = cam_pos[2] - math.cos(math.radians(cam_yaw))
                renderer.set_camera(cam_pos, (tx, ty, tz))
                sel = renderer.pick_cell(ev.pos[0], ev.pos[1], world)
                if sel and sel in world.cells:
                    cell = world.cells[sel]
                    if ev.button == 1:  # left click = ignite + highlight
                        if cell['state'] == TreeState.HEALTHY:
                            cell['state'] = TreeState.BURNING
                            cell['burning_time'] = 0
                    elif ev.button == 3:  # right click = morph
                        current_terrain_index = (current_terrain_index + 1) % len(TERRAIN_TYPES)
                        new_type = TERRAIN_TYPES[current_terrain_index]
                        world.start_morph(sel, new_type)

        # movement input
        keys = pygame.key.get_pressed()
        forward = [math.sin(math.radians(cam_yaw)), 0, -math.cos(math.radians(cam_yaw))]
        right = [math.cos(math.radians(cam_yaw)), 0, math.sin(math.radians(cam_yaw))]

        if keys[K_w]:
            cam_pos[0] += forward[0] * move_speed * dt * 30.0
            cam_pos[2] += forward[2] * move_speed * dt * 30.0
        if keys[K_s]:
            cam_pos[0] -= forward[0] * move_speed * dt * 30.0
            cam_pos[2] -= forward[2] * move_speed * dt * 30.0
        if keys[K_a]:
            cam_pos[0] -= right[0] * move_speed * dt * 30.0
            cam_pos[2] -= right[2] * move_speed * dt * 30.0
        if keys[K_d]:
            cam_pos[0] += right[0] * move_speed * dt * 30.0
            cam_pos[2] += right[2] * move_speed * dt * 30.0
        if keys[K_q]:
            cam_pos[1] += move_speed * dt * 30.0
        if keys[K_e]:
            cam_pos[1] -= move_speed * dt * 30.0

        if keys[K_LEFT]:
            cam_yaw -= rot_speed
        if keys[K_RIGHT]:
            cam_yaw += rot_speed
        if keys[K_UP]:
            cam_pitch = min(80.0, cam_pitch + rot_speed)
        if keys[K_DOWN]:
            cam_pitch = max(-80.0, cam_pitch - rot_speed)

        # clamp camera to barrier box
        max_bound = half * TILE_WORLD_SIZE
        cam_pos[0] = max(-max_bound, min(max_bound, cam_pos[0]))
        cam_pos[2] = max(-max_bound, min(max_bound, cam_pos[2]))

        # simulation update
        if not paused:
            world.update(dt)
            for c in world.cells.values():
                if c['state'] == TreeState.BURNING:
                    particles.spawn_tree_particles(c['position'], c['height'], count=1)
            particles.update(dt)

        # camera target and render
        tx = cam_pos[0] + math.sin(math.radians(cam_yaw))
        ty = cam_pos[1] + math.tan(math.radians(cam_pitch))
        tz = cam_pos[2] - math.cos(math.radians(cam_yaw))
        target = (tx, ty, tz)

        renderer.clear()
        renderer.set_camera(cam_pos, target)

        renderer.draw_ground(world.grid_size, TILE_WORLD_SIZE)

        # draw barrier
        for (x0, z0), (x1, z1) in barriers:
            renderer.draw_barrier(x0, z0, x1, z1)

        # draw world objects
        for pos, cell in world.cells.items():
            renderer.draw_object(cell['position'], cell['height'], TILE_WORLD_SIZE,
                                 state=cell['state'], cell_type=cell['type'],
                                 particles=particles, morph_info=cell.get('morph', None))

        particles.render_gl()

        # UI overlay
        healthy = sum(1 for c in world.cells.values() if c['state'] == TreeState.HEALTHY)
        burning = sum(1 for c in world.cells.values() if c['state'] == TreeState.BURNING)
        burnt = sum(1 for c in world.cells.values() if c['state'] == TreeState.BURNT)
        draw_ui_opengl(font, paused, TERRAIN_TYPES[current_terrain_index], (healthy, burning, burnt))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

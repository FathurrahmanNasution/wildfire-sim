# main.py
import pygame
import numpy as np
import math
from config import SCREEN_W, SCREEN_H, FPS
from world import WildfireWorld
from particles import ParticleSystem
from renderer import Renderer
from states import TreeState

def find_clicked_tree(renderer, world, mx, my, click_radius=30):
    best = None
    best_dist2 = float('inf')
    for pos, cell in world.cells.items():
        world_pos = cell['position'].copy()
        world_pos[1] = cell['height'] * 0.6
        sp = renderer.world_to_screen(world_pos)
        if sp is None:
            continue
        sx, sy, depth = sp
        dx = sx - mx
        dy = sy - my
        d2 = dx*dx + dy*dy
        if d2 < best_dist2:
            best_dist2 = d2
            best = pos
    if best is not None and best_dist2 <= click_radius * click_radius:
        return best
    return None

def draw_instructions(screen):
    """Draw instruction panel on the screen"""
    font = pygame.font.Font(None, 20)
    instructions = [
        "CONTROLS:",
        "WASD: Move camera",
        "QE: Move up/down", 
        "Arrows: Rotate camera",
        "Shift: Fast movement",
        "Ctrl: Slow movement",
        "+/-: Zoom in/out",
        "Left click: Ignite fire",
        "Right click: Cycle terrain",
        "F: Random fire",
        "R: Reset world",
        "SPACE: Pause/unpause",
        "C: Recenter camera",
        "K: Save map",
        "L: Load map",
        "H: Toggle help"
    ]
    
    # Create semi-transparent background
    panel_width = 220
    panel_height = len(instructions) * 22 + 10
    panel_surf = pygame.Surface((panel_width, panel_height), flags=pygame.SRCALPHA)
    panel_surf.fill((0, 0, 0, 180))
    
    # Draw instructions
    for i, line in enumerate(instructions):
        color = (255, 255, 0) if line == "CONTROLS:" else (255, 255, 255)
        text_surf = font.render(line, True, color)
        panel_surf.blit(text_surf, (5, 5 + i * 22))
    
    screen.blit(panel_surf, (10, 10))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("3D Wildfire Simulator (Enhanced)")

    world = WildfireWorld(grid_size=20)
    particles = ParticleSystem()
    renderer = Renderer(screen, world)

    clock = pygame.time.Clock()
    running = True
    paused = False
    show_help = True
    time_step = 0

    # Terrain types for cycling
    terrain_types = ['forest', 'dry_forest', 'grass', 'dry_grass', 'brush', 'dry_brush', 
                     'city', 'river', 'stone', 'swamp']
    current_terrain_index = 0

    while running:
        # Input handling
        keys = pygame.key.get_pressed()
        move_speed = 3.0
        rot_speed = 0.025
        
        # Camera movement
        if keys[pygame.K_w]:
            dx = math.sin(renderer.camera_angle_y) * move_speed
            dz = math.cos(renderer.camera_angle_y) * move_speed
            renderer.camera_pos[0] += dx
            renderer.camera_pos[2] += dz
        if keys[pygame.K_s]:
            dx = math.sin(renderer.camera_angle_y) * move_speed
            dz = math.cos(renderer.camera_angle_y) * move_speed
            renderer.camera_pos[0] -= dx
            renderer.camera_pos[2] -= dz
        if keys[pygame.K_a]:
            dx = math.sin(renderer.camera_angle_y - math.pi/2) * move_speed
            dz = math.cos(renderer.camera_angle_y - math.pi/2) * move_speed
            renderer.camera_pos[0] += dx
            renderer.camera_pos[2] += dz
        if keys[pygame.K_d]:
            dx = math.sin(renderer.camera_angle_y + math.pi/2) * move_speed
            dz = math.cos(renderer.camera_angle_y + math.pi/2) * move_speed
            renderer.camera_pos[0] += dx
            renderer.camera_pos[2] += dz
        if keys[pygame.K_q]:
            renderer.camera_pos[1] += move_speed
        if keys[pygame.K_e]:
            renderer.camera_pos[1] -= move_speed

        # Camera rotation
        if keys[pygame.K_LEFT]:
            renderer.camera_angle_y -= rot_speed
        if keys[pygame.K_RIGHT]:
            renderer.camera_angle_y += rot_speed
        if keys[pygame.K_UP]:
            renderer.camera_angle_x = max(-math.pi/3, renderer.camera_angle_x - rot_speed)
        if keys[pygame.K_DOWN]:
            renderer.camera_angle_x = min(math.pi/3, renderer.camera_angle_x + rot_speed)

        # Zoom
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            renderer.zoom = min(3.0, renderer.zoom * 1.01)
        if keys[pygame.K_MINUS]:
            renderer.zoom = max(0.35, renderer.zoom * 0.99)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    world.reset()
                    particles.particles.clear()
                    time_step = 0
                elif event.key == pygame.K_f:
                    world.random_ignite(1)
                elif event.key == pygame.K_c:
                    renderer.camera_pos = np.array([0.0, 18.0, -80.0], dtype=np.float32)
                    renderer.camera_angle_x = 0.0
                    renderer.camera_angle_y = 0.0
                    renderer.zoom = 1.0
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_k:
                    world.save_map("maps/user_custom.txt")
                elif event.key == pygame.K_l:
                    world.load_map("maps/user_custom.txt")
                    particles.particles.clear()
                    time_step = 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click - ignite fire
                    mx, my = event.pos
                    sel = find_clicked_tree(renderer, world, mx, my)
                    if sel is not None:
                        cell = world.cells[sel]
                        if cell['state'] == TreeState.HEALTHY:
                            cell['state'] = TreeState.BURNING
                            cell['burning_time'] = 0
                elif event.button == 3:  # Right click - cycle terrain type
                    mx, my = event.pos
                    sel = find_clicked_tree(renderer, world, mx, my)
                    if sel is not None:
                        current_terrain_index = (current_terrain_index + 1) % len(terrain_types)
                        new_terrain = terrain_types[current_terrain_index]
                        cell = world.cells[sel]
                        cell['type'] = new_terrain
                        cell['state'] = TreeState.HEALTHY
                        cell['burning_time'] = 0
                        # Adjust height based on terrain type
                        if new_terrain in ['forest', 'dry_forest']:
                            cell['height'] = np.random.uniform(8.0, 14.0)
                        elif new_terrain == 'city':
                            cell['height'] = np.random.uniform(6.0, 20.0)
                        elif new_terrain in ['brush', 'dry_brush']:
                            cell['height'] = np.random.uniform(2.0, 4.0)
                        elif new_terrain in ['grass', 'dry_grass']:
                            cell['height'] = np.random.uniform(0.5, 1.5)
                        elif new_terrain == 'stone':
                            cell['height'] = np.random.uniform(3.0, 8.0)
                        elif new_terrain == 'swamp':
                            cell['height'] = np.random.uniform(1.0, 3.0)
                        else:  # river
                            cell['height'] = 0.2

        # Update simulation
        if not paused:
            world.spread_fire()
            particles.update()
            time_step += 1

        # Render everything
        renderer.clear()
        renderer.draw_ground_tiles()

        # Sort objects by depth for proper rendering
        visible = []
        for pos, cell in world.cells.items():
            proj = renderer.world_to_screen(cell['position'] + np.array([0.0, cell['height']*0.5, 0.0]))
            if proj:
                visible.append((proj[2], pos, cell))
        visible.sort(reverse=True, key=lambda x: x[0])

        # Draw all 3D objects
        for depth, pos, cell in visible:
            renderer.draw_3d_object(cell['position'], cell['height'], cell['state'], cell['type'], particle_system=particles)

        # Draw particles
        surf = pygame.Surface((SCREEN_W, SCREEN_H), flags=pygame.SRCALPHA)
        for p in particles.particles:
            proj = renderer.world_to_screen(p['pos'])
            if not proj:
                continue
            x, y = proj[0], proj[1]
            if p['type'] == 'ember':
                life_factor = max(0.0, p['life'] / 90.0)
                col = (255, int(160 * life_factor), int(40 * life_factor), int(200 * life_factor))
                radius = max(1, int(p['size'] * (0.8 + 0.6 * life_factor)))
                pygame.draw.circle(surf, col, (x, y), radius)
            else:
                life_factor = max(0.0, p['life'] / 90.0)
                col = (80, 80, 80, int(160 * life_factor))
                radius = max(1, int(p['size'] * (1.2 + 1.4 * (1-life_factor))))
                pygame.draw.circle(surf, col, (x, y), radius)
        screen.blit(surf, (0, 0))

        # Draw UI
        font = pygame.font.Font(None, 22)
        healthy = sum(1 for t in world.cells.values() if t['state'] == TreeState.HEALTHY)
        burning = sum(1 for t in world.cells.values() if t['state'] == TreeState.BURNING)
        burnt = sum(1 for t in world.cells.values() if t['state'] == TreeState.BURNT)
        stats = f"Time:{time_step}  Healthy:{healthy}  Burning:{burning}  Burnt:{burnt}  {'PAUSED' if paused else 'RUN'}"
        surf_stats = font.render(stats, True, (255, 255, 0))
        screen.blit(surf_stats, (8, SCREEN_H - 28))

        # Show current terrain type being placed
        terrain_info = f"Current terrain: {terrain_types[current_terrain_index]} (Right-click to change)"
        surf_terrain = font.render(terrain_info, True, (200, 200, 255))
        screen.blit(surf_terrain, (8, SCREEN_H - 50))

        # Draw instructions if enabled
        if show_help:
            draw_instructions(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
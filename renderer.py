# renderer.py
import pygame
import numpy as np
import math
from config import SCREEN_W, SCREEN_H, BACKGROUND_SKY, TILE_WORLD_SIZE, COLOR_TRUNK
from states import TreeState

# Base colors for each terrain type
TYPE_COLOR = {
    "grass": (60, 200, 60),
    "dry_grass": (200, 200, 60),
    "brush": (34, 139, 34),
    "dry_brush": (180, 100, 40),
    "forest": (34, 100, 34),
    "dry_forest": (180, 80, 20),
    "city": (120, 120, 120),
    "river": (40, 100, 200),
    "stone": (100, 100, 100),
    "swamp": (50, 120, 80)
}

class Renderer:
    def __init__(self, screen, world):
        self.screen = screen
        self.world = world
        self.width = SCREEN_W
        self.height = SCREEN_H

        self.camera_pos = np.array([0.0, 18.0, -80.0], dtype=np.float32)
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.zoom = 1.0

        self.light_dir = np.array([0.5, -1.0, 0.2], dtype=np.float32)
        self.light_dir /= (np.linalg.norm(self.light_dir) + 1e-9)

    # --------------- projection helpers ----------------
    def world_to_camera(self, world_pos):
        p = world_pos.astype(np.float32) - self.camera_pos.astype(np.float32)
        cy = math.cos(self.camera_angle_y); sy = math.sin(self.camera_angle_y)
        cx = math.cos(self.camera_angle_x); sx = math.sin(self.camera_angle_x)
        xz_x = p[0] * cy - p[2] * sy
        xz_z = p[0] * sy + p[2] * cy
        p[0], p[2] = xz_x, xz_z
        yz_y = p[1] * cx - p[2] * sx
        yz_z = p[1] * sx + p[2] * cx
        p[1], p[2] = yz_y, yz_z
        return p

    def world_to_screen(self, world_pos):
        cam = self.world_to_camera(world_pos)
        if cam[2] <= 0.001:
            return None
        f = 400.0 * self.zoom
        sx = (cam[0] * f) / cam[2] + self.width * 0.5
        sy = -(cam[1] * f) / cam[2] + self.height * 0.5
        return (int(sx), int(sy), cam[2])

    def lit_color(self, base_color, normal=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
        n = normal / (np.linalg.norm(normal) + 1e-9)
        intensity = max(0.0, np.dot(n, -self.light_dir))
        ambient = 0.35
        scale = min(1.0, ambient + 0.65 * intensity)
        r = min(255, max(0, int(base_color[0] * scale)))
        g = min(255, max(0, int(base_color[1] * scale)))
        b = min(255, max(0, int(base_color[2] * scale)))
        return (r, g, b)

    # ------------------ drawing ------------------
    def clear(self):
        self.screen.fill(BACKGROUND_SKY)

    def draw_ground_tiles(self):
        tile_half = TILE_WORLD_SIZE * 0.5
        tiles = []
        for (gx, gz), c in self.world.cells.items():
            cx = gx * TILE_WORLD_SIZE
            cz = gz * TILE_WORLD_SIZE
            corners = [
                np.array([cx - tile_half, 0.0, cz - tile_half], dtype=np.float32),
                np.array([cx + tile_half, 0.0, cz - tile_half], dtype=np.float32),
                np.array([cx + tile_half, 0.0, cz + tile_half], dtype=np.float32),
                np.array([cx - tile_half, 0.0, cz + tile_half], dtype=np.float32)
            ]
            proj = [self.world_to_screen(corner) for corner in corners]
            if all(p is not None for p in proj):
                depth = sum(p[2] for p in proj) / 4.0

                # Choose color based on state and type
                base_col = TYPE_COLOR.get(c['type'], (120, 120, 120))
                if c['state'] == TreeState.BURNING:
                    tile_color = (220, 80, 20)  # orange-red
                elif c['state'] == TreeState.BURNT:
                    tile_color = (70, 70, 70)   # ash gray
                else:
                    tile_color = base_col

                tiles.append((depth, [p[:2] for p in proj], tile_color))
        tiles.sort(reverse=True, key=lambda x: x[0])
        for depth, pts, col in tiles:
            pygame.draw.polygon(self.screen, col, pts)

    def draw_3d_object(self, world_pos, height, state, cell_type, particle_system=None):
        """Draw 3D representation based on terrain type"""
        if cell_type in ["forest", "dry_forest"]:
            return self.draw_tree_3d(world_pos, height, state, cell_type, particle_system)
        elif cell_type == "city":
            return self.draw_building(world_pos, height, state, particle_system)
        elif cell_type == "river":
            return self.draw_water(world_pos, height, state)
        elif cell_type in ["brush", "dry_brush"]:
            return self.draw_bush(world_pos, height, state, cell_type, particle_system)
        elif cell_type in ["grass", "dry_grass"]:
            return self.draw_grass(world_pos, height, state, cell_type, particle_system)
        elif cell_type == "stone":
            return self.draw_rock(world_pos, height, state)
        elif cell_type == "swamp":
            return self.draw_swamp(world_pos, height, state, particle_system)
        return False

    def draw_tree_3d(self, world_pos, height, state, cell_type, particle_system=None):
        """Draw tree with trunk and crown"""
        base = world_pos + np.array([0.0, 0.0, 0.0], dtype=np.float32)
        crown_top = world_pos + np.array([0.0, height, 0.0], dtype=np.float32)

        p_base = self.world_to_screen(base)
        p_crown_top = self.world_to_screen(crown_top)
        if p_base is None or p_crown_top is None:
            return False

        # Draw trunk
        trunk_segments = 6
        trunk_radius = 0.5
        trunk_color = self.lit_color(COLOR_TRUNK, normal=np.array([0.0, 1.0, 0.0]))
        for i in range(trunk_segments):
            ang = (2.0 * math.pi * i) / trunk_segments
            x = world_pos[0] + trunk_radius * math.cos(ang)
            z = world_pos[2] + trunk_radius * math.sin(ang)
            p0 = self.world_to_screen(np.array([x, 0.0, z], dtype=np.float32))
            p1 = self.world_to_screen(np.array([x, height * 0.55, z], dtype=np.float32))
            if p0 and p1:
                pygame.draw.line(self.screen, trunk_color, p0[:2], p1[:2], 2)

        # Draw crown
        crown_radius = max(1.2, height * 0.18)
        crown_segments = 8
        proj_base_ring = []
        for i in range(crown_segments):
            ang = (2.0 * math.pi * i) / crown_segments
            x = world_pos[0] + crown_radius * math.cos(ang)
            z = world_pos[2] + crown_radius * math.sin(ang)
            p_ring = self.world_to_screen(np.array([x, height * 0.45, z], dtype=np.float32))
            proj_base_ring.append(p_ring)

        # Crown color based on state
        if state == TreeState.HEALTHY:
            base_col = (34, 139, 34) if cell_type == "forest" else (180, 80, 20)
        elif state == TreeState.BURNING:
            base_col = (255, 140, 0)
        else:
            base_col = (100, 100, 100)
        lit_col = self.lit_color(base_col, normal=np.array([0.0, 1.0, 0.0]))

        for i in range(crown_segments):
            pA = proj_base_ring[i]
            pB = proj_base_ring[(i+1) % crown_segments]
            if pA and pB and p_crown_top:
                tri = [pA[:2], pB[:2], p_crown_top[:2]]
                pygame.draw.polygon(self.screen, lit_col, tri)
                pygame.draw.polygon(self.screen, (0, 0, 0), tri, 1)

        if state == TreeState.BURNING and particle_system is not None:
            particle_system.spawn_tree_particles(world_pos, height)

        return True

    def draw_building(self, world_pos, height, state, particle_system=None):
        """Draw rectangular building"""
        corners = []
        for dx in [-1.5, 1.5]:
            for dz in [-1.5, 1.5]:
                for dy in [0.0, height]:
                    pos = world_pos + np.array([dx, dy, dz], dtype=np.float32)
                    proj = self.world_to_screen(pos)
                    corners.append(proj)

        if any(c is None for c in corners):
            return False

        # Building color based on state
        if state == TreeState.BURNING:
            base_col = (255, 100, 0)
        elif state == TreeState.BURNT:
            base_col = (60, 60, 60)
        else:
            base_col = (120, 120, 120)

        lit_col = self.lit_color(base_col)

        # Draw faces (simplified cube)
        # Front face
        if corners[0] and corners[1] and corners[5] and corners[4]:
            face = [corners[0][:2], corners[1][:2], corners[5][:2], corners[4][:2]]
            pygame.draw.polygon(self.screen, lit_col, face)
            pygame.draw.polygon(self.screen, (0, 0, 0), face, 1)

        # Right face
        if corners[1] and corners[3] and corners[7] and corners[5]:
            face = [corners[1][:2], corners[3][:2], corners[7][:2], corners[5][:2]]
            darker_col = tuple(max(0, c - 30) for c in lit_col)
            pygame.draw.polygon(self.screen, darker_col, face)
            pygame.draw.polygon(self.screen, (0, 0, 0), face, 1)

        # Top face
        if corners[4] and corners[5] and corners[7] and corners[6]:
            face = [corners[4][:2], corners[5][:2], corners[7][:2], corners[6][:2]]
            lighter_col = tuple(min(255, c + 20) for c in lit_col)
            pygame.draw.polygon(self.screen, lighter_col, face)
            pygame.draw.polygon(self.screen, (0, 0, 0), face, 1)

        if state == TreeState.BURNING and particle_system is not None:
            particle_system.spawn_tree_particles(world_pos, height)

        return True

    def draw_water(self, world_pos, height, state):
        """Draw water surface"""
        p_center = self.world_to_screen(world_pos + np.array([0.0, height, 0.0]))
        if p_center is None:
            return False

        # Water is always blue, never burns
        water_color = self.lit_color((40, 100, 200))
        
        # Draw simple water surface as a circle
        radius = int(30 / max(1.0, p_center[2] * 0.1))
        if radius > 0:
            pygame.draw.circle(self.screen, water_color, p_center[:2], radius)
            pygame.draw.circle(self.screen, (0, 0, 100), p_center[:2], radius, 2)
        
        return True

    def draw_bush(self, world_pos, height, state, cell_type, particle_system=None):
        """Draw small bush/shrub"""
        segments = 6
        bush_radius = height * 0.8
        
        # Bush color based on type and state
        if state == TreeState.BURNING:
            base_col = (255, 140, 0)
        elif state == TreeState.BURNT:
            base_col = (80, 80, 80)
        else:
            base_col = (34, 139, 34) if cell_type == "brush" else (180, 100, 40)
        
        lit_col = self.lit_color(base_col)
        
        # Draw bush as hemisphere
        p_center = self.world_to_screen(world_pos + np.array([0.0, height * 0.5, 0.0]))
        if p_center is None:
            return False
            
        # Draw multiple circles to simulate volume
        for i in range(3):
            y_offset = height * (i * 0.3)
            radius_factor = (3 - i) / 3.0
            p = self.world_to_screen(world_pos + np.array([0.0, y_offset, 0.0]))
            if p:
                radius = max(1, int(bush_radius * radius_factor * 400 / max(1, p[2])))
                if radius > 0:
                    shade = tuple(max(0, c - i * 20) for c in lit_col)
                    pygame.draw.circle(self.screen, shade, p[:2], radius)

        if state == TreeState.BURNING and particle_system is not None:
            particle_system.spawn_tree_particles(world_pos, height, count_range=(0, 2))

        return True

    def draw_grass(self, world_pos, height, state, cell_type, particle_system=None):
        """Draw grass tufts"""
        if state == TreeState.BURNING:
            base_col = (255, 200, 0)
        elif state == TreeState.BURNT:
            base_col = (100, 100, 100)
        else:
            base_col = (60, 200, 60) if cell_type == "grass" else (200, 200, 60)
        
        lit_col = self.lit_color(base_col)
        
        # Draw several grass blades
        for i in range(8):
            angle = (2 * math.pi * i) / 8
            offset_x = 0.3 * math.cos(angle)
            offset_z = 0.3 * math.sin(angle)
            
            p_base = self.world_to_screen(world_pos + np.array([offset_x, 0.0, offset_z]))
            p_top = self.world_to_screen(world_pos + np.array([offset_x, height, offset_z]))
            
            if p_base and p_top:
                pygame.draw.line(self.screen, lit_col, p_base[:2], p_top[:2], 1)

        if state == TreeState.BURNING and particle_system is not None:
            particle_system.spawn_tree_particles(world_pos, height, count_range=(0, 1))

        return True

    def draw_rock(self, world_pos, height, state):
        """Draw rock formation"""
        rock_color = self.lit_color((100, 100, 100))
        
        # Draw rock as irregular shape
        segments = 8
        proj_points = []
        
        for i in range(segments):
            angle = (2 * math.pi * i) / segments
            # Vary radius for irregular shape
            radius = 1.0 + 0.3 * math.sin(3 * angle)
            x = world_pos[0] + radius * math.cos(angle)
            z = world_pos[2] + radius * math.sin(angle)
            
            p_base = self.world_to_screen(np.array([x, 0.0, z]))
            p_top = self.world_to_screen(np.array([x, height, z]))
            
            if p_base and p_top:
                proj_points.append((p_base, p_top))

        # Draw rock faces
        for i in range(len(proj_points)):
            if i < len(proj_points) - 1:
                p1_base, p1_top = proj_points[i]
                p2_base, p2_top = proj_points[i + 1]
                
                face = [p1_base[:2], p2_base[:2], p2_top[:2], p1_top[:2]]
                shade = tuple(max(0, c - (i * 10) % 40) for c in rock_color)
                pygame.draw.polygon(self.screen, shade, face)
                pygame.draw.polygon(self.screen, (0, 0, 0), face, 1)

        return True

    def draw_swamp(self, world_pos, height, state, particle_system=None):
        """Draw swamp with water and vegetation"""
        # Swamp has both water and some vegetation
        if state == TreeState.BURNING:
            base_col = (255, 140, 0)
        elif state == TreeState.BURNT:
            base_col = (80, 80, 80)
        else:
            base_col = (50, 120, 80)
        
        lit_col = self.lit_color(base_col)
        
        # Draw water base
        p_water = self.world_to_screen(world_pos + np.array([0.0, 0.1, 0.0]))
        if p_water:
            radius = max(5, int(25 / max(1.0, p_water[2] * 0.1)))
            water_col = self.lit_color((40, 80, 120))
            pygame.draw.circle(self.screen, water_col, p_water[:2], radius)
        
        # Draw swamp vegetation (small reeds/plants)
        for i in range(6):
            angle = (2 * math.pi * i) / 6
            offset_x = 1.0 * math.cos(angle)
            offset_z = 1.0 * math.sin(angle)
            
            p_base = self.world_to_screen(world_pos + np.array([offset_x, 0.0, offset_z]))
            p_top = self.world_to_screen(world_pos + np.array([offset_x, height, offset_z]))
            
            if p_base and p_top:
                pygame.draw.line(self.screen, lit_col, p_base[:2], p_top[:2], 2)

        if state == TreeState.BURNING and particle_system is not None:
            particle_system.spawn_tree_particles(world_pos, height, count_range=(0, 2))

        return True

    # Legacy method name for compatibility
    def draw_tree(self, world_pos, height, state, cell_type, particle_system=None):
        """Legacy method - redirects to tree 3D drawing"""
        return self.draw_tree_3d(world_pos, height, state, cell_type, particle_system)
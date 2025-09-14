# renderer_opengl.py
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from config import TILE_WORLD_SIZE, TYPE_COLOR, GRID_SIZE
from states import TreeState
from algorithms import bresenham_line, midpoint_circle_offsets, scanfill_polygon

class GLRenderer:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self._init_gl()

    def _init_gl(self):
        glViewport(0, 0, self.w, self.h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, float(self.w) / float(self.h), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (1.0, 4.0, 2.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))

    def clear(self):
        glClearColor(0.53, 0.81, 0.92, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def set_camera(self, cam_pos, cam_target):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(cam_pos[0], cam_pos[1], cam_pos[2],
                  cam_target[0], cam_target[1], cam_target[2],
                  0.0, 1.0, 0.0)
        
    def draw_barrier(self, x0, z0, x1, z1):
        """Draw a red line barrier using Bresenham algorithm (algorithms.py)."""
        # convert grid coords to world coords
        pts = bresenham_line(x0, z0, x1, z1)
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 0.0, 0.0)  # red
        glBegin(GL_LINES)
        for (gx, gz) in pts:
            wx = gx * TILE_WORLD_SIZE
            wz = gz * TILE_WORLD_SIZE
            glVertex3f(wx, 0.1, wz)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_polygon3d(self, x, y, z, offsets, col, height=0.2, scale=0.3):
        """Extrude polygon dari midpoint_circle ke 3D shape."""
        glDisable(GL_LIGHTING)
        glColor3f(*col)

        # buat koordinat polygon (loop tertutup, urut)
        pts = [(x + dx * scale, z + dz * scale) for dx, dz in offsets]

        # --- atas (filled) ---
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x, y+height, z)  # center
        for px, pz in pts:
            glVertex3f(px, y+height, pz)
        # loop kembali ke titik awal
        px, pz = pts[0]
        glVertex3f(px, y+height, pz)
        glEnd()

        # --- bawah (filled) ---
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x, y, z)  # center
        for px, pz in pts:
            glVertex3f(px, y, pz)
        px, pz = pts[0]
        glVertex3f(px, y, pz)
        glEnd()

        # --- sisi (extrude walls) ---
        glBegin(GL_QUAD_STRIP)
        for px, pz in pts:
            glVertex3f(px, y, pz)
            glVertex3f(px, y+height, pz)
        # close loop
        px, pz = pts[0]
        glVertex3f(px, y, pz)
        glVertex3f(px, y+height, pz)
        glEnd()

        glEnable(GL_LIGHTING)

    def draw_ground(self, grid_size, tile_size):
        half = grid_size / 2.0
        x0 = -half * tile_size
        x1 = half * tile_size
        z0 = -half * tile_size
        z1 = half * tile_size
        glDisable(GL_LIGHTING)
        glColor3f(0.502, 0.341, 0.051)
        glBegin(GL_QUADS)
        glVertex3f(x0, 0.0, z0)
        glVertex3f(x1, 0.0, z0)
        glVertex3f(x1, 0.0, z1)
        glVertex3f(x0, 0.0, z1)
        glEnd()
        glEnable(GL_LIGHTING)

    def debug_cube(self):
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 0.2, 0.2)
        self._unit_cube_at(0.0, 0.0, 0.0, size=2.0)
        glEnable(GL_LIGHTING)

    def draw_object(self, world_pos, height, tile_size, state=None, cell_type="forest", particles=None, morph_info=None):
        x, y, z = world_pos
        # If morph present draw blended shapes
        if morph_info:
            t = max(0.0, min(1.0, morph_info['progress']))
            prev = morph_info['prev_type']
            target = morph_info['target_type']
            # draw prev with (1-t) scale and fade, then target with t
            self._draw_morphed_pair(x, y, z, height, tile_size, state, prev, target, t, particles)
            return

        # normal draw
        base_col = TYPE_COLOR.get(cell_type, (0.6, 0.6, 0.6))
        if state == TreeState.BURNING:
            col = (1.0, 0.35, 0.06)
            if particles:
                particles.spawn_tree_particles(world_pos, height, count=1)
        elif state == TreeState.BURNT:
            col = (0.12, 0.12, 0.12)
        else:
            col = base_col

        self._draw_by_type(x, y, z, height, tile_size, cell_type, col)

    def _draw_morphed_pair(self, x, y, z, h, tile_size, state, prev_type, target_type, t, particles):
        """Draw both types blended using t in [0..1]."""

        # ==== Custom morph rules ====
        if prev_type == "city" and target_type == "river":
            # Building tenggelam, river naik
            col_city = TYPE_COLOR.get("city", (0.6, 0.6, 0.6))
            col_river = (0.2, 0.4, 1.0)

            # City shrink & sink
            self._draw_building(x, y - t * 0.5, z, h * (1.0 - t), col_city)

            # River grow & fade in
            glColor4f(col_river[0], col_river[1], col_river[2], 0.4 + 0.6 * t)
            glPushMatrix()
            glTranslatef(x, y + 0.05, z)
            glScalef(1.2 * t, 0.7, 1.2 * t)
            quad = gluNewQuadric()
            gluSphere(quad, TILE_WORLD_SIZE * 0.6, 20, 20)
            glPopMatrix()
            return

        if prev_type == "river" and target_type == "stone":
            # Air memudar → batu muncul
            col_river = (0.2, 0.4, 1.0)
            col_stone = TYPE_COLOR.get("stone", (0.5, 0.5, 0.5))

            # River fade out
            glColor4f(col_river[0], col_river[1], col_river[2], max(0.0, 0.7 * (1.0 - t)))
            glPushMatrix()
            glTranslatef(x, y + 0.05, z)
            glScalef(1.2, 0.7, 1.2)
            quad = gluNewQuadric()
            gluSphere(quad, TILE_WORLD_SIZE * 0.6, 20, 20)
            glPopMatrix()

            # Stone grow
            self._draw_rock(x, y, z, h * t, col_stone)
            return

        # ==== Default morph (semua selain custom) ====
        s_prev = 1.0 - t
        s_tgt = t
        c_prev = TYPE_COLOR.get(prev_type, (0.6,0.6,0.6))
        c_tgt = TYPE_COLOR.get(target_type, (0.6,0.6,0.6))
        col_prev = tuple(c_prev[i] * (1.0 - 0.5*t) for i in range(3))
        col_tgt = tuple(c_tgt[i] * (0.5 + 0.5*t) for i in range(3))

        self._draw_by_type(x, y, z, h * s_prev, tile_size, prev_type, col_prev, scale_uniform=s_prev, particles=None)
        self._draw_by_type(x, y, z, h * max(0.5, s_tgt), tile_size, target_type, col_tgt, scale_uniform=max(0.5, s_tgt), particles=particles)


    def _draw_by_type(self, x, y, z, h, tile_size, ttype, col, scale_uniform=1.0, particles=None):
        """Dispatch to type-specific drawing; scale_uniform applied to overall size."""
        if ttype in ("forest", "dry_forest"):
            self._draw_tree(x, y, z, h * scale_uniform, col)
        elif ttype in ("grass", "dry_grass"):
            self._draw_grass(x, y, z, h * scale_uniform, col)
        elif ttype in ("brush", "dry_brush"):
            self._draw_bush(x, y, z, h * scale_uniform, col)
        elif ttype == "city":
            self._draw_building(x, y, z, h * max(0.6, scale_uniform*1.2), col)
        elif ttype == "river":
            self._draw_water(x, y, z, col)
        elif ttype == "stone":
            self._draw_rock(x, y, z, h * scale_uniform, col)
        elif ttype == "swamp":
            self._draw_swamp(x, y, z, h * scale_uniform, col)
        else:
            self._unit_cube_at(x, y, z, size=tile_size*0.8*scale_uniform, height=h*scale_uniform)

    # primitives
    def _unit_cube_at(self, x, y, z, size=1.0, height=None):
        half = size/2.0
        if height is None:
            height = size
        glPushMatrix()
        glTranslatef(x, y + height/2.0, z)
        glScalef(size, height, size)
        glBegin(GL_QUADS)
        # front
        glNormal3f(0,0,1)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f( 0.5, -0.5, 0.5)
        glVertex3f( 0.5,  0.5, 0.5)
        glVertex3f(-0.5,  0.5, 0.5)
        # back
        glNormal3f(0,0,-1)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f( 0.5, -0.5, -0.5)
        glVertex3f( 0.5,  0.5, -0.5)
        glVertex3f(-0.5,  0.5, -0.5)
        # left
        glNormal3f(-1,0,0)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5,  0.5)
        glVertex3f(-0.5,  0.5,  0.5)
        glVertex3f(-0.5,  0.5, -0.5)
        # right
        glNormal3f(1,0,0)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5,  0.5)
        glVertex3f(0.5,  0.5,  0.5)
        glVertex3f(0.5,  0.5, -0.5)
        # top
        glNormal3f(0,1,0)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f( 0.5, 0.5, -0.5)
        glVertex3f( 0.5, 0.5,  0.5)
        glVertex3f(-0.5, 0.5,  0.5)
        # bottom
        glNormal3f(0,-1,0)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f( 0.5, -0.5, -0.5)
        glVertex3f( 0.5, -0.5,  0.5)
        glVertex3f(-0.5, -0.5,  0.5)
        glEnd()
        glPopMatrix()

    def _draw_tree(self, x, y, z, h, col):
        # trunk box
        glColor3f(0.45, 0.28, 0.14)
        self._unit_cube_at(x, y, z, size=TILE_WORLD_SIZE*0.22, height=h*0.55)
        # crown (skip if burnt)
        if col != (0.12,0.12,0.12):
            glColor3f(*col)
            glPushMatrix()
            glTranslatef(x, y + h * 0.75, z)
            quad = gluNewQuadric()
            gluSphere(quad, TILE_WORLD_SIZE * 0.35 * max(0.3, h/6.0), 12, 12)
            glPopMatrix()

    def _draw_building(self, x, y, z, h, col):
        glColor3f(*col)
        self._unit_cube_at(x, y, z, size=TILE_WORLD_SIZE*0.9, height=h)

    def _draw_grass(self, x, y, z, h, col):
        offsets = midpoint_circle_offsets(8)  # radius kecil
        self._draw_polygon3d(x, y, z, offsets, col, height=0.15, scale=0.3)

    def _draw_bush(self, x, y, z, h, col):
        glColor3f(*col)
        glPushMatrix()
        glTranslatef(x, y + h*0.45, z)
        quad = gluNewQuadric()
        gluSphere(quad, TILE_WORLD_SIZE * 0.35, 10, 10)
        glPopMatrix()
       

    def _draw_rock(self, x, y, z, h, col):
        glColor3f(*col)
        self._unit_cube_at(x, y, z, size=TILE_WORLD_SIZE * 0.6, height=h*0.5)

    def _draw_swamp(self, x, y, z, h, col):
        offsets = midpoint_circle_offsets(12)  # radius lebih besar
        self._draw_polygon3d(x, y, z, offsets, col, height=0.5, scale=0.35)

    def _draw_water(self, x, y, z, col):
        """River as a wide, thin blue surface using scaled sphere."""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(0.2, 0.4, 1.0, 0.7)  # biru transparan
        glPushMatrix()
        glTranslatef(x, y + 0.05, z)
        glScalef(1.2, 0.7, 1.2)
        quad = gluNewQuadric()
        gluSphere(quad, TILE_WORLD_SIZE * 0.6, 20, 20)
        glPopMatrix()

        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    # picking
    def pick_cell(self, mx, my, world):
        viewport = glGetIntegerv(GL_VIEWPORT)
        win_x = float(mx)
        win_y = float(viewport[3] - my)
        model = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        nx, ny, nz = gluUnProject(win_x, win_y, 0.0, model, proj, viewport)
        fx, fy, fz = gluUnProject(win_x, win_y, 1.0, model, proj, viewport)
        rx = fx - nx; ry = fy - ny; rz = fz - nz
        if abs(ry) < 1e-6:
            return None
        t = - (ny) / ry
        if t < 0:
            return None
        wx = nx + rx * t
        wz = nz + rz * t
        gx = int(round(wx / TILE_WORLD_SIZE))
        gz = int(round(wz / TILE_WORLD_SIZE))
        if (gx, gz) in world.cells:
            return (gx, gz)
        return None

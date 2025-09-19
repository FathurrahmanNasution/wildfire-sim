# tree_click_fall.py
import math, random, ctypes, time
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

# ---------------- Matrix & Vector helpers (manual) ----------------
def mat_identity():
    return np.array([[1.0,0.0,0.0,0.0],
                     [0.0,1.0,0.0,0.0],
                     [0.0,0.0,1.0,0.0],
                     [0.0,0.0,0.0,1.0]], dtype=np.float32)

def mat_translation(tx,ty,tz):
    M = mat_identity(); M[0,3]=tx; M[1,3]=ty; M[2,3]=tz; return M

def mat_scaling(sx,sy,sz):
    M = mat_identity(); M[0,0]=sx; M[1,1]=sy; M[2,2]=sz; return M

def mat_rotation_x(a):
    c=math.cos(a); s=math.sin(a)
    M = mat_identity()
    M[1,1]=c; M[1,2]=-s; M[2,1]=s; M[2,2]=c
    return M

def mat_rotation_y(a):
    c=math.cos(a); s=math.sin(a)
    M = mat_identity()
    M[0,0]=c; M[0,2]=s; M[2,0]=-s; M[2,2]=c
    return M

def mat_rotation_axis(axis, angle):
    # axis: 3-array normalized
    x,y,z = axis
    c = math.cos(angle); s = math.sin(angle); C = 1-c
    # Rodrigues formula -> 3x3
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ], dtype=np.float32)
    M = mat_identity()
    M[0:3,0:3] = R
    return M

def mat_mul(A,B):
    return np.dot(A,B)

def vec_norm(v):
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8: return np.array(v, dtype=np.float64)
    return v / n

def cross(a,b):
    return np.cross(a,b)

# ---------------- FastTree (manual geometry -> VBO) ----------------
class FastTree:
    def __init__(self, cylinder_segments=12, sphere_lat=6, sphere_lon=8):
        self.cylinder_segments = cylinder_segments
        self.sphere_lat = sphere_lat
        self.sphere_lon = sphere_lon

        # CPU-side geometry
        self.trunk_vertices = []
        self.trunk_normals  = []
        self.trunk_indices  = []

        self.leaf_vertices = []
        self.leaf_normals  = []
        self.leaf_indices  = []

        # GPU buffers
        self.vbo_trunk = None
        self.ebo_trunk = None
        self.trunk_index_count = 0

        self.vbo_leaf = None
        self.ebo_leaf = None
        self.leaf_index_count = 0

    def clear_geometry(self):
        self.trunk_vertices.clear(); self.trunk_normals.clear(); self.trunk_indices.clear()
        self.leaf_vertices.clear(); self.leaf_normals.clear(); self.leaf_indices.clear()
        # delete GL buffers if exist
        try:
            if self.vbo_trunk: glDeleteBuffers(1, [self.vbo_trunk]); self.vbo_trunk=None
            if self.ebo_trunk: glDeleteBuffers(1, [self.ebo_trunk]); self.ebo_trunk=None
            if self.vbo_leaf:  glDeleteBuffers(1, [self.vbo_leaf]);  self.vbo_leaf=None
            if self.ebo_leaf:  glDeleteBuffers(1, [self.ebo_leaf]);  self.ebo_leaf=None
        except Exception:
            pass

    # Cylinder with caps (pivot at base y=0)
    def generate_cylinder_with_caps(self, transform_matrix, height, radius):
        start = len(self.trunk_vertices)
        seg = self.cylinder_segments
        # rim vertices (bottom and top)
        for i in range(seg):
            ang = 2.0*math.pi*i/seg
            x = radius*math.cos(ang); z = radius*math.sin(ang)
            bottom_local = [x, 0.0, z, 1.0]
            top_local    = [x, height, z, 1.0]
            bw = np.dot(transform_matrix, np.array(bottom_local, dtype=np.float32))
            tw = np.dot(transform_matrix, np.array(top_local, dtype=np.float32))
            self.trunk_vertices.append([bw[0], bw[1], bw[2]])
            self.trunk_vertices.append([tw[0], tw[1], tw[2]])
        # sides
        for i in range(seg):
            a = start + i*2
            b = start + ((i+1)%seg)*2
            self.trunk_indices.extend([a, b, a+1])
            self.trunk_indices.extend([a+1, b, b+1])
        # bottom cap center
        bottom_center_local = [0.0,0.0,0.0,1.0]
        bc = np.dot(transform_matrix, np.array(bottom_center_local, dtype=np.float32))
        bc_idx = len(self.trunk_vertices)
        self.trunk_vertices.append([bc[0], bc[1], bc[2]])
        for i in range(seg):
            rim_i = start + i*2
            rim_next = start + ((i+1)%seg)*2
            # center, next, current (to make normal downward or outward - check winding)
            self.trunk_indices.extend([bc_idx, rim_next, rim_i])
        # top cap center
        top_center_local = [0.0, height, 0.0, 1.0]
        tc = np.dot(transform_matrix, np.array(top_center_local, dtype=np.float32))
        tc_idx = len(self.trunk_vertices)
        self.trunk_vertices.append([tc[0], tc[1], tc[2]])
        for i in range(seg):
            rim_top_i = start + i*2 + 1
            rim_top_next = start + ((i+1)%seg)*2 + 1
            self.trunk_indices.extend([tc_idx, rim_top_i, rim_top_next])

    def generate_sphere(self, transform_matrix, radius):
        start = len(self.leaf_vertices)
        latN = self.sphere_lat; lonN = self.sphere_lon
        for lat in range(latN+1):
            phi = math.pi * lat / latN
            for lon in range(lonN+1):
                theta = 2.0*math.pi*lon/lonN
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                p_local = [x,y,z,1.0]
                pw = np.dot(transform_matrix, np.array(p_local, dtype=np.float32))
                self.leaf_vertices.append([pw[0], pw[1], pw[2]])
        for lat in range(latN):
            for lon in range(lonN):
                v1 = start + lat*(lonN+1) + lon
                v2 = v1 + (lonN+1)
                v3 = v1 + 1
                v4 = v2 + 1
                if lat != 0:
                    self.leaf_indices.extend([v1, v2, v3])
                if lat != latN-1:
                    self.leaf_indices.extend([v3, v2, v4])

    def build_tree(self, transform_matrix, height, radius, depth, max_depth):
        if depth > max_depth: return
        self.generate_cylinder_with_caps(transform_matrix, height, radius)
        if depth == max_depth:
            leaf_count = random.randint(3,6)
            for _ in range(leaf_count):
                ox = random.uniform(-0.35, 0.35)
                oy = random.uniform(-0.05, 0.25)
                oz = random.uniform(-0.35, 0.35)
                leaf_translation = mat_translation(ox, height + oy, oz)
                leaf_transform = mat_mul(transform_matrix, leaf_translation)
                leaf_size = 0.30 * (0.8**depth) * random.uniform(0.85,1.15)
                self.generate_sphere(leaf_transform, leaf_size)
            return
        branch_count = max(2, 4-depth) + random.randint(0,1)
        for i in range(branch_count):
            base_angle = 2.0*math.pi*i/branch_count
            final_angle = base_angle + random.uniform(-0.4,0.4)
            tilt_angle = random.uniform(math.radians(25), math.radians(60))
            attach_height = height * random.uniform(0.6,0.85)
            T1 = mat_translation(0.0, attach_height, 0.0)
            R1 = mat_rotation_y(final_angle)
            R2 = mat_rotation_x(-tilt_angle)
            temp = mat_mul(T1, R1)
            temp = mat_mul(temp, R2)
            branch_transform = mat_mul(transform_matrix, temp)
            new_h = height * random.uniform(0.6, 0.75)
            new_r = radius * 0.65
            self.build_tree(branch_transform, new_h, new_r, depth+1, max_depth)

    def compute_normals(self):
        # trunk normals
        nv = len(self.trunk_vertices)
        self.trunk_normals = [[0.0,0.0,0.0] for _ in range(nv)]
        for i in range(0, len(self.trunk_indices), 3):
            ia,ib,ic = self.trunk_indices[i:i+3]
            va = np.array(self.trunk_vertices[ia]); vb = np.array(self.trunk_vertices[ib]); vc = np.array(self.trunk_vertices[ic])
            e1 = vb - va; e2 = vc - va
            n = np.cross(e1, e2)
            for idx in (ia,ib,ic):
                self.trunk_normals[idx][0] += n[0]; self.trunk_normals[idx][1] += n[1]; self.trunk_normals[idx][2] += n[2]
        for i in range(nv):
            v = self.trunk_normals[i]; l = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
            if l>1e-8: self.trunk_normals[i] = [v[0]/l, v[1]/l, v[2]/l]
            else: self.trunk_normals[i] = [0.0,1.0,0.0]

        # leaf normals
        nv = len(self.leaf_vertices)
        self.leaf_normals = [[0.0,0.0,0.0] for _ in range(nv)]
        for i in range(0, len(self.leaf_indices), 3):
            ia,ib,ic = self.leaf_indices[i:i+3]
            va = np.array(self.leaf_vertices[ia]); vb = np.array(self.leaf_vertices[ib]); vc = np.array(self.leaf_vertices[ic])
            e1 = vb - va; e2 = vc - va
            n = np.cross(e1, e2)
            for idx in (ia,ib,ic):
                self.leaf_normals[idx][0] += n[0]; self.leaf_normals[idx][1] += n[1]; self.leaf_normals[idx][2] += n[2]
        for i in range(len(self.leaf_normals)):
            v = self.leaf_normals[i]; l = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
            if l>1e-8: self.leaf_normals[i] = [v[0]/l, v[1]/l, v[2]/l]
            else: self.leaf_normals[i] = [0.0,1.0,0.0]

    def upload_to_gpu(self):
        # trunk
        if len(self.trunk_vertices):
            pos = np.array(self.trunk_vertices, dtype=np.float32)
            nor = np.array(self.trunk_normals, dtype=np.float32)
            inter = np.empty((pos.shape[0], 6), dtype=np.float32)
            inter[:,0:3] = pos; inter[:,3:6] = nor
            flat = inter.flatten()
            idxs = np.array(self.trunk_indices, dtype=np.uint32)
            self.vbo_trunk = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_trunk)
            glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.ebo_trunk = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_trunk)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.nbytes, idxs, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            self.trunk_index_count = idxs.size
        # leaf
        if len(self.leaf_vertices):
            pos = np.array(self.leaf_vertices, dtype=np.float32)
            nor = np.array(self.leaf_normals, dtype=np.float32)
            inter = np.empty((pos.shape[0], 6), dtype=np.float32)
            inter[:,0:3] = pos; inter[:,3:6] = nor
            flat = inter.flatten()
            idxs = np.array(self.leaf_indices, dtype=np.uint32)
            self.vbo_leaf = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_leaf)
            glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.ebo_leaf = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_leaf)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.nbytes, idxs, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            self.leaf_index_count = idxs.size

    def draw(self):
        # trunk
        if self.vbo_trunk and self.ebo_trunk:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_trunk)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_trunk)
            stride = 6 * 4
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(3*4))
            glColor3f(0.6, 0.4, 0.2)
            glDrawElements(GL_TRIANGLES, self.trunk_index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        # leaves
        if self.vbo_leaf and self.ebo_leaf:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_leaf)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_leaf)
            stride = 6 * 4
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(3*4))
            glColor3f(0.2, 0.8, 0.2)
            glDrawElements(GL_TRIANGLES, self.leaf_index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

# ---------------- TreeInstance: per-model transform + falling state ----------------
class TreeInstance:
    def __init__(self, model: FastTree, position=(0,0,0), base_height=3.0, base_radius=0.3):
        self.model = model
        self.pos = np.array(position, dtype=np.float32)
        self.yaw = random.uniform(0.0, 2.0*math.pi)
        self.scale = random.uniform(0.8, 1.3)
        self.base_height = base_height
        self.base_radius = base_radius

        # falling state
        self.falling = False
        self.fall_axis = np.array([1.0,0.0,0.0], dtype=np.float32)  # axis in local coords
        self.fall_angle = 0.0
        self.fall_speed = math.radians(90)  # radians/sec to reach 90deg in 1s (adjust)
        self.max_fall = math.radians(90.0)

    def model_matrix(self):
        S = mat_scaling(self.scale, self.scale, self.scale)
        Ry = mat_rotation_y(self.yaw)
        Rf = mat_rotation_axis(self.fall_axis, self.fall_angle) if self.falling or self.fall_angle!=0.0 else mat_identity()
        T = mat_translation(self.pos[0], self.pos[1], self.pos[2])
        # M = T * Rf * Ry * S
        M = mat_mul(T, mat_mul(Rf, mat_mul(Ry, S)))
        return M

    def update(self, dt):
        if self.falling:
            # increment angle
            self.fall_angle += self.fall_speed * dt
            if self.fall_angle >= self.max_fall:
                self.fall_angle = self.max_fall
                self.falling = False  # finished
        # else nothing

    def start_fall_towards(self, target_point):
        # target_point: world coords of click hit (x,y,z)
        # direction on XZ plane from base to target:
        dir_xz = np.array([target_point[0]-self.pos[0], 0.0, target_point[2]-self.pos[2]], dtype=np.float64)
        if np.linalg.norm(dir_xz) < 1e-4:
            # clicked near base — random direction
            theta = random.uniform(0, 2*math.pi)
            dir_xz = np.array([math.cos(theta), 0.0, math.sin(theta)])
        else:
            dir_xz = dir_xz / np.linalg.norm(dir_xz)
        up = np.array([0.0,1.0,0.0])
        axis = np.cross(up, dir_xz)
        if np.linalg.norm(axis) < 1e-6:
            # degenerate, pick arbitrary horizontal axis
            axis = np.array([1.0,0.0,0.0])
        axis = axis / np.linalg.norm(axis)
        # set fall axis in LOCAL model coords: because we will apply Rf before Ry and S,
        # we want axis relative to model basis. Our model basis is same as world for simplicity,
        # but must rotate axis by inverse yaw so fall rotates relative to tree yaw.
        # compute axis_local = Ry^{-1} * axis
        Ry = mat_rotation_y(self.yaw)
        Ry_inv = np.linalg.inv(Ry)
        axis4 = np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float32)
        axis_local4 = np.dot(Ry_inv, axis4)
        axis_local = axis_local4[0:3]
        axis_local = axis_local / (np.linalg.norm(axis_local)+1e-9)
        self.fall_axis = axis_local.astype(np.float32)
        self.falling = True
        self.fall_angle = 0.0

# ---------------- OpenGL + scene ----------------
def setup_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 5.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2,0.2,0.2,1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0,1.0,0.9,1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.1,0.1,0.1,1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 10.0)
    glClearColor(0.1,0.1,0.15,1.0)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glShadeModel(GL_SMOOTH)

def mouse_to_world_ray(mx, my, width, height):
    # returns (ray_origin, ray_dir) in world coords using gluUnProject
    viewport = glGetIntegerv(GL_VIEWPORT)
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    # note: pygame mouse y=0 top, gluUnProject expects same
    win_x = float(mx)
    win_y = float(viewport[3]) - float(my)
    # near
    nz = 0.0
    fx, fy, fz = gluUnProject(win_x, win_y, nz, modelview, projection, viewport)
    # far (z=1)
    nz = 1.0
    gx, gy, gz = gluUnProject(win_x, win_y, nz, modelview, projection, viewport)
    origin = np.array([fx, fy, fz], dtype=np.float64)
    far = np.array([gx, gy, gz], dtype=np.float64)
    dir = far - origin
    dir = dir / np.linalg.norm(dir)
    return origin, dir

def ray_vs_capped_cylinder(ray_o, ray_d, cyl_center, radius, height):
    # cylinder axis along +Y, base at (cx,0,cz), top at y=height
    cx, cz = cyl_center[0], cyl_center[2]
    # project to XZ plane
    ox = ray_o[0] - cx; oz = ray_o[2] - cz
    dx = ray_d[0]; dz = ray_d[2]
    a = dx*dx + dz*dz
    b = 2.0*(ox*dx + oz*dz)
    c = ox*ox + oz*oz - radius*radius
    disc = b*b - 4*a*c
    hits = []
    if abs(a) > 1e-8 and disc >= 0.0:
        sqrt_d = math.sqrt(disc)
        t1 = (-b - sqrt_d) / (2*a)
        t2 = (-b + sqrt_d) / (2*a)
        for t in (t1,t2):
            if t <= 1e-6: continue
            y = ray_o[1] + ray_d[1]*t
            if 0.0 <= y <= height:
                hits.append((t, np.array([ray_o[0]+ray_d[0]*t, y, ray_o[2]+ray_d[2]*t])))
    # also check caps (top and bottom) as planes
    # bottom plane y=0
    if abs(ray_d[1])>1e-8:
        t_plane = (0.0 - ray_o[1]) / ray_d[1]
        if t_plane>1e-6:
            p = ray_o + ray_d * t_plane
            if (p[0]-cx)**2 + (p[2]-cz)**2 <= radius*radius:
                hits.append((t_plane, p))
        # top plane y=height
        t_plane2 = (height - ray_o[1]) / ray_d[1]
        if t_plane2>1e-6:
            p2 = ray_o + ray_d * t_plane2
            if (p2[0]-cx)**2 + (p2[2]-cz)**2 <= radius*radius:
                hits.append((t_plane2, p2))
    if not hits: return None
    # return nearest positive hit
    hits.sort(key=lambda x: x[0])
    return hits[0]  # (t, point)

def main():
    pygame.init()
    size = (1200, 800)
    pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Tree: click to fall (R=regenerate)")
    setup_opengl()
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45.0, size[0]/size[1], 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    camera = np.array([8.0,5.0,10.0])
    gluLookAt(camera[0],camera[1],camera[2], 0.0,2.0,0.0, 0.0,1.0,0.0)

    # build initial tree model (single)
    model = FastTree(cylinder_segments=12, sphere_lat=6, sphere_lon=8)
    def build_tree_random(seed=None):
        if seed is not None: random.seed(seed)
        model.clear_geometry()
        root_transform = mat_identity()
        model.build_tree(root_transform, height=3.0, radius=0.3, depth=0, max_depth=4)
        model.compute_normals()
        model.upload_to_gpu()
    build_tree_random(seed=42)

    # create an instance placed at origin (on ground y=0)
    inst = TreeInstance(model, position=(0.0, 0.0, 0.0), base_height=3.0, base_radius=0.3)

    clock = pygame.time.Clock()
    running = True
    last_time = time.time()
    # no auto rotation: remove scene rot
    while running:
        dt = clock.tick(60) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running=False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: running=False
                elif ev.key == pygame.K_r:
                    # regenerate shape and randomize instance scale/yaw
                    build_tree_random(seed=None)
                    inst.yaw = random.uniform(0, 2*math.pi)
                    inst.scale = random.uniform(0.75, 1.35)
                    inst.falling = False; inst.fall_angle = 0.0
                    print("Regenerated tree. scale:", inst.scale, "yaw:", inst.yaw)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = pygame.mouse.get_pos()
                ray_o, ray_d = mouse_to_world_ray(mx, my, size[0], size[1])
                # test ray vs cylinder (position inst.pos, radius*scale, height*scale)
                cyl_center = np.array([inst.pos[0], 0.0, inst.pos[2]])
                r = inst.base_radius * inst.scale
                h = inst.base_height * inst.scale
                hit = ray_vs_capped_cylinder(ray_o, ray_d, cyl_center, r, h)
                if hit is not None:
                    t, pt = hit
                    inst.start_fall_towards(pt)
                    print("Tree hit! will fall toward", pt)
                else:
                    print("Miss")

        # update
        inst.update(dt)

        # draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        # apply instance model matrix
        M = inst.model_matrix().T  # OpenGL expects column-major; but glMultMatrixf expects sequence
        glMultMatrixf(M.flatten())
        inst.model.draw()
        glPopMatrix()

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

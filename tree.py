import math, random, ctypes, time
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

# -------------- Helpers --------------
def mat_identity():
    return np.array([[1.0,0.0,0.0,0.0],
                     [0.0,1.0,0.0,0.0],
                     [0.0,0.0,1.0,0.0],
                     [0.0,0.0,0.0,1.0]], dtype=np.float32)

def normalize(v):
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

def orthonormal_basis_from_axis(axis):
    up = normalize(axis)
    if abs(up[1]) < 0.999:
        tmp = np.array([0.0,1.0,0.0], dtype=np.float64)
    else:
        tmp = np.array([1.0,0.0,0.0], dtype=np.float64)
    right = np.cross(tmp, up)
    if np.linalg.norm(right) < 1e-9:
        tmp = np.array([1.0,0.0,0.0], dtype=np.float64)
        right = np.cross(tmp, up)
    right = normalize(right)
    forward = np.cross(up, right)
    forward = normalize(forward)
    return right, forward, up

# ---------------- PSEUDOCODE IMPLEMENTATION: scale_vertices ----------------
def scale_vertices(vertices, s):
    """
    vertices: list of [x,y,z]
    s: uniform scalar
    returns list of scaled vertices [x',y',z'] where x' = s*x etc.
    This implements the pseudocode you provided.
    """
    if not vertices:
        return []
    arr = np.array(vertices, dtype=np.float64)   # shape (N,3)
    scaled = arr * float(s)                     # broadcast multiply
    return scaled.astype(np.float32).tolist()

# ---------------- Normal recompute ----------------
def compute_normals_from_mesh(vertices, indices):
    nv = len(vertices)
    acc = np.zeros((nv, 3), dtype=np.float64)
    v = np.array(vertices, dtype=np.float64)
    for i in range(0, len(indices), 3):
        ia, ib, ic = indices[i:i+3]
        va = v[ia]; vb = v[ib]; vc = v[ic]
        e1 = vb - va; e2 = vc - va
        fn = np.cross(e1, e2)
        acc[ia] += fn; acc[ib] += fn; acc[ic] += fn
    norms = []
    for i in range(nv):
        n = acc[i]
        l = np.linalg.norm(n)
        if l > 1e-8:
            norms.append([float(n[0]/l), float(n[1]/l), float(n[2]/l)])
        else:
            norms.append([0.0, 1.0, 0.0])
    return norms

# ---------------- Recursive tree builder ----------------
class RecursiveTree:
    def __init__(self, cylinder_segments=12, sphere_lat=8, sphere_lon=12):
        self.cylinder_segments = cylinder_segments
        self.sphere_lat = sphere_lat
        self.sphere_lon = sphere_lon

        # unscaled geometry (world-space)
        self.trunk_vertices = []
        self.trunk_indices = []
        self.leaf_vertices = []
        self.leaf_indices = []

        # scaled geometry & normals (these are used for GPU upload)
        self.trunk_vertices_scaled = []
        self.trunk_normals_scaled = []
        self.leaf_vertices_scaled = []
        self.leaf_normals_scaled = []

        # GL buffers
        self.vbo_trunk = None; self.ebo_trunk = None; self.trunk_index_count = 0
        self.vbo_leaf = None;  self.ebo_leaf = None;  self.leaf_index_count = 0

    def clear_geometry(self):
        self.trunk_vertices.clear(); self.trunk_indices.clear()
        self.leaf_vertices.clear(); self.leaf_indices.clear()
        try:
            if self.vbo_trunk: glDeleteBuffers(1, [self.vbo_trunk]); self.vbo_trunk=None
            if self.ebo_trunk: glDeleteBuffers(1, [self.ebo_trunk]); self.ebo_trunk=None
            if self.vbo_leaf:  glDeleteBuffers(1, [self.vbo_leaf]);  self.vbo_leaf=None
            if self.ebo_leaf:  glDeleteBuffers(1, [self.ebo_leaf]);  self.ebo_leaf=None
        except Exception:
            pass

    def generate_cylinder_world(self, origin, axis, height, radius):
        origin = np.array(origin, dtype=np.float64)
        axis = normalize(np.array(axis, dtype=np.float64))
        right, forward, up = orthonormal_basis_from_axis(axis)

        start = len(self.trunk_vertices)
        seg = self.cylinder_segments

        for i in range(seg):
            theta = 2.0 * math.pi * i / seg
            cx = radius * math.cos(theta)
            cz = radius * math.sin(theta)
            offset = right * cx + forward * cz
            bottom = origin + offset
            top = origin + up * height + offset
            self.trunk_vertices.append([float(bottom[0]), float(bottom[1]), float(bottom[2])])
            self.trunk_vertices.append([float(top[0]),    float(top[1]),    float(top[2])])

        for i in range(seg):
            a = start + i*2
            b = start + ((i+1)%seg)*2
            self.trunk_indices.extend([a,b,a+1])
            self.trunk_indices.extend([a+1,b,b+1])

        # caps
        bc_idx = len(self.trunk_vertices)
        self.trunk_vertices.append([float(origin[0]), float(origin[1]), float(origin[2])])
        for i in range(seg):
            rim_i = start + i*2
            rim_next = start + ((i+1)%seg)*2
            self.trunk_indices.extend([bc_idx, rim_next, rim_i])
        top_center = origin + up * height
        tc_idx = len(self.trunk_vertices)
        self.trunk_vertices.append([float(top_center[0]), float(top_center[1]), float(top_center[2])])
        for i in range(seg):
            rim_top_i = start + i*2 + 1
            rim_top_next = start + ((i+1)%seg)*2 + 1
            self.trunk_indices.extend([tc_idx, rim_top_i, rim_top_next])

    def generate_sphere_world(self, center, radius):
        center = np.array(center, dtype=np.float64)
        start = len(self.leaf_vertices)
        latN = self.sphere_lat; lonN = self.sphere_lon
        for lat in range(latN+1):
            phi = math.pi * lat / latN
            for lon in range(lonN+1):
                theta = 2.0*math.pi*lon/lonN
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                p = center + np.array([x,y,z])
                self.leaf_vertices.append([float(p[0]), float(p[1]), float(p[2])])
        for lat in range(latN):
            for lon in range(lonN):
                v1 = start + lat*(lonN+1) + lon
                v2 = v1 + (lonN+1)
                v3 = v1 + 1
                v4 = v2 + 1
                if lat != 0:
                    self.leaf_indices.extend([v1,v2,v3])
                if lat != latN-1:
                    self.leaf_indices.extend([v3,v2,v4])

    def build_tree_recursive(self, origin, axis, height, radius, depth, max_depth):
        if depth > max_depth: return
        self.generate_cylinder_world(origin, axis, height, radius)
        up = normalize(axis)
        tip = origin + up * height

        if depth == max_depth:
            right, forward, upb = orthonormal_basis_from_axis(up)
            leaf_count = random.randint(4, 8)
            for _ in range(leaf_count):
                ox = random.uniform(-0.35, 0.35)
                oy = random.uniform(-0.1, 0.35)
                oz = random.uniform(-0.35, 0.35)
                center = tip + right*ox + upb*oy + forward*oz
                size = 0.35 * (0.8 ** depth) * random.uniform(0.8, 1.2)
                self.generate_sphere_world(center, size)
            return

        branch_count = max(2, 4 - depth) + random.randint(0, 1)
        for i in range(branch_count):
            base_angle = 2.0 * math.pi * i / branch_count
            final_angle = base_angle + random.uniform(-0.4, 0.4)
            tilt_angle = random.uniform(math.radians(20), math.radians(60))
            attach_height = height * random.uniform(0.5, 0.85)
            attach_point = origin + up * attach_height

            bx = math.sin(tilt_angle) * math.cos(final_angle)
            by = math.cos(tilt_angle)
            bz = math.sin(tilt_angle) * math.sin(final_angle)
            branch_axis = normalize([bx, by, bz])

            new_h = height * random.uniform(0.55, 0.75)
            new_r = radius * 0.65
            self.build_tree_recursive(attach_point, branch_axis, new_h, new_r, depth+1, max_depth)

    # ------------------ IMPLEMENTASI YANG DIMINTA ------------------
    def apply_uniform_scale_and_upload(self, scale):
        # 1) Scale vertex positions (pseudocode implementation)
        self.trunk_vertices_scaled = scale_vertices(self.trunk_vertices, scale)
        self.leaf_vertices_scaled = scale_vertices(self.leaf_vertices, scale)

        # 2) Recompute normals from scaled geometry
        self.trunk_normals_scaled = compute_normals_from_mesh(self.trunk_vertices_scaled, self.trunk_indices)
        self.leaf_normals_scaled = compute_normals_from_mesh(self.leaf_vertices_scaled, self.leaf_indices)

        # 3) Upload trunk VBO/EBO (scaled)
        if len(self.trunk_vertices_scaled):
            pos = np.array(self.trunk_vertices_scaled, dtype=np.float32)
            nor = np.array(self.trunk_normals_scaled, dtype=np.float32)
            inter = np.empty((pos.shape[0], 6), dtype=np.float32)
            inter[:,0:3] = pos; inter[:,3:6] = nor
            flat = inter.flatten()
            idxs = np.array(self.trunk_indices, dtype=np.uint32)
            # generate/replace buffers
            if self.vbo_trunk: glDeleteBuffers(1, [self.vbo_trunk])
            if self.ebo_trunk: glDeleteBuffers(1, [self.ebo_trunk])
            self.vbo_trunk = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_trunk)
            glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.ebo_trunk = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_trunk)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.nbytes, idxs, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            self.trunk_index_count = idxs.size

        # 4) Upload leaf VBO/EBO (scaled)
        if len(self.leaf_vertices_scaled):
            pos = np.array(self.leaf_vertices_scaled, dtype=np.float32)
            nor = np.array(self.leaf_normals_scaled, dtype=np.float32)
            inter = np.empty((pos.shape[0], 6), dtype=np.float32)
            inter[:,0:3] = pos; inter[:,3:6] = nor
            flat = inter.flatten()
            idxs = np.array(self.leaf_indices, dtype=np.uint32)
            if self.vbo_leaf: glDeleteBuffers(1, [self.vbo_leaf])
            if self.ebo_leaf: glDeleteBuffers(1, [self.ebo_leaf])
            self.vbo_leaf = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_leaf)
            glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.ebo_leaf = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_leaf)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idxs.nbytes, idxs, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            self.leaf_index_count = idxs.size

    # draw uses the scaled VBOs
    def draw(self):
        if self.vbo_trunk and self.ebo_trunk:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_trunk)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_trunk)
            stride = 6 * 4
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(3*4))
            glColor3f(0.6,0.4,0.2)
            glDrawElements(GL_TRIANGLES, self.trunk_index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        if self.vbo_leaf and self.ebo_leaf:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_leaf)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_leaf)
            stride = 6 * 4
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(3*4))
            glColor3f(0.2,0.8,0.2)
            glDrawElements(GL_TRIANGLES, self.leaf_index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

# ---------------- OpenGL + Main ----------------
def setup_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 5.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2,0.2,0.2,1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0,1.0,0.95,1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.1,0.1,0.1,1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 10.0)
    glClearColor(0.1,0.1,0.15,1.0)
    glEnable(GL_CULL_FACE); glCullFace(GL_BACK); glShadeModel(GL_SMOOTH)

def main():
    pygame.init()
    size = (1200, 800)
    pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Tree: scale_vertices() used in apply_uniform_scale_and_upload()")
    setup_opengl()
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45.0, size[0]/size[1], 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    camera = np.array([8.0,5.0,10.0])
    gluLookAt(camera[0],camera[1],camera[2], 0.0,2.0,0.0, 0.0,1.0,0.0)

    model = RecursiveTree(cylinder_segments=12, sphere_lat=8, sphere_lon=10)

    def build_and_scale(seed=None, scale_override=None):
        if seed is not None: random.seed(seed)
        model.clear_geometry()
        root_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        root_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        model.build_tree_recursive(root_origin, root_axis, height=3.0, radius=0.3, depth=0, max_depth=4)
        s = scale_override if scale_override is not None else random.uniform(0.8, 1.3)
        print("Applying uniform scale s =", s)
        model.apply_uniform_scale_and_upload(s)

    build_and_scale(seed=42)

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running=False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: running=False
                elif ev.key == pygame.K_r:
                    build_and_scale(seed=None, scale_override=None)

        # draw (no per-instance transform)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glMultMatrixf(mat_identity().T.flatten())  # identity; vertices already scaled on CPU
        model.draw()
        glPopMatrix()

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

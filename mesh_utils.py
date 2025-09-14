# mesh_utils.py
def make_box_mesh(sx, sz, height):
    """Return (vertices, indices) for a box centered on XZ origin, base at y=0, top at y=height.
    vertices: list of (x,y,z)
    indices: flat list of ints, 3 per triangle."""
    hx = sx/2.0
    hz = sz/2.0
    y0 = 0.0
    y1 = float(height)
    verts = [
        (-hx, y0, -hz), ( hx, y0, -hz), ( hx, y1, -hz), (-hx, y1, -hz),
        (-hx, y0,  hz), ( hx, y0,  hz), ( hx, y1,  hz), (-hx, y1,  hz)
    ]
    inds = []
    # back
    inds += [0,1,2, 0,2,3]
    # front
    inds += [4,6,5, 4,7,6]
    # left
    inds += [4,0,3, 4,3,7]
    # right
    inds += [1,5,6, 1,6,2]
    # top
    inds += [3,2,6, 3,6,7]
    # bottom
    inds += [4,5,1, 4,1,0]
    return verts, inds

def extrude_path_2d(path2d, height):
    """Extrude a 2D polygon vertically. Returns (vertices, indices). path2d is list of (x,z)."""
    verts = []
    for x,z in path2d:
        verts.append((x, 0.0, z))
    for x,z in path2d:
        verts.append((x, height, z))
    n = len(path2d)
    inds = []
    # sides
    for i in range(n):
        a = i
        b = (i+1) % n
        inds += [a, b, n+b, a, n+b, n+a]
    # top fan
    center_idx = len(verts)
    cx = sum(p[0] for p in path2d)/n
    cz = sum(p[1] for p in path2d)/n
    verts.append((cx, height, cz))
    for i in range(n):
        inds += [n+i, n+((i+1)%n), center_idx]
    # bottom fan
    center_idx2 = len(verts)
    verts.append((cx, 0.0, cz))
    for i in range(n):
        inds += [i, (i+1)%n, center_idx2]
    return verts, inds

# algorithms.py
import math

def dda_line(x0, y0, x1, y1):
    """Digital Differential Analyzer line algorithm.
    Returns list of integer (x,y) pixels along the line."""
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))
    if steps == 0:
        return [(int(round(x0)), int(round(y0)))]
    x_inc = dx / steps
    y_inc = dy / steps
    x, y = x0, y0
    pts = []
    for _ in range(steps + 1):
        pts.append((int(round(x)), int(round(y))))
        x += x_inc
        y += y_inc
    return pts

def bresenham_line(x0, y0, x1, y1):
    """Integer Bresenham line algorithm. Returns list of (x,y)."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    pts = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts

def midpoint_circle_offsets(r):
    if r <= 0:
        return [(0,0)]
    pts_set = set()
    x = r
    y = 0
    p = 1 - r
    while x >= y:
        pts_set.update({
            ( x,  y), (-x,  y), ( x, -y), (-x, -y),
            ( y,  x), (-y,  x), ( y, -x), (-y, -x)
        })
        y += 1
        if p <= 0:
            p = p + 2*y + 1
        else:
            x -= 1
            p = p + 2*(y - x) + 1

    # sort by angle to make a continuous loop
    pts = list(pts_set)
    pts.sort(key=lambda t: math.atan2(t[1], t[0]))
    return pts

def scanfill_polygon(points):
    """Scanline fill polygon sederhana."""
    if not points:
        return []

    # Cari bounding box
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    filled = []

    for y in range(min_y, max_y + 1):
        # cari semua titik potong dengan scanline
        intersections = []
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i+1) % len(points)]
            if y1 == y2:
                continue
            if y1 <= y < y2 or y2 <= y < y1:
                x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                intersections.append(int(round(x_int)))
        intersections.sort()
        for i in range(0, len(intersections), 2):
            if i+1 < len(intersections):
                filled.append((intersections[i], intersections[i+1], y))
    return filled


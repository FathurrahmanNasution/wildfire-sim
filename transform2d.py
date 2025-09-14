# transform2d.py
import math

def translate(points, tx, ty):
    return [(x + tx, y + ty) for (x, y) in points]

def rotate(points, angle_deg, origin=(0,0)):
    angle = math.radians(angle_deg)
    ox, oy = origin
    cos_a = math.cos(angle); sin_a = math.sin(angle)
    out = []
    for x, y in points:
        x -= ox; y -= oy
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        out.append((xr + ox, yr + oy))
    return out

def scale(points, sx, sy=None, origin=(0,0)):
    if sy is None: sy = sx
    ox, oy = origin
    return [((x-ox) * sx + ox, (y-oy) * sy + oy) for x, y in points]

def shear(points, shx=0.0, shy=0.0):
    return [(x + shx*y, shy*x + y) for (x,y) in points]

def morph(points_a, points_b, t):
    """Linear interpolation between two lists of equal length."""
    if len(points_a) != len(points_b):
        raise ValueError("morph requires equal-length point lists")
    out = []
    for a, b in zip(points_a, points_b):
        x = a[0] * (1-t) + b[0] * t
        y = a[1] * (1-t) + b[1] * t
        out.append((x, y))
    return out

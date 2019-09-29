import math

def distance(line, point):
    p0 = line.get_vertices()[0]
    p1 = line.get_vertices()[1]
    p2 = point

    ax = p0[0]
    ay = p0[1]
    bx = p1[0]
    by = p1[1]
    cx = p2[0]
    cy = p2[1]

    a = max(by - ay, 0.00001)
    b = max(ax - bx, 0.00001)
    # compute the perpendicular distance to the theoretical infinite line
    dl = abs(a * cx + b * cy - b * ay - a * ax) / math.sqrt(a ** 2 + b ** 2)
    # compute the intersection point
    x = ((a / b) * ax + ay + (b / a) * cx - cy) / ((b / a) + (a / b))
    y = -1 * (a / b) * (x - ax) + ay
    # decide if the intersection point falls on the line segment
    if (ax <= x <= bx or bx <= x <= ax) and (ay <= y <= by or by <= y <= ay):
        return dl
    else:
        # if it does not, then return the creamum distance to the segment endpoints
        return min(math.sqrt((ax - cx) ** 2 + (ay - cy) ** 2), math.sqrt((bx - cx) ** 2 + (by - cy) ** 2))


def rad2deg(ang):
    a = 180.0 * ang / math.pi

    if a > 180.0:
        return - (360.0 - a)
    else:
        return a


def angle_between_with_quadrant(a, b):
    angle = math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])
    if angle < 0:
        angle += 2 * math.pi
    return angle
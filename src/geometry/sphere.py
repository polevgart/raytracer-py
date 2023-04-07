import attr
import math
from .vector import Vector
from . import base


def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                raise RuntimeError("INF solves")
            else:
                return None
        else:
            x = -c / b
            return (x, x)
    else:
        d = b * b - 4 * a * c
        if d < 0:
            return None
        sqrt_d = math.sqrt(d)
        x1 = (-b - sqrt_d) / (2.0 * a)
        x2 = (-b + sqrt_d) / (2.0 * a)
    return (x1, x2)



@attr.s(slots=True, kw_only=True)
class Sphere(base.BaseObject):
    center: Vector = attr.ib()
    radius: float = attr.ib(converter=float)
    
    def intersect(self, ray: base.Ray) -> base.Intersection | None:
        dpos = ray.origin - self.center
        dir = ray.direction
        a = dir.dot(dir)
        b = 2 * dpos.dot(dir)
        c = dpos.dot(dpos) - self.radius ** 2
        roots = solve_quadratic(a, b, c)
        if not roots:
            return None

        distance = roots[0] if 0 < roots[0] < roots[1] else roots[1]
        if distance < 0:
            return None
        
        point = ray.origin + dir * distance
        return base.Intersection(point, self.get_normal(point), distance)

    def get_normal(self, point: Vector) -> Vector:
        a = point - self.center
        a /= self.radius
        return a
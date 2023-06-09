import attr
import math

from .base import BaseObject, Intersection, Ray
from .vector import Vector


def solve_quadratic(a, b, c) -> tuple[float, float] | None:
    if a == 0:
        if b == 0:
            if c == 0:
                raise RuntimeError("INF solutions")
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
        if a < 0:
            x1, x2 = x2, x1
    return (x1, x2)


@attr.s(slots=True, kw_only=True)
class Sphere(BaseObject):
    center: Vector = attr.ib()
    radius: float = attr.ib(converter=float)

    def intersect(self, ray: Ray) -> Intersection | None:
        origin = ray.origin
        dir = ray.direction

        dpos = origin - self.center
        a = dir.dot(dir)
        b = 2 * dpos.dot(dir)
        c = dpos.dot(dpos) - self.radius ** 2
        roots = solve_quadratic(a, b, c)
        if roots is None:
            return None

        distance = roots[0] if roots[0] > 0 else roots[1]
        if distance < 0:
            return None

        point = origin + distance * dir
        normal = self.get_normal(point)
        if c < 0:
            normal *= -1
        return Intersection(point, normal, distance)

    def get_normal(self, point: Vector) -> Vector:
        normal = point - self.center
        normal /= self.radius
        return normal

    def has_volume(self) -> bool:
        return True

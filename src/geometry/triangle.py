import attr

from .vector import Vector
from .base import BaseObject, Intersection, Ray


EPS = 1e-8


def get_triangle_area(v1: Vector, v2: Vector, v3: Vector) -> float:
    left_side = v2 - v1
    right_side = v3 - v1
    return left_side.cross(right_side).length / 2


@attr.s(slots=True)
class Triangle(BaseObject):
    _vertices: tuple[Vector] = attr.ib(converter=tuple)

    @property
    def area(self) -> float:
        return get_triangle_area(*self._vertices)

    def __getitem__(self, idx: int) -> Vector:
        return self._vertices[idx]

    def intersect(self, ray: Ray) -> Intersection | None:
        left_side = self._vertices[1] - self._vertices[0]
        right_side = self._vertices[2] - self._vertices[0]
        height = ray.direction.cross(right_side)
        det = left_side.dot(height)
        if abs(det) < EPS:
            return None

        inv_det = 1 / det
        vertex2origin = ray.origin - self._vertices[0]
        first_ratio = inv_det * vertex2origin.dot(height)
        if not 0 <= first_ratio <= 1:
            return None

        outer = vertex2origin.cross(left_side)
        second_ratio = inv_det * ray.direction.dot(outer)
        if not 0 <= second_ratio <= 1 - first_ratio:
            return None

        dist = inv_det * right_side.dot(outer)
        if dist < 0:
            return None

        pos = ray.origin + dist * ray.direction
        norm = left_side.cross(right_side)
        if ray.direction.dot(norm) > 0:
            norm *= -1
        norm.normalize()
        return Intersection(pos, norm, dist)

    def get_barycentric_coords(self, point: Vector) -> Vector:
        area = self.area
        return Vector(
            get_triangle_area(self._vertices[1], self._vertices[2], point) / area,
            get_triangle_area(self._vertices[2], self._vertices[0], point) / area,
            get_triangle_area(self._vertices[0], self._vertices[1], point) / area,
        )

    def has_volume(self) -> bool:
        return False
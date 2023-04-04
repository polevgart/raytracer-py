import attr

from .ray import Ray
from .vector import Vector


class Color(Vector):
    def to_tuple(self):
        resp = super().to_tuple()
        return tuple(map(int, resp))


@attr.s(slots=True, kw_only=True)
class Material:
    color: Color = attr.ib()


class Texture:
    pass


@attr.s(slots=True)
class Intersection:
    position: Vector = attr.ib()
    normal: Vector = attr.ib()
    distance: float = attr.ib()


@attr.s(slots=True, kw_only=True)
class BaseObject:
    material: Material = attr.ib(default=None)
    texture: Texture = attr.ib(default=None)

    def intersect(self, ray: Ray) -> Intersection | None:
        raise NotImplementedError()
    
    def get_normal(self, pos: Vector) -> Vector:
        raise NotImplementedError()

    def get_color(self, pos: Vector) -> Vector:
        raise NotImplementedError()
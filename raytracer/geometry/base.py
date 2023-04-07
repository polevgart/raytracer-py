import attr
import numpy as np

from .ray import Ray
from .vector import Vector


@attr.s(slots=True, kw_only=True)
class Material:
    ambient_color: Vector = attr.ib(factory=Vector)
    diffuse_color: Vector = attr.ib(factory=Vector)
    specular_color: Vector = attr.ib(factory=Vector)
    specular_exponent: float = attr.ib(default=0)
    refraction_index: float = attr.ib(default=1)
    albedo: Vector = attr.ib()

    @albedo.default
    def _(self) -> Vector:
        return Vector(1, 0, 0)


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

    def has_volume(self) -> bool:
        raise NotImplementedError()

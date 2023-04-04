import attr

from .vector import Vector

import attr
import math

from .vector import Vector


@attr.s(slots=True, kw_only=True)
class Ray:
    origin: Vector = attr.ib()
    direction: Vector = attr.ib(converter=lambda x: (x.normalize(), x)[1])


def reflect(direction: Vector, normal: Vector) -> Vector:
    cos_incidence = -normal.dot(direction)
    return direction + 2 * cos_incidence * normal

def refract(direction: Vector, normal: Vector, eta: float) -> Vector | None:
    cos_incidence = -normal.dot(direction)
    beta = 1 - eta ** 2 * (1 - cos_incidence ** 2)
    if beta < 0:
        return None
    return eta * direction + (eta * cos_incidence - math.sqrt(beta)) * normal
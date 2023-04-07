import attr

from .vector import Vector


@attr.s(slots=True, kw_only=True)
class Ray:
    origin: Vector = attr.ib()
    direction: Vector = attr.ib(converter=lambda x: (x.normalize(), x)[1])
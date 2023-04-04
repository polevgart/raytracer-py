import attr
from .vector import Vector
from .base import BaseObject


@attr.s(slots=True)
class Triangle(BaseObject):
    _vertices: tuple[Vector] = attr.ib(converter=tuple)

    @property
    def area(self):
        pass
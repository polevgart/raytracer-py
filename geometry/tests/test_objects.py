import attr

from .. import *


class TestSphere:
    def test_init(self):
        assert Sphere(center=Vector(1, 2, 3), radius=5).radius == 5


class TestTriangle:
    def test_init(self):
        vertices = [Vector(1, 2, 3), Vector(2, 3, 4)]
        assert Triangle(vertices)
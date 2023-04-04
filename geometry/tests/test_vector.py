import math
import numpy as np

from ..ray import reflect, refract
from ..vector import Vector


class TestVector:
    def test_length(self):
        assert Vector(0, 1, 0).length == 1.0
        assert Vector(3, 4, 0).length == 5.0
        assert Vector(1, 1, 1).length - math.sqrt(3) < 0.00000001

    def test_normalize(self):
        v = Vector(3, 4, 0)
        v.normalize()
        assert v.length == 1.0
        assert str(v) == "Vector(0.60, 0.80, 0.00)"

    def test_str(self):
        assert str(Vector(0, 0, 0)) == "Vector(0.00, 0.00, 0.00)"
        assert str(Vector(1, 2, 3.14)) == "Vector(1.00, 2.00, 3.14)"

    def test_binary_operations(self):
        a = Vector(1, 2, 3)
        b = Vector(1, 2, 3)
        assert a + b == Vector(2, 4, 6)
        assert a - b == Vector(0, 0, 0)
        assert a * 3 == Vector(3, 6, 9)

        assert a / 2. == Vector(0.5, 1, 1.5)
        a /= 2.
        assert a == Vector(0.5, 1, 1.5)

    def test_to_tuple(self):
        t = Vector(1, 2, 3).to_tuple()
        assert isinstance(t, tuple)
        assert isinstance(t[0], float)
        assert t == (1., 2., 3.)

    def test_reflect(self):
        normal = Vector(0, 1, 0)
        ray = Vector(0.707107, -0.707107, 0)
        reflected = reflect(ray, normal)
        assert np.allclose(reflected.x, 0.707107)
        assert np.allclose(reflected.y, 0.707107)

    def test_refract(self):
        normal = Vector(0, 1, 0)
        ray = Vector(0.707107, -0.707107, 0)
        refracted = refract(ray, normal, 0.9)
        assert np.allclose(refracted.x, 0.636396)
        assert np.allclose(refracted.y, -0.771362)

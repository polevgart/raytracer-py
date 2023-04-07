import math
import numpy as np

from ..ray import reflect, refract
from ..vector import Vector


class TestVector:
    def test_length(self):
        assert Vector(0, 1, 0).length == 1.0
        assert Vector(3, 4, 0).length == 5.0
        assert np.allclose(Vector(1, 1, 1).length, math.sqrt(3))

    def test_normalize(self):
        v = Vector(3, 4, 0)
        v.normalize()
        assert v.length == 1.0
        assert str(v) == "Vector(0.60, 0.80, 0.00)"

    def test_str(self):
        assert str(Vector(0, 0, 0)) == "Vector(0.00, 0.00, 0.00)"
        assert str(Vector(1, 2, 3.14)) == "Vector(1.00, 2.00, 3.14)"

    def test_dot_product(self):
        assert np.allclose(Vector(47, 1, 0).dot(Vector(1, -47, 0)), 0)
        assert np.allclose(Vector(0, 47, -1).dot(Vector(0, 1, 47)), 0)
        assert np.allclose(Vector(47, 42, 22).dot(Vector(1, 0, 0)), 47)
        assert np.allclose(Vector(47, 42, 22).dot(Vector(0, 1, 0)), 42)
        assert np.allclose(Vector(47, 42, 22).dot(Vector(0, 0, 1)), 22)

    def test_cross_product(self):
        assert Vector(47, 0, 0).cross(Vector(0, 42, 0)) == Vector(0, 0, 42 * 47)
        assert Vector(0, 47, 0).cross(Vector(0, 0, 42)) == Vector(42 * 47, 0, 0)
        assert Vector(0, 0, 47).cross(Vector(42, 0, 0)) == Vector(0, 42 * 47, 0)
        assert Vector(1, 2, 3).cross(Vector(4, 5, 6)) == Vector(-3, 6, -3)

    def test_binary_operations(self):
        a = Vector(1, 2, 3)
        b = Vector(1, 2, 3)
        assert a + b == Vector(2, 4, 6) == b + a
        assert a - b == Vector(0, 0, 0) == -(b - a)
        assert a * 3 == Vector(3, 6, 9) == 3 * a == a + 2 * a == 5 * a - 2 * a

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

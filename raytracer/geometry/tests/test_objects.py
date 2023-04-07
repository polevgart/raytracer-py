import attr
import numpy as np

from .. import *


class TestSphere:
    def test_init(self):
        assert Sphere(center=Vector(1, 2, 3), radius=5).radius == 5

    def test_intersect(self):
        sphere = Sphere(center=Vector(0, 0, 0), radius=2)
        ray = Ray(origin=Vector(5, 0, 2.2), direction=Vector(-1, 0, 0))
        intersection = sphere.intersect(ray)
        assert intersection is None

        ray = Ray(origin=Vector(5, 0, 0), direction=Vector(-1, 0, 0))
        intersection = sphere.intersect(ray)
        assert intersection is not None
        assert np.allclose(intersection.position.x, 2)
        assert np.allclose(intersection.normal.x, 1)
        assert np.allclose(intersection.distance, 3)

        ray = Ray(origin=Vector(5, 0, 2), direction=Vector(-1, 0, 0))
        intersection = sphere.intersect(ray)
        assert intersection is not None
        assert np.allclose(intersection.position.z, 2)
        assert np.allclose(intersection.normal.z, 1)
        assert np.allclose(intersection.distance, 5)

        ray = Ray(origin=Vector(0, 0, 0), direction=Vector(-1, 0, 0))
        intersection = sphere.intersect(ray)
        assert intersection is not None
        assert np.allclose(intersection.position.x, -2)
        assert np.allclose(intersection.normal.x, 1)
        assert np.allclose(intersection.distance, 2)


class TestTriangle:
    def test_init(self):
        vertices = [Vector(0, 2, 1 + 47), Vector(0, 2, 1), Vector(42, 2, 1)]
        assert np.allclose(Triangle(vertices).area, 0.5 * 47 * 42)

    def test_intersect(self):
        triangle = Triangle([Vector(0, 0, 0), Vector(4, 0, 0), Vector(0, 4, 0)])
        ray = Ray(origin=Vector(3, 3, 1), direction=Vector(-1, -1, 0))
        intersection = triangle.intersect(ray)
        assert intersection is None

        ray = Ray(origin=Vector(2, 2, 1), direction=Vector(0, 0, -1))
        intersection = triangle.intersect(ray)
        assert intersection is not None
        assert np.allclose(intersection.position.x, 2)
        assert np.allclose(intersection.position.y, 2)
        assert np.allclose(intersection.normal.z, 1)
        assert np.allclose(intersection.distance, 1)

    def test_barycentric_coords(self):
        triangle = Triangle([
            Vector(0, 0, 0),
            Vector(2, 0, 0),
            Vector(0, 2, 0),
        ])
        on_edge = triangle.get_barycentric_coords(Vector(1, 1, 0))
        assert np.allclose(on_edge.y, 0.5)
        assert np.allclose(on_edge.z, 0.5)

        on_vertex = triangle.get_barycentric_coords(Vector(2, 0, 0))
        assert np.allclose(on_vertex.y, 1)

        inside = triangle.get_barycentric_coords(Vector(0.2, 0.2, 0))
        assert np.allclose(inside.x, 0.8)
        assert np.allclose(inside.y, 0.1)
        assert np.allclose(inside.z, 0.1)

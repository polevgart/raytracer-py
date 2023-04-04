import numpy as np

from ..geometry import Vector


EPS = 1e-8


class Matrix:
    def __init__(self, right: Vector, up: Vector, forward: Vector, from_: Vector):
        self._data = np.zeros((4, 4), dtype=float)
        self._data[0, :3] = right.to_array()
        self._data[1, :3] = up.to_array()
        self._data[2, :3] = forward.to_array()
        self._data[3, :3] = from_.to_array()
        self._data[3, 3] = 1

    def __getitem__(self, slice):
        return self._data[slice]


def look_at(look_from: Vector, look_to: Vector) -> Matrix:
    forward = look_from - look_to
    forward.normalize()

    up = Vector(0, 1, 0)
    right = up.cross(forward)
    if right.length < EPS:
        right = Vector(0, 0, 1).cross(forward)
        if up.dot(forward) > 0:
            right *= -1
    right.normalize()

    up = forward.cross(right)
    return Matrix(right, up, forward, look_from)


def vector_matrix_multiply(matrix: Matrix, vector: Vector) -> Vector:
    return Vector(*(matrix[:3, :3].T @ vector.to_array()))


def point_matrix_multiply(matrix: Matrix, point: Vector) -> Vector:
    result = vector_matrix_multiply(matrix, point)
    result += Vector(*matrix[3, :3])
    depth = point.to_array() @ matrix[3, :3] + matrix[3, 3]
    return result / depth

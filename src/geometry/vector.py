import copy
import math
import numpy as np


class Vector:
    def __init__(self, x, y, z):
        self._data = np.array([x, y, z], dtype=np.float64)

    def __repr__(self):
        cords_repr = ", ".join("{:.2f}".format(cord) for cord in self._data)
        return f"Vector({cords_repr})"

    @property
    def length(self) -> float:
        return math.sqrt(np.dot(self._data, self._data))

    def normalize(self):
        self._data /= self.length

    def dot(self, other: "Vector") -> float:
        return self._data @ other._data
    
    def cross(self, other: "Vector") -> "Vector":
        return Vector(*np.cross(self._data, other._data))

    def __add__(self, other):
        return Vector(*(self._data + other._data))

    def __iadd__(self, other):
        self._data += other._data
        return self

    def __sub__(self, other):
        return Vector(*(self._data - other._data))

    def __isub__(self, other):
        self._data -= other._data
        return self

    def __mul__(self, other):
        return Vector(*(self._data * other))
    
    def __imul__(self, other):
        self._data *= other
        return self

    def __truediv__(self, other: float):
        return Vector(*(self._data / other))
    
    def __itruediv__(self, other):
        self._data /= float(other)
        return self

    def to_tuple(self):
        return tuple(self._data)

    def __eq__(self, other):
        return np.allclose(self._data, other._data)
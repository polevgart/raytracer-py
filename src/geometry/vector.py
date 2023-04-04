import copy
import math
import numpy as np


class Vector:
    def __init__(self, x=0, y=None, z=None, /):
        if y is None:
            y = z = x
        elif z is None:
            raise ValueError("You should specify either 1 or all 3 vector coordinates")
        self._data = np.array([x, y, z], dtype=float)

    def __repr__(self) -> str:
        coords_repr = ", ".join("{:.2f}".format(coord) for coord in self._data)
        return f"Vector({coords_repr})"

    @property
    def x(self) -> float:
        return self._data[0]

    @property
    def y(self) -> float:
        return self._data[1]

    @property
    def z(self) -> float:
        return self._data[2]

    def __getitem__(self, idx: int) -> float:
        return self._data[idx]

    @property
    def length(self) -> float:
        return math.sqrt(np.dot(self._data, self._data))

    def normalize(self) -> "Vector":
        self._data /= self.length
        return self

    def dot(self, other: "Vector") -> float:
        return self._data @ other._data

    def cross(self, other: "Vector") -> "Vector":
        return Vector(*np.cross(self._data, other._data))

    def __add__(self, other) -> "Vector":
        return Vector(*(self._data + other._data))

    def __iadd__(self, other) -> "Vector":
        self._data += other._data
        return self

    def __sub__(self, other) -> "Vector":
        return Vector(*(self._data - other._data))

    def __isub__(self, other) -> "Vector":
        self._data -= other._data
        return self

    def __neg__(self) -> "Vector":
        return Vector(*(-self._data))

    def __mul__(self, other) -> "Vector":
        if isinstance(other, Vector):
            return Vector(*(self._data * other._data))
        return Vector(*(self._data * other))

    def __rmul__(self, other) -> "Vector":
        if isinstance(other, Vector):
            return Vector(*(self._data * other._data))
        return Vector(*(self._data * other))

    def __imul__(self, other) -> "Vector":
        self._data *= other
        return self

    def __truediv__(self, other: float) -> "Vector":
        return Vector(*(self._data / other))

    def __itruediv__(self, other) -> "Vector":
        self._data /= float(other)
        return self

    def to_tuple(self) -> tuple[float]:
        return tuple(self._data)

    def to_array(self):
        return self._data

    def __eq__(self, other) -> bool:
        return np.allclose(self._data, other._data)

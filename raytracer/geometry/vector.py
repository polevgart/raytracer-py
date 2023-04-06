import copy
import math
import numpy as np
import operator
from typing import Any, Callable


BinaryOp = Callable[[float, float], float]


def zip_tuples(op: BinaryOp, lhs: tuple[float], rhs: tuple[float]) -> tuple[float]:
    return tuple(map(op, lhs, rhs))


class Vector:
    def __init__(self, x: float = 0, y: float | None = None, z: float | None = None, /):
        if y is None:
            y = z = x
        elif z is None:
            raise ValueError("You should specify either 1 or all 3 vector coordinates")
        self._data = (x, y, z)

    @classmethod
    def from_array(cls, arr) -> "Vector":
        return cls(*arr)

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
        return math.sqrt(self.dot(self))

    def normalize(self) -> "Vector":
        length = self.length
        self._data = tuple(coord / length for coord in self._data)
        return self

    def dot(self, other: "Vector") -> float:
        return sum(zip_tuples(operator.mul, self._data, other._data))

    def cross(self, other: "Vector") -> "Vector":
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(*zip_tuples(operator.add, self._data, other._data))

    def __iadd__(self, other: "Vector") -> "Vector":
        self._data = zip_tuples(operator.add, self._data, other._data)
        return self

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(*zip_tuples(operator.sub, self._data, other._data))

    def __isub__(self, other: "Vector") -> "Vector":
        self._data = zip_tuples(operator.sub, self._data, other._data)
        return self

    def __neg__(self) -> "Vector":
        return Vector(*(-coord for coord in self._data))

    def __mul__(self, other: float) -> "Vector":
        return Vector(*(coord * other for coord in self._data))

    def hadamard(self, other: "Vector") -> "Vector":
        return Vector(*zip_tuples(operator.mul, self._data, other._data))

    def __rmul__(self, other: float) -> "Vector":
        return Vector(*(coord * other for coord in self._data))

    def __imul__(self, other: float) -> "Vector":
        self._data = tuple(coord * other for coord in self._data)
        return self

    def __truediv__(self, other: float) -> "Vector":
        return Vector(*(coord / other for coord in self._data))

    def __itruediv__(self, other: float) -> "Vector":
        self._data = tuple(coord / other for coord in self._data)
        return self

    def to_tuple(self) -> tuple[float]:
        return tuple(float(coord) for coord in self._data)

    def to_array(self):
        return np.array(self._data, dtype=float)

    def __eq__(self, other: "Vector") -> bool:
        return np.allclose(self._data, other._data)

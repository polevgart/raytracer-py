import copy
import math
import numpy as np

from typing import Any


class Vector:
    def __init__(
        self,
        x: float = 0,
        y: float | None = None,
        z: float | None = None,
        /,
        empty: bool = False,
    ):
        if not empty:
            if y is None:
                y = z = x
            elif z is None:
                raise ValueError("You should specify either 1 or all 3 vector coordinates")
            self._data = np.array([x, y, z], dtype=float)

    @classmethod
    def from_array(cls, arr) -> "Vector":
        vec = cls(empty=True)
        vec._data = arr.copy()
        return vec

    def clone(self) -> "Vector":
        return Vector.from_array(self._data)

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
        return math.sqrt(self._data @ self._data)

    def normalize(self) -> "Vector":
        self._data /= self.length
        return self

    def dot(self, other: "Vector") -> float:
        return self._data @ other._data

    def cross(self, other: "Vector") -> "Vector":
        return Vector.from_array(np.cross(self._data, other._data))

    def __add__(self, other: "Vector") -> "Vector":
        result = self.clone()
        result += other
        return result

    def __iadd__(self, other: "Vector") -> "Vector":
        self._data += other._data
        return self

    def __sub__(self, other: "Vector") -> "Vector":
        result = self.clone()
        result -= other
        return result

    def __isub__(self, other: "Vector") -> "Vector":
        self._data -= other._data
        return self

    def __neg__(self) -> "Vector":
        result = self.clone()
        result._data *= -1
        return result

    def __mul__(self, other: float) -> "Vector":
        result = self.clone()
        result._data *= other
        return result

    def hadamard(self, other: "Vector") -> "Vector":
        result = self.clone()
        result._data *= other._data
        return result

    def __rmul__(self, other: float) -> "Vector":
        result = self.clone()
        result._data *= other
        return result

    def __imul__(self, other: float) -> "Vector":
        self._data *= other
        return self

    def __truediv__(self, other: float) -> "Vector":
        result = self.clone()
        result._data /= other
        return result

    def __itruediv__(self, other: float) -> "Vector":
        self._data /= other
        return self

    def to_tuple(self) -> tuple[float]:
        return tuple(self._data)

    def to_array(self):
        return self._data.copy()

    def __eq__(self, other: "Vector") -> bool:
        return np.allclose(self._data, other._data)

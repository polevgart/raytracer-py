import math
import numpy as np
import operator
from typing import Callable


BinaryOp = Callable[[float, float], float]


def zip_tuples(op: BinaryOp, lhs: tuple[float], rhs: tuple[float]) -> tuple[float, ...]:
    return tuple(map(op, lhs, rhs))


class Vector:
    def __init__(self, x: float = 0, y: float | None = None, z: float | None = None, /):
        if y is None:
            y = z = x
        
        assert z is not None, "You should specify either 1 or all 3 vector coordinates"
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @classmethod
    def from_array(cls, arr) -> "Vector":
        return cls(*arr)

    def __repr__(self) -> str:
        return f"Vector({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        elif idx == 2:
            return self.z
        raise IndexError(f"index {idx} not found")

    @property
    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> "Vector":
        length = self.length
        self /= length
        return self

    def dot(self, other: "Vector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector") -> "Vector":
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __iadd__(self, other: "Vector") -> "Vector":
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __isub__(self, other: "Vector") -> "Vector":
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y, -self.z)

    def __mul__(self, other: float) -> "Vector":
        return Vector(self.x * other, self.y * other, self.z * other)

    def hadamard(self, other: "Vector") -> "Vector":
        return Vector(self.x * other.x, self.y * other.y, self.z * other.z)

    def __rmul__(self, other: float) -> "Vector":
        return Vector(self.x * other, self.y * other, self.z * other)

    def __imul__(self, other: float) -> "Vector":
        self.x *= other
        self.y *= other
        self.z *= other
        return self

    def __truediv__(self, other: float) -> "Vector":
        return Vector(self.x / other, self.y / other, self.z / other)

    def __itruediv__(self, other: float) -> "Vector":
        self.x /= other
        self.y /= other
        self.z /= other
        return self

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_array(self):
        return np.array(self.to_tuple(), dtype=float)

    def __eq__(self, other: "Vector") -> bool:
        return np.allclose(self.to_tuple(), other.to_tuple())

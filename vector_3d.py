import numpy as np
from transformations.arrow_3d import Arrow3D


class Vector3D:

    def __init__(self, x, y, z, origin=None, text=None, color='k'):
        self.x = x
        self.y = y
        self.z = z
        self.v = self.components()
        self.origin = origin
        self.color = color
        self.text = text
        if self.origin is not None:
            self.arrow = self.arrow_3d()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'3DVector(x={round(self.x, 3)}, y={round(self.y, 3)}, z={round(self.z, 3)})'

    # subtract two Vector3D objects
    def __sub__(self, other):
        rx, ry, rz = np.subtract(self.v, other.v)
        return Vector3D(rx, ry, rz, origin=self.origin, text=self.text, color=self.color)

    # sum two Vector3D objects
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z, origin=self.origin, text=self.text, color=self.color)

    # vector components
    def components(self):
        return np.array([self.x, self.y, self.z])

    # vector magnitude
    def magnitude(self):
        return np.linalg.norm(self.v)

    # cross product with another 3DVector object
    def cross(self, other):
        v = np.cross(self.v, other.v)
        return Vector3D(v[0], v[1], v[2], origin=self.origin, text=self.text, color=self.color)

    # dot product with another 3DVector object
    def dot(self, other):
        return np.dot(self.v, other.v)

    # get unit vector
    def unit(self):
        rx, ry, rz = self.v / self.magnitude()
        return Vector3D(rx, ry, rz, origin=self.origin, text=self.text, color=self.color)

    # scale self
    def scale(self, c):
        self.x = self.x * c
        self.y = self.y * c
        self.z = self.z * c
        return self

    def arrow_3d(self):
        return Arrow3D([self.origin.x, self.x],
                       [self.origin.y, self.y],
                       [self.origin.z, self.z],
                       mutation_scale=10,
                       lw=1,
                       arrowstyle='-|>',
                       color=self.color)

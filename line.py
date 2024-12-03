from scipy.spatial.transform import Rotation
import numpy as np


class Line:
    """
    norm_vec is along line
    """

    def __init__(self, pt0=None, pt1=None, norm_vec=None):
        if norm_vec is None:
            self._norm_vec = np.array(pt1, dtype=float) - np.array(pt0, dtype=float)
        else:
            self._norm_vec = np.array(norm_vec, dtype=float)
        self._norm_vec /= np.linalg.norm(self._norm_vec)

        self._pt = np.array(pt0, dtype=float)

    def perpendicular_vector(self):
        # Rotate norm vec by 90 degrees
        return (Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()[:2, :2] @ np.vstack(self._norm_vec)).flatten()

    def dist(self, pt):
        PQ = np.array(pt, dtype=float) - self._pt
        return np.dot(PQ, self.perpendicular_vector())

    def dist_from_meshgrid(self, R, C):
        CR = np.array([C, R])
        CR = np.moveaxis(CR, 0, 2)
        return self.dist(CR)        

    def plot(self, length, ax):
        pt2 = self._pt + length * self._norm_vec
        ax.plot([self._pt[0], pt2[0]], [self._pt[1], pt2[1]])

    def y_from_x(self, x):
        v = self.perpendicular_vector()
        c = -(v[0] * self._pt[0] + v[1] * self._pt[1])
        return (-c - v[0] * x) / v[1]
    
    def x_from_y(self, y):
        v = self.perpendicular_vector()
        c = -(v[0] * self._pt[0] + v[1] * self._pt[1])
        return (-c - v[1] * y) / v[0]
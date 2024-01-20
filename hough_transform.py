import open3d as o3d
import  math
import numpy as np

def estimate_orientation(pc: o3d.geometry.PointCloud):
    # Uses KNN to find normals of each point in pc.
    pc.estimate_normals()

"""
normal represents input of normal of a point.
@:return spherical co-ordinates equivalent.
"""
def get_spherical_coords(normal: np.ndarray):
    x, y, z = normal
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    phi = math.atan2(z, math.sqrt(x**2 + y**2))

    return np.array([r, theta, phi])

"""
Computes rotation matrix from z-axis based on normal in spherical co-ordinates.
"""
def rotation_matrix(normal: np.ndarray):
    # Create e3 basis vector.
    z = np.zeros(3)
    z[2] = 1

    # Compute normalised difference from e3 of normal.
    diff = z - normal
    diff_norm = np.linalg.norm(diff)
    diff = (1 / diff_norm) * diff

    identity = np.eye(3, 3)
    R = identity - 2 * np.dot(diff, diff) * identity

    return R





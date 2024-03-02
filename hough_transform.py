import open3d as o3d
import math
import numpy as np
from sklearn import neighbors
from sklearn.neighbors._kd_tree import KDTree


def estimate_orientation(pc: o3d.geometry.PointCloud):
    # Uses KNN to find normals of each point in pc.
    normals = np.asarray(pc.estimate_normals())
    points = np.asarray(pc.points)

    # Initialise Hough Gaussian Sphere.
    hg_sphere = {}

    for normal in normals:
        norm_sph = get_spherical_coords(normal)
        if norm_sph not in hg_sphere:
            hg_sphere[norm_sph] = 0

        rotator = get_rotation_matrix(norm_sph)
        for point in points:
            vote(point, norm_sph, rotator, hg_sphere)

    return get_most_voted(hg_sphere)


"""
normal represents input of normal of a point.
@:return spherical co-ordinates equivalent.
"""


def get_spherical_coords(normal: np.ndarray):
    x, y, z = normal
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = math.atan2(y, x)
    phi = math.atan2(z, math.sqrt(x ** 2 + y ** 2))

    return np.array([r, theta, phi])


"""
Computes rotation matrix from z-axis based on normal in spherical co-ordinates.
"""


def get_rotation_matrix(normal: np.ndarray):
    # Create e3 basis vector.
    z = np.zeros(3)
    z[2] = 1

    # Compute normalised difference from e3 of normal.
    diff = z - normal
    diff_norm = np.linalg.norm(diff)
    diff = (1 / diff_norm) * diff

    identity = np.eye(3, 3)
    rotator = identity - 2 * np.dot(diff, diff)

    return rotator


"""
Point votes for a normal if the point lies on corresponding great circle.
"""


def vote(
        point: np.ndarray, normal: np.ndarray, rotator: np.matrix, norm_count: dict[np.ndarray, int]
):
    # Check if a point satisfies rotated parametric circle's equation, i.e., whether it lies on it. If so, vote.
    inverse_rot = np.linalg.inv(rotator)
    reversed_point = np.dot(inverse_rot, point)
    point_on_unit_circ = (reversed_point[0] ** 2 + reversed_point[1] ** 2) == 1
    if point_on_unit_circ:
        norm_count[normal] += 1


def get_most_voted(dictionary):
    max_key, max_value = None, 0
    for key, value in dictionary.items():
        if value > max_value:
            max_key, max_value = key, value

    return max_key


def estimate_radius_position(voted_normals: np.ndarray):
    normals = voted_normals
    kd_tree = KDTree(normals, leaf_size=40, metric='minowski')

    # Avoid incongruency just in case.
    normals, _ = kd_tree.get_arrays()

    k = 5
    for normal in normals:
        nearest_neighbors_indices = find_nearest_neighbors(normal, kd_tree, k)
        for index in nearest_neighbors_indices:
            find_orthogonal_basis(normals[index])


def find_nearest_neighbors(normal: np.ndarray, kd_tree: neighbors.KDTree, k: int) -> np.ndarray:
    dist, indices = kd_tree.query([normal], k=k, return_distance=False)
    return indices


def find_orthogonal_basis(normal: np.ndarray):
    basis = np.ndarray(normal, normal + [0, 0, 1], normal + [0, 1, 0])
    orthogonal_basis, _ = np.linalg.qr(basis)
    return orthogonal_basis

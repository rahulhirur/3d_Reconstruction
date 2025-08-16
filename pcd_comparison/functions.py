import numpy as np
from plyfile import PlyData, PlyElement
import io
import open3d as o3d
from scipy.spatial import cKDTree

import plotly.graph_objects as go
import json
import datetime
import os

def read_ply_file(ply_file):

    if isinstance(ply_file, str):
        return o3d.io.read_point_cloud(ply_file)
    elif hasattr(ply_file, 'read'):
        buffer = io.BytesIO(ply_file.read())
        plydata = PlyData.read(buffer)
        return plydata
    else:
        return None

def load_and_preprocess_pcd(uploaded_file):
    try:
        plydata = read_ply_file(uploaded_file)

        if 'vertex' not in plydata:
            print(f"PLY file {uploaded_file.name} does not contain any point data.")
            raise ValueError(f"PLY file {uploaded_file.name} does not contain a 'vertex' element.")
        
        # Extract the structured data from the 'vertex' element
        vertex_data = plydata['vertex'].data
        
        # Check if the data is a structured array and extract x, y, z
        if 'x' in vertex_data.dtype.names and 'y' in vertex_data.dtype.names and 'z' in vertex_data.dtype.names:
            points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
        else:
            # Fallback for simple unstructured arrays
            points = np.asarray(vertex_data)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
    except Exception as e:
        
        print(f"Error loading or preprocessing {uploaded_file}: {e}")
        return None

def generate_transformation_init(method = "Identity", translation_range=(-0.01, 0.01)):
    """
    Generates a random 4x4 transformation matrix with a valid random rotation
    and a random translation within a specified range.

    Args:
        translation_range (tuple): A tuple (min, max) for the translation values.

    Returns:
        np.ndarray: A random 4x4 transformation matrix.
    """
    #Make method variable lower case
    method = method.lower()
    if method == "identity":
        # Identity transformation
        return np.identity(4)

    elif method == "random":
        alpha, beta, gamma = np.random.uniform(-np.pi, np.pi, 3)

        # Rotation matrices for each axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])
        
        R_y = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        R_z = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        # Combine the rotations
        R = R_z @ R_y @ R_x

        # 2. Generate a random translation vector within the specified range
        t = np.random.uniform(translation_range[0], translation_range[1], 3)

        # 3. Combine rotation and translation into a 4x4 matrix
        trans_init = np.identity(4)
        trans_init[:3, :3] = R
        trans_init[:3, 3] = t

        return trans_init
    else:
        raise ValueError("Unknown transformation initialization method. Use 'Identity' or 'Random'.")

def perform_icp_registration(source_pcd, target_pcd, threshold=0.02, trans_init=np.identity(4)):
    """
    Performs ICP registration to align a source point cloud to a target point cloud.

    Args:
        source_pcd (o3d.geometry.PointCloud): The point cloud to be transformed.
        target_pcd (o3d.geometry.PointCloud): The reference point cloud.
        threshold (float): The maximum distance for a point-to-point correspondence.
        trans_init (np.ndarray): The initial 4x4 transformation matrix.

    Returns:
        o3d.pipelines.registration.RegistrationResult: An object containing the
                                                       transformation matrix and
                                                       registration metrics.
    """
    try:
        # Apply point-to-point ICP registration
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return reg_p2p
    
    except Exception as e:
        print(f"ICP registration failed: {e}")
        return None

def icp_align_and_compare(source_pcd, target_pcd, threshold=0.02, trans_init=np.identity(4), max_iter=50):
    """
    Aligns source_pcd to target_pcd using ICP, then computes RMSE, Chamfer, and Hausdorff distances.
    source_pcd and target_pcd are Open3D point clouds.
    """
    # Run ICP (align source to target)
    reg_icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    source_aligned = source_pcd.transform(reg_icp.transformation)

    # Convert to numpy arrays
    target_pts = np.asarray(target_pcd.points)
    source_pts = np.asarray(source_aligned.points)

    # Build KD-trees
    target_tree = o3d.geometry.KDTreeFlann(target_pcd)
    source_tree = o3d.geometry.KDTreeFlann(source_aligned)

    def nn_distances(source_pts, target_tree, target_pts):
        dists = []
        for p in source_pts:
            _, idx, _ = target_tree.search_knn_vector_3d(p, 1)
            nearest = target_pts[idx[0]]
            dists.append(np.linalg.norm(p - nearest))
        return np.array(dists)

    # Distances in both directions
    dists_target_to_source = nn_distances(target_pts, source_tree, source_pts)
    dists_source_to_target = nn_distances(source_pts, target_tree, target_pts)

    # Metrics
    rmse = np.sqrt(np.mean(dists_target_to_source**2))
    chamfer = np.mean(dists_target_to_source) + np.mean(dists_source_to_target)
    hausdorff = max(np.max(dists_target_to_source), np.max(dists_source_to_target))

    return reg_icp.transformation, rmse, chamfer, hausdorff

def save_comparison_results(output_dir, result_transformation, rmse, chamfer, hausdorff):
    result_data = {
        "transformation_matrix": result_transformation.tolist(),
        "rmse": rmse,
        "chamfer_distance": chamfer,
        "hausdorff_distance": hausdorff
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f'pcd_comparison_{timestamp}.ply')

    with open(file_path, "w") as f:
        json.dump(result_data, f, indent=4)

def add_noise_to_pcd(pcd, std_dev=0.05):
    """
    Adds Gaussian noise to a point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        std_dev (float): The standard deviation of the Gaussian noise.
                         A larger value means more noise.

    Returns:
        o3d.geometry.PointCloud: A new point cloud with added noise.
    """
    # 1. Convert the Open3D point cloud to a NumPy array
    points = np.asarray(pcd.points)

    # 2. Generate random noise from a normal distribution
    #    The noise array has the same shape as the points array.
    noise = np.random.normal(0, std_dev, size=points.shape)

    # 3. Add the noise to the original points
    noisy_points = points + noise

    # 4. Create a new point cloud with the noisy points
    noisy_pcd = o3d.geometry.PointCloud()
    noisy_pcd.points = o3d.utility.Vector3dVector(noisy_points)
    
    # Copy original colors if they exist
    if pcd.has_colors():
        noisy_pcd.colors = pcd.colors
        
    return noisy_pcd

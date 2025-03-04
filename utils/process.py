from typing import Tuple
import open3d as o3d
import numpy as np


def voxelize(pc_xyz:np.ndarray, pc_rgb:np.ndarray, voxel_size:float) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_pc_xyz = np.asarray(downsampled_pcd.points)
    downsampled_pc_rgb = np.asarray(downsampled_pcd.colors)
    return (downsampled_pc_xyz, downsampled_pc_rgb)


def mesh2pc(obj_path:str, num_points:int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(obj_path)
    pc = mesh.sample_points_uniformly(number_of_points=num_points)
    pc = np.asarray(pc.points)
    return pc

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def transform_pc(pc_camera:np.ndarray, c2w:np.ndarray) -> np.ndarray:
    # pc_camera: (N, 3), c2w: (4, 4)
    pc_camera_hm = np.concatenate([pc_camera, np.ones((pc_camera.shape[0], 1), dtype=pc_camera.dtype)], axis=-1)        # (N, 4)
    pc_world_hm = pc_camera_hm @ c2w.T                                                          # (N, 4)
    pc_world = pc_world_hm[:, :3]                                                               # (N, 3)
    return pc_world

def transform_dir(dir_camera:np.ndarray, c2w:np.ndarray) -> np.ndarray:
    # dir_camera: (N, 3), c2w: (4, 4)
    dir_camera_hm = np.concatenate([dir_camera, np.zeros((dir_camera.shape[0], 1), dtype=dir_camera.dtype)], axis=-1)   # (N, 4)
    dir_world_hm = dir_camera_hm @ c2w.T                                                        # (N, 4)
    dir_world = dir_world_hm[:, :3]                                                             # (N, 3)
    return dir_world


def xyzquat2mat(xyz:np.ndarray, quat:np.ndarray) -> np.ndarray:
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = Rot.from_quat(quat).as_matrix()
    pose_4x4[:3, 3] = xyz
    return pose_4x4


def delta_xyz(xyz1:np.ndarray, xyz2:np.ndarray) -> float:
    delta = xyz1 - xyz2
    delta_value = np.linalg.norm(delta)
    return delta_value

def delta_quat(quat1:np.ndarray, quat2:np.ndarray) -> float:
    delta = Rot.from_quat(quat1).inv() * Rot.from_quat(quat2)
    delta_value = delta.magnitude()
    return delta_value

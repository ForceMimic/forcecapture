import os
import configargparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
import tqdm
import h5py

from r3kit.devices.camera.realsense.t265 import T265
from r3kit.devices.camera.realsense.l515 import L515
from r3kit.devices.ftsensor.ati.pyati import PyATI as Pyft
from r3kit.devices.encoder.pdcd.angler import Angler
from configs.pose import *
from utils.annotation import search_stage
from utils.transformation import transform_pc
from utils.process import voxelize, mesh2pc

'''
Synchronize with `visualize_data.py` some part
'''


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--clip_object', action='store_true')
    parser.add_argument('--clip_pyft', action='store_true')
    parser.add_argument('--clip_base', action='store_true')
    parser.add_argument('--render_pyft_num', type=int)
    parser.add_argument('--voxel_size', type=float)
    parser.add_argument('--pc_num', type=int)

    args = parser.parse_args()
    return args


def main(args):
    # general config
    data_path = args.data_path
    save_path = args.save_path
    clip_object = args.clip_object
    clip_pyft = args.clip_pyft
    clip_base = args.clip_base
    render_pyft_num = args.render_pyft_num
    voxel_size = args.voxel_size
    pc_num = args.pc_num

    # load stage data
    with open(os.path.join(data_path, 'stage_timestamp_ms.json'), 'r') as f:
        stage_timestamp_ms = json.load(f)
        t265_pose_start_timestamp_ms = stage_timestamp_ms['t265_pose_start_timestamp_ms']
        t265_pose_end_timestamp_ms = stage_timestamp_ms['t265_pose_end_timestamp_ms']
        start_timestamp_ms = stage_timestamp_ms['start_timestamp_ms']
    # load l515 data
    l515_path = os.path.join(data_path, 'l515')
    l515_intrinsics = np.loadtxt(os.path.join(l515_path, 'intrinsics.txt'))     # (4,), float64
    l515_depth_scale = np.loadtxt(os.path.join(l515_path, 'depth_scale.txt')).item()
    l515_timestamps = np.load(os.path.join(l515_path, 'timestamps.npy'))
    # load t265 data
    t265_path = os.path.join(data_path, 't265')
    t265_pose_path = os.path.join(t265_path, 'pose')
    t265_pose_timestamps = np.load(os.path.join(t265_pose_path, 'timestamps.npy'))
    t265_xyzs = np.load(os.path.join(t265_pose_path, 'xyz.npy'))
    t265_quats = np.load(os.path.join(t265_pose_path, 'quat.npy'))
    # load pyft data
    pyft_path = os.path.join(data_path, 'pyft')
    with open(os.path.join(pyft_path, 'tare_pyft.json'), 'r') as f:
        pyft_tare = json.load(f)
    pyft_timestamps = np.load(os.path.join(pyft_path, 'timestamps.npy'))
    pyft_fts = np.load(os.path.join(pyft_path, 'ft.npy'))
    # load angler data
    angler_path = os.path.join(data_path, 'angler')
    angler_timestamps = np.load(os.path.join(angler_path, 'timestamps.npy'))
    angler_angles = np.load(os.path.join(angler_path, 'angle.npy'))
    # load annotation data
    annotation_path = os.path.join(data_path, 'annotation.json')
    with open(annotation_path, 'r') as f:
        stages = json.load(f)

    # deal with t265 special prepare
    t265_initial_pose_mask = np.logical_and(t265_pose_timestamps > t265_pose_start_timestamp_ms, t265_pose_timestamps < t265_pose_end_timestamp_ms)
    t265_initial_xyz = np.median(t265_xyzs[t265_initial_pose_mask, :], axis=0)
    t265_initial_quat = np.median(t265_quats[t265_initial_pose_mask, :], axis=0)
    t265_initial_pose = T265.raw2pose(t265_initial_xyz, t265_initial_quat)          # c02w

    # deal with angler special prepare
    angler_angles = Angler.raw2angle(angler_angles)
    angler_angles[angler_angles < 0] = 0.0
    angler_widths = angler_angles * ANGLE_2_WIDTH

    # process loop
    data_dict = {}
    stage_idxs = {'grasp': 0, 'shave': 0, 'turn': 0}
    stage_idx_map = {}
    for l515_current_idx in tqdm.trange(len(l515_timestamps)):
        # process l515 variables
        l515_current_timestamp = l515_timestamps[l515_current_idx]
        # process t265 variables
        t265_pose_current_idx = np.searchsorted(t265_pose_timestamps, l515_current_timestamp)
        t265_pose_current_idx = min(t265_pose_current_idx, len(t265_pose_timestamps)-1)
        # process pyft variables
        pyft_current_idx = np.searchsorted(pyft_timestamps, l515_current_timestamp)
        pyft_current_idx = min(pyft_current_idx, len(pyft_timestamps)-1)
        # process angler variables
        angler_current_idx = np.searchsorted(angler_timestamps, l515_current_timestamp)
        angler_current_idx = min(angler_current_idx, len(angler_timestamps)-1)

        # process stage
        stage_idx = search_stage(l515_current_timestamp, stages)
        stage = stages[stage_idx]
        t265_xyz_t265w_bias = np.array(stage['t265_xyz_t265w_bias'])
        if stage['stage'] == 'unrelated':
            continue
        if stage_idx not in stage_idx_map:
            stage_idx_map[stage_idx] = stage['stage'] + '_' + str(stage_idxs[stage['stage']]).zfill(2)
            stage_idxs[stage['stage']] = stage_idxs[stage['stage']] + 1
            data_dict[stage_idx_map[stage_idx]] = {
                'l515_pc_xyzs_l515': [], 
                'l515_pc_rgbs': [], 
                'l515_pc_xyzs_l515_mesh': [], 
                'l515_pc_rgbs_mesh': [], 
                'pyft_xyzs_l515': [], 
                'pyft_quats_l515': [], 
                'pyft_fs_pyft': [], 
                'pyft_ts_pyft': [], 
                'pyft_fs_l515': [], 
                'pyft_ts_l515': [], 
                'angler_widths': [], 
            }

        # process t265 elements
        t265_xyz_t265w, t265_quat_t265w = t265_xyzs[t265_pose_current_idx], t265_quats[t265_pose_current_idx]
        t265_xyz_t265w = t265_xyz_t265w + t265_xyz_t265w_bias
        t265_pose_t265w = T265.raw2pose(t265_xyz_t265w, t265_quat_t265w)                # c2w
        t265_pose_t2650 = np.linalg.inv(t265_initial_pose) @ t265_pose_t265w            # c2c0 = w2c0 @ c2w
        t265_pose_l515 = np.linalg.inv(L515_2_T265) @ t265_pose_t2650                   # c2l = c02l @ c2c0
        # process pyft elements
        pyft_pose_l515 = t265_pose_l515 @ np.linalg.inv(T265_2_PYFT)                    # f2l = c2l @ f2c
        pyft_xyz_l515, pyft_quat_l515 = pyft_pose_l515[:3, 3], Rot.from_matrix(pyft_pose_l515[:3, :3]).as_quat()
        data_dict[stage_idx_map[stage_idx]]['pyft_xyzs_l515'].append(pyft_xyz_l515)
        data_dict[stage_idx_map[stage_idx]]['pyft_quats_l515'].append(pyft_quat_l515)
        pyft_ft_pyft = pyft_fts[pyft_current_idx]
        pyft_pose_base = L515_2_BASE @ pyft_pose_l515                                   # f2b = l2b @ f2l
        pyft_ft_pyft = Pyft.raw2tare(pyft_ft_pyft, pyft_tare, pyft_pose_base[:3, :3])
        pyft_f_pyft, pyft_t_pyft = pyft_ft_pyft[:3], pyft_ft_pyft[3:]
        pyft_f_l515 = pyft_pose_l515[:3, :3] @ pyft_f_pyft
        pyft_t_l515 = pyft_pose_l515[:3, :3] @ pyft_t_pyft
        data_dict[stage_idx_map[stage_idx]]['pyft_fs_pyft'].append(pyft_f_pyft)
        data_dict[stage_idx_map[stage_idx]]['pyft_ts_pyft'].append(pyft_t_pyft)
        data_dict[stage_idx_map[stage_idx]]['pyft_fs_l515'].append(pyft_f_l515)
        data_dict[stage_idx_map[stage_idx]]['pyft_ts_l515'].append(pyft_t_l515)
        # process angler elements
        angler_width = angler_widths[angler_current_idx]
        data_dict[stage_idx_map[stage_idx]]['angler_widths'].append(angler_width)
        # process l515 elements
        l515_color_img = cv2.imread(os.path.join(l515_path, 'color', f'{str(l515_current_idx).zfill(16)}.png'), cv2.IMREAD_COLOR)
        l515_color_img = cv2.cvtColor(l515_color_img, cv2.COLOR_BGR2RGB)                    # (H, W, 3), uint8
        l515_color_img = l515_color_img / 255.                                              # (H, W, 3), float64
        l515_depth_img = cv2.imread(os.path.join(l515_path, 'depth', f'{str(l515_current_idx).zfill(16)}.png'), cv2.IMREAD_ANYDEPTH)    # (H, W), uint16
        l515_depth_img = l515_depth_img * l515_depth_scale                                  # (H, W), float64
        l515_pc_xyz_l515, l515_pc_rgb = L515.img2pc(l515_depth_img, l515_intrinsics, l515_color_img)
        if clip_object:
            l515_pc_xyz_base = transform_pc(l515_pc_xyz_l515, L515_2_BASE)
            clip_object_mask = (l515_pc_xyz_base[:, 0] > OBJECT_SPACE[0][0]) & (l515_pc_xyz_base[:, 0] < OBJECT_SPACE[0][1]) & \
                                (l515_pc_xyz_base[:, 1] > OBJECT_SPACE[1][0]) & (l515_pc_xyz_base[:, 1] < OBJECT_SPACE[1][1]) & \
                                (l515_pc_xyz_base[:, 2] > OBJECT_SPACE[2][0]) & (l515_pc_xyz_base[:, 2] < OBJECT_SPACE[2][1])
        else:
            clip_object_mask = np.zeros((l515_pc_xyz_l515.shape[0],), dtype=bool)
        if clip_pyft:
            l515_pc_xyz_pyft = transform_pc(l515_pc_xyz_l515, np.linalg.inv(pyft_pose_l515))
            clip_pyft_mask = (l515_pc_xyz_pyft[:, 0] > PYFT_SPACE[0][0]) & (l515_pc_xyz_pyft[:, 0] < PYFT_SPACE[0][1]) & \
                                (l515_pc_xyz_pyft[:, 1] > PYFT_SPACE[1][0]) & (l515_pc_xyz_pyft[:, 1] < PYFT_SPACE[1][1]) & \
                                (l515_pc_xyz_pyft[:, 2] > PYFT_SPACE[2][0]) & (l515_pc_xyz_pyft[:, 2] < PYFT_SPACE[2][1])
        else:
            clip_pyft_mask = np.zeros((l515_pc_xyz_l515.shape[0],), dtype=bool)
        if clip_base:
            l515_pc_xyz_base = transform_pc(l515_pc_xyz_l515, L515_2_BASE)
            clip_base_mask = (l515_pc_xyz_base[:, 0] > BASE_SPACE[0][0]) & (l515_pc_xyz_base[:, 0] < BASE_SPACE[0][1]) & \
                                (l515_pc_xyz_base[:, 1] > BASE_SPACE[1][0]) & (l515_pc_xyz_base[:, 1] < BASE_SPACE[1][1]) & \
                                (l515_pc_xyz_base[:, 2] > BASE_SPACE[2][0]) & (l515_pc_xyz_base[:, 2] < BASE_SPACE[2][1])
        else:
            clip_base_mask = np.ones((l515_pc_xyz_l515.shape[0],), dtype=bool)
        valid_mask = np.logical_and(clip_base_mask, np.logical_or(clip_object_mask, clip_pyft_mask))
        # TODO: hardcode to throw out hands
        l515_pc_xyz_pyft = transform_pc(l515_pc_xyz_l515, np.linalg.inv(pyft_pose_l515))
        valid_mask = np.logical_and(valid_mask, l515_pc_xyz_pyft[:, 2] > 0.)
        valid_mask = np.where(valid_mask)[0]
        l515_pc_xyz_l515 = l515_pc_xyz_l515[valid_mask]
        l515_pc_rgb = l515_pc_rgb[valid_mask]
        l515_pc_xyz_l515_mesh = l515_pc_xyz_l515.copy()
        l515_pc_rgb_mesh = l515_pc_rgb.copy()
        if render_pyft_num != 0:
            rfinger_pc_xyz_rfinger_mesh = mesh2pc(os.path.join("objs", "right_finger.obj"), num_points=render_pyft_num//2)
            lfinger_pc_xyz_lfinger_mesh = mesh2pc(os.path.join("objs", "left_finger.obj"), num_points=render_pyft_num//2)
            angler_finger_pose_pyft = np.identity(4)
            angler_finger_pose_pyft[0, 3] = angler_width / 2.
            pyft_right_finger_pose_l515 = pyft_pose_l515 @ angler_finger_pose_pyft
            rfinger_pc_xyz_l515_mesh = transform_pc(rfinger_pc_xyz_rfinger_mesh, pyft_right_finger_pose_l515)
            angler_finger_pose_pyft[0, 3] = -angler_width / 2.
            pyft_left_finger_pose_l515 = pyft_pose_l515 @ angler_finger_pose_pyft
            lfinger_pc_xyz_l515_mesh = transform_pc(lfinger_pc_xyz_lfinger_mesh, pyft_left_finger_pose_l515)
            rfinger_pc_rgb_mesh = np.ones_like(rfinger_pc_xyz_l515_mesh)
            lfinger_pc_rgb_mesh = np.ones_like(lfinger_pc_xyz_l515_mesh)
            l515_pc_xyz_l515_mesh = np.concatenate([l515_pc_xyz_l515_mesh, rfinger_pc_xyz_l515_mesh, lfinger_pc_xyz_l515_mesh], axis=0)
            l515_pc_rgb_mesh = np.concatenate([l515_pc_rgb_mesh, rfinger_pc_rgb_mesh, lfinger_pc_rgb_mesh], axis=0)
        if voxel_size != 0:
            l515_pc_xyz_l515, l515_pc_rgb = voxelize(l515_pc_xyz_l515, l515_pc_rgb, voxel_size)
            l515_pc_xyz_l515_mesh, l515_pc_rgb_mesh = voxelize(l515_pc_xyz_l515_mesh, l515_pc_rgb_mesh, voxel_size)
        if pc_num != -1:
            if l515_pc_xyz_l515.shape[0] > pc_num:
                valid_mask = np.random.choice(l515_pc_xyz_l515.shape[0], pc_num, replace=False)
            elif l515_pc_xyz_l515.shape[0] < pc_num:
                print(f"Warning: {l515_pc_xyz_l515.shape[0] = }")
                valid_mask = np.concatenate([np.arange(l515_pc_xyz_l515.shape[0]), np.random.choice(l515_pc_xyz_l515.shape[0], pc_num - l515_pc_xyz_l515.shape[0], replace=False)], axis=0)
            l515_pc_xyz_l515 = l515_pc_xyz_l515[valid_mask]
            l515_pc_rgb = l515_pc_rgb[valid_mask]
            if l515_pc_xyz_l515_mesh.shape[0] > pc_num:
                valid_mask = np.random.choice(l515_pc_xyz_l515_mesh.shape[0], pc_num, replace=False)
            elif l515_pc_xyz_l515_mesh.shape[0] < pc_num:
                print(f"Warning: {l515_pc_xyz_l515_mesh.shape[0] = }")
                valid_mask = np.concatenate([np.arange(l515_pc_xyz_l515_mesh.shape[0]), np.random.choice(l515_pc_xyz_l515_mesh.shape[0], pc_num - l515_pc_xyz_l515_mesh.shape[0], replace=False)], axis=0)
            l515_pc_xyz_l515_mesh = l515_pc_xyz_l515_mesh[valid_mask]
            l515_pc_rgb_mesh = l515_pc_rgb_mesh[valid_mask]
        data_dict[stage_idx_map[stage_idx]]['l515_pc_xyzs_l515'].append(l515_pc_xyz_l515)
        data_dict[stage_idx_map[stage_idx]]['l515_pc_rgbs'].append(l515_pc_rgb)
        data_dict[stage_idx_map[stage_idx]]['l515_pc_xyzs_l515_mesh'].append(l515_pc_xyz_l515_mesh)
        data_dict[stage_idx_map[stage_idx]]['l515_pc_rgbs_mesh'].append(l515_pc_rgb_mesh)
    
    # hdf5 loop
    os.makedirs(save_path, exist_ok=True)
    for stage_name in tqdm.tqdm(data_dict.keys()):
        with h5py.File(os.path.join(save_path, stage_name + '.hdf5'), 'w') as stage_hdf5:
            stage_hdf5_data_group = stage_hdf5.create_group('data')

            # save observation
            stage_hdf5_o_group = stage_hdf5_data_group.create_group('o')
            stage_hdf5_o_group.create_dataset('l515_pc_xyzs_l515', data=np.array(data_dict[stage_name]['l515_pc_xyzs_l515']))
            stage_hdf5_o_group.create_dataset('l515_pc_rgbs', data=np.array(data_dict[stage_name]['l515_pc_rgbs']))
            stage_hdf5_o_group.create_dataset('l515_pc_xyzs_l515_mesh', data=np.array(data_dict[stage_name]['l515_pc_xyzs_l515_mesh']))
            stage_hdf5_o_group.create_dataset('l515_pc_rgbs_mesh', data=np.array(data_dict[stage_name]['l515_pc_rgbs_mesh']))
            stage_hdf5_o_group.create_dataset('pyft_xyzs_l515', data=np.array(data_dict[stage_name]['pyft_xyzs_l515']))
            stage_hdf5_o_group.create_dataset('pyft_quats_l515', data=np.array(data_dict[stage_name]['pyft_quats_l515']))
            stage_hdf5_o_group.create_dataset('pyft_fs_pyft', data=np.array(data_dict[stage_name]['pyft_fs_pyft']))
            stage_hdf5_o_group.create_dataset('pyft_ts_pyft', data=np.array(data_dict[stage_name]['pyft_ts_pyft']))
            stage_hdf5_o_group.create_dataset('pyft_fs_l515', data=np.array(data_dict[stage_name]['pyft_fs_l515']))
            stage_hdf5_o_group.create_dataset('pyft_ts_l515', data=np.array(data_dict[stage_name]['pyft_ts_l515']))
            stage_hdf5_o_group.create_dataset('angler_widths', data=np.array(data_dict[stage_name]['angler_widths']))
            # save attributes
            stage_hdf5_data_group.attrs['num_samples'] = len(data_dict[stage_name]['l515_pc_xyzs_l515'])
            stage_hdf5_o_group.attrs['clip_object'] = clip_object
            stage_hdf5_o_group.attrs['clip_pyft'] = clip_pyft
            stage_hdf5_o_group.attrs['clip_base'] = clip_base
            stage_hdf5_o_group.attrs['render_pyft_num'] = render_pyft_num
            stage_hdf5_o_group.attrs['voxel_size'] = voxel_size
            stage_hdf5_o_group.attrs['pc_num'] = pc_num


if __name__ == '__main__':
    args = config_parse()
    main(args)

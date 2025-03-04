import os
import configargparse
import numpy as np
import open3d as o3d
from pynput import keyboard
import tqdm
import h5py

from r3kit.utils.vis import rotation_vec2mat
from utils.transformation import xyzquat2mat

'''
Synchronize with `visualize_merge.py` some part
'''


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--hdf5_path', type=str)
    parser.add_argument('--pc_mesh', action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    # general config
    hdf5_path = args.hdf5_path
    pc_mesh = args.pc_mesh

    # load hdf5
    with h5py.File(hdf5_path, 'r') as data_hdf5:
        data_group = data_hdf5['data']
        num_samples = data_group.attrs['num_samples']
        data_o_group = data_group['o']
        if pc_mesh:
            l515_pc_xyzs_l515 = data_o_group['l515_pc_xyzs_l515_mesh'][:].astype(np.float32)
            l515_pc_rgbs = data_o_group['l515_pc_rgbs_mesh'][:].astype(np.float32)
        else:
            l515_pc_xyzs_l515 = data_o_group['l515_pc_xyzs_l515'][:].astype(np.float32)
            l515_pc_rgbs = data_o_group['l515_pc_rgbs'][:].astype(np.float32)
        pyft_xyzs_l515 = data_o_group['pyft_xyzs_l515'][:].astype(np.float32)
        pyft_quats_l515 = data_o_group['pyft_quats_l515'][:].astype(np.float32)
        pyft_fs_pyft = data_o_group['pyft_fs_pyft'][:].astype(np.float32)
        pyft_ts_pyft = data_o_group['pyft_ts_pyft'][:].astype(np.float32)
        angler_widths = data_o_group['angler_widths'][:].astype(np.float32)

    # create keyboard listener
    quit = False
    reset = False
    pause = True
    zero = False
    forward = False
    backward = False
    speed = 1
    def _on_press(key):
        nonlocal quit, reset, pause, zero, forward, backward, speed
        if hasattr(key, 'char') and key.char == 'q':
            quit = True
            print("quit")
        if hasattr(key, 'char') and key.char == 'r':
            reset = True
            print("reset")
        if hasattr(key, 'char') and key.char == 'p':
            pause = not pause
            forward = False
            backward = False
            print("pause" if pause else "continue")
        if key == keyboard.Key.backspace:
            zero = True
            print("zero")
        if pause and key == keyboard.Key.right:
            forward = True
            print("forward")
        if pause and key == keyboard.Key.left:
            backward = True
            print("backward")
        if key == keyboard.Key.up:
            speed *= 2
            print(f"speed {speed}")
        if key == keyboard.Key.down:
            speed //= 2
            speed = max(speed, 1)
            print(f"speed {speed}")
    def _on_release(key):
        pass
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    # process variables
    current_idx = 0

    # create visualizer
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=1280, height=720, left=200, top=200, visible=True, window_name='data')

    # add l515 elements
    l515_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    visualizer.add_geometry(l515_frame)
    l515_pc_xyz_l515, l515_pc_rgb = l515_pc_xyzs_l515[current_idx], l515_pc_rgbs[current_idx]
    l515_pcd = o3d.geometry.PointCloud()
    l515_pcd.points = o3d.utility.Vector3dVector(l515_pc_xyz_l515)
    l515_pcd.colors = o3d.utility.Vector3dVector(l515_pc_rgb)
    visualizer.add_geometry(l515_pcd)
    # add pyft elements
    pyft_xyz_l515, pyft_quat_l515 = pyft_xyzs_l515[current_idx], pyft_quats_l515[current_idx]
    pyft_pose_l515 = xyzquat2mat(pyft_xyz_l515, pyft_quat_l515)
    pyft_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    pyft_frame.transform(pyft_pose_l515)
    visualizer.add_geometry(pyft_frame)
    pyft_f_pyft, pyft_t_pyft = pyft_fs_pyft[current_idx], pyft_ts_pyft[current_idx]
    pyft_f_l515 = pyft_pose_l515[:3, :3] @ pyft_f_pyft
    pyft_f_value = np.linalg.norm(pyft_f_l515)
    pyft_f_rotation_l515 = rotation_vec2mat(pyft_f_l515 / pyft_f_value)
    pyft_f_translation_l515 = pyft_pose_l515[:3, 3]
    pyft_t_l515 = pyft_pose_l515[:3, :3] @ pyft_t_pyft
    pyft_t_value = np.linalg.norm(pyft_t_l515)
    pyft_t_rotation_l515 = rotation_vec2mat(pyft_t_l515 / pyft_t_value)
    pyft_t_translation_l515 = pyft_pose_l515[:3, 3]
    pyft_f_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.04 * 0.025, cone_radius=0.04 * 0.05, cylinder_height=0.04 * 0.875, cone_height=0.04 * 0.125, 
                                                            resolution=20, cylinder_split=4, cone_split=1)
    pyft_f_arrow.paint_uniform_color([1., 1., 0.])
    pyft_f_arrow.scale(pyft_f_value, np.array([[0], [0], [0]]))
    pyft_f_arrow.rotate(pyft_f_rotation_l515, np.array([[0], [0], [0]]))
    pyft_f_arrow.translate(pyft_f_translation_l515)
    visualizer.add_geometry(pyft_f_arrow)
    pyft_t_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.4 * 0.025, cone_radius=0.4 * 0.05, cylinder_height=0.4 * 0.875, cone_height=0.4 * 0.125, 
                                                            resolution=20, cylinder_split=4, cone_split=1)
    pyft_t_arrow.paint_uniform_color([0., 1., 1.])
    pyft_t_arrow.scale(pyft_t_value, np.array([[0], [0], [0]]))
    pyft_t_arrow.rotate(pyft_t_rotation_l515, np.array([[0], [0], [0]]))
    pyft_t_arrow.translate(pyft_t_translation_l515)
    visualizer.add_geometry(pyft_t_arrow)
    pyft_gripper = o3d.io.read_triangle_mesh(os.path.join("objs", "gripper.obj"))
    pyft_gripper.transform(pyft_pose_l515)
    visualizer.add_geometry(pyft_gripper)
    # add angler elements
    angler_width = angler_widths[current_idx]
    angler_right_finger = o3d.io.read_triangle_mesh(os.path.join("objs", "right_finger.obj"))
    angler_left_finger = o3d.io.read_triangle_mesh(os.path.join("objs", "left_finger.obj"))
    angler_finger_pose_pyft = np.identity(4)
    angler_finger_pose_pyft[0, 3] = angler_width / 2.
    pyft_right_finger_pose_l515 = pyft_pose_l515 @ angler_finger_pose_pyft
    angler_right_finger.transform(pyft_right_finger_pose_l515)
    visualizer.add_geometry(angler_right_finger)
    angler_finger_pose_pyft[0, 3] = -angler_width / 2.
    pyft_left_finger_pose_l515 = pyft_pose_l515 @ angler_finger_pose_pyft
    angler_left_finger.transform(pyft_left_finger_pose_l515)
    visualizer.add_geometry(angler_left_finger)

    # visualizer setup
    view_control = visualizer.get_view_control()
    visualizer.get_render_option().background_color = [0, 0, 0]
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    # visualize loop
    pyft_xyz_deltas, pyft_quat_deltas, pyft_f_deltas, pyft_t_deltas, angler_width_deltas = [], [], [], [], []
    with tqdm.tqdm(total=num_samples) as pbar:
        while current_idx < num_samples:
            # update l515 elements
            l515_pc_xyz_l515, l515_pc_rgb = l515_pc_xyzs_l515[current_idx], l515_pc_rgbs[current_idx]
            l515_pcd.points = o3d.utility.Vector3dVector(l515_pc_xyz_l515)
            l515_pcd.colors = o3d.utility.Vector3dVector(l515_pc_rgb)
            visualizer.update_geometry(l515_pcd)
            # update pyft elements
            pyft_xyz_l515, pyft_quat_l515 = pyft_xyzs_l515[current_idx], pyft_quats_l515[current_idx]
            pyft_pose_l515_last = pyft_pose_l515.copy()
            pyft_pose_l515 = xyzquat2mat(pyft_xyz_l515, pyft_quat_l515)
            pyft_frame.transform(np.linalg.inv(pyft_pose_l515_last))
            pyft_frame.transform(pyft_pose_l515)
            pyft_delta_pose = np.dot(np.linalg.inv(pyft_pose_l515_last), pyft_pose_l515)
            visualizer.update_geometry(pyft_frame)
            pyft_f_pyft_last, pyft_t_pyft_last = pyft_f_pyft.copy(), pyft_t_pyft.copy()
            pyft_f_pyft, pyft_t_pyft = pyft_fs_pyft[current_idx], pyft_ts_pyft[current_idx]
            pyft_delta_f, pyft_delta_t = pyft_f_pyft - pyft_f_pyft_last, pyft_t_pyft - pyft_t_pyft_last
            pyft_f_l515 = pyft_pose_l515[:3, :3] @ pyft_f_pyft
            pyft_f_value_last = pyft_f_value.copy()
            pyft_f_value = np.linalg.norm(pyft_f_l515)
            pyft_f_rotation_l515_last = pyft_f_rotation_l515.copy()
            pyft_f_rotation_l515 = rotation_vec2mat(pyft_f_l515 / pyft_f_value)
            pyft_f_translation_l515_last = pyft_f_translation_l515.copy()
            pyft_f_translation_l515 = pyft_pose_l515[:3, 3]
            pyft_t_l515 = pyft_pose_l515[:3, :3] @ pyft_t_pyft
            pyft_t_value_last = pyft_t_value.copy()
            pyft_t_value = np.linalg.norm(pyft_t_l515)
            pyft_t_rotation_l515_last = pyft_t_rotation_l515.copy()
            pyft_t_rotation_l515 = rotation_vec2mat(pyft_t_l515 / pyft_t_value)
            pyft_t_translation_l515_last = pyft_t_translation_l515.copy()
            pyft_t_translation_l515 = pyft_pose_l515[:3, 3]
            pyft_f_arrow.translate(-pyft_f_translation_l515_last)
            pyft_f_arrow.rotate(np.linalg.inv(pyft_f_rotation_l515_last), np.array([[0], [0], [0]]))
            pyft_f_arrow.scale(1/pyft_f_value_last, np.array([[0], [0], [0]]))
            pyft_f_arrow.scale(pyft_f_value, np.array([[0], [0], [0]]))
            pyft_f_arrow.rotate(pyft_f_rotation_l515, np.array([[0], [0], [0]]))
            pyft_f_arrow.translate(pyft_f_translation_l515)
            visualizer.update_geometry(pyft_f_arrow)
            pyft_t_arrow.translate(-pyft_t_translation_l515_last)
            pyft_t_arrow.rotate(np.linalg.inv(pyft_t_rotation_l515_last), np.array([[0], [0], [0]]))
            pyft_t_arrow.scale(1/pyft_t_value_last, np.array([[0], [0], [0]]))
            pyft_t_arrow.scale(pyft_t_value, np.array([[0], [0], [0]]))
            pyft_t_arrow.rotate(pyft_t_rotation_l515, np.array([[0], [0], [0]]))
            pyft_t_arrow.translate(pyft_t_translation_l515)
            visualizer.update_geometry(pyft_t_arrow)
            pyft_gripper.transform(np.linalg.inv(pyft_pose_l515_last))
            pyft_gripper.transform(pyft_pose_l515)
            visualizer.update_geometry(pyft_gripper)
            # update angler elements
            angler_width_last = angler_width.copy()
            angler_width = angler_widths[current_idx]
            angler_finger_pose_pyft = np.identity(4)
            angler_finger_pose_pyft[0, 3] = angler_width / 2.
            pyft_right_finger_pose_l515_last = pyft_right_finger_pose_l515.copy()
            pyft_right_finger_pose_l515 = pyft_pose_l515 @ angler_finger_pose_pyft
            angler_right_finger.transform(np.linalg.inv(pyft_right_finger_pose_l515_last))
            angler_right_finger.transform(pyft_right_finger_pose_l515)
            visualizer.update_geometry(angler_right_finger)
            angler_finger_pose_pyft[0, 3] = -angler_width / 2.
            pyft_left_finger_pose_l515_last = pyft_left_finger_pose_l515.copy()
            pyft_left_finger_pose_l515 = pyft_pose_l515 @ angler_finger_pose_pyft
            angler_left_finger.transform(np.linalg.inv(pyft_left_finger_pose_l515_last))
            angler_left_finger.transform(pyft_left_finger_pose_l515)
            visualizer.update_geometry(angler_left_finger)

            # visualizer update
            visualizer.poll_events()
            visualizer.update_renderer()

            # pbar update
            current_idx_last = current_idx
            if not pause:
                current_idx += speed
                pbar.update(speed)
            else:
                if forward:
                    current_idx += speed
                    pbar.update(speed)
                    forward = False
                elif backward:
                    current_idx -= speed
                    pbar.update(-speed)
                    backward = False
                else:
                    pass
            pbar.set_postfix(f=pyft_f_value, t=pyft_t_value)

            # keyboard quit
            if quit:
                break
            # keyboard reset
            if reset:
                view_control = visualizer.get_view_control()
                params = view_control.convert_to_pinhole_camera_parameters()
                params.extrinsic = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
                view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
                reset = False
            # keyboard zero
            if zero:
                pbar.update(-current_idx)
                current_idx = 0
                zero = False
            
            if current_idx != current_idx_last:
                pyft_xyz_delta_mm = np.linalg.norm(pyft_delta_pose[:3, 3]) * 1000
                pyft_quat_delta_deg = np.arccos(np.clip((np.trace(pyft_delta_pose[:3, :3]) - 1) / 2, -1, 1)) / np.pi * 180
                angler_width_delta_mm = abs(angler_width - angler_width_last) * 1000
                pyft_f_delta_n = np.linalg.norm(pyft_delta_f)
                pyft_t_delta = np.linalg.norm(pyft_delta_t)

                pyft_xyz_deltas.append(pyft_xyz_delta_mm)
                pyft_quat_deltas.append(pyft_quat_delta_deg)
                angler_width_deltas.append(angler_width_delta_mm)
                pyft_f_deltas.append(pyft_f_delta_n)
                pyft_t_deltas.append(pyft_t_delta)

    visualizer.destroy_window()
    listener.stop()

    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    args = config_parse()
    main(args)

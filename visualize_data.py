import os
import configargparse
import json
import numpy as np
import cv2
import open3d as o3d
from pynput import keyboard
import tqdm

from r3kit.devices.camera.realsense.t265 import T265
from r3kit.devices.camera.realsense.l515 import L515
from r3kit.devices.ftsensor.ati.pyati import PyATI as Pyft
from r3kit.devices.encoder.pdcd.angler import Angler
from r3kit.utils.vis import rotation_vec2mat
from configs.pose import *
from utils.annotation import search_stage

'''
Synchronize with `create_hdf5.py` some part
'''


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--fps', type=int, default=10)

    args = parser.parse_args()
    return args


def main(args):
    # general config
    data_path = args.data_path
    fps = args.fps
    frame_interval_ms = 1000. / fps

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
    ### l515_depth_img, l515_color_img loaded during iteration
    # load t265 data
    t265_path = os.path.join(data_path, 't265')
    ### t265_image_path = os.path.join(t265_path, 'image')
    ### t265_image_timestamps = np.load(os.path.join(t265_image_path, 'timestamps.npy'))
    ### t265_left_img, t265_right_img loaded during iteration
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
    has_annotation = os.path.exists(annotation_path)
    if has_annotation:
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

    # deal with t265 special prepare
    t265_initial_pose_mask = np.logical_and(t265_pose_timestamps > t265_pose_start_timestamp_ms, t265_pose_timestamps < t265_pose_end_timestamp_ms)
    t265_initial_xyz = np.median(t265_xyzs[t265_initial_pose_mask, :], axis=0)
    t265_initial_quat = np.median(t265_quats[t265_initial_pose_mask, :], axis=0)
    t265_initial_pose = T265.raw2pose(t265_initial_xyz, t265_initial_quat)    # c02w

    # deal with angler special prepare
    angler_angles = Angler.raw2angle(angler_angles)
    angler_angles[angler_angles < 0] = 0.0
    angler_widths = angler_angles * ANGLE_2_WIDTH

    # process l515 variables
    l515_current_idx = 0
    l515_current_timestamp = l515_timestamps[l515_current_idx]
    l515_start_timestamp = l515_timestamps[0]
    l515_end_timestamp = l515_timestamps[-1]
    # process t265 variables
    ### t265_image_current_idx = np.searchsorted(t265_image_timestamps, l515_current_timestamp)
    t265_pose_current_idx = np.searchsorted(t265_pose_timestamps, l515_current_timestamp)
    # process pyft variables
    pyft_current_idx = np.searchsorted(pyft_timestamps, l515_current_timestamp)
    # process angler variables
    angler_current_idx = np.searchsorted(angler_timestamps, l515_current_timestamp)

    # create keyboard listener
    quit = False
    reset = False
    pause = False
    zero = False
    forward = False
    backward = False
    speed = 1
    if not has_annotation:
        minus = False
        t265_xyz_t265w_bias = np.array([0., 0., 0.])
        stages = [{'timestamp_ms': l515_current_timestamp, 
                   't265_xyz_t265w_bias': t265_xyz_t265w_bias.tolist(), 
                   'stage': 'unrelated'}]
    else:
        stages = annotation
        stage_idx = search_stage(l515_current_timestamp, stages)
        stage = stages[stage_idx]
        t265_xyz_t265w_bias = np.array(stage['t265_xyz_t265w_bias'])
    def _on_press(key):
        nonlocal quit, reset, pause, zero, forward, backward, speed
        nonlocal current_timestamp, stages, minus, t265_xyz_t265w_bias
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
        if pause and key == keyboard.Key.up:
            speed *= 2
            print(f"speed {speed}")
        if pause and key == keyboard.Key.down:
            speed //= 2
            speed = max(speed, 1)
            print(f"speed {speed}")
        if not has_annotation:
            if hasattr(key, 'char') and key.char == 'u':
                stages.append({'timestamp_ms': current_timestamp, 
                                't265_xyz_t265w_bias': t265_xyz_t265w_bias.tolist(), 
                                'stage': 'unrelated'})
                print(f"unrelated from {current_timestamp}")
            if hasattr(key, 'char') and key.char == 'g':
                stages.append({'timestamp_ms': current_timestamp, 
                                't265_xyz_t265w_bias': t265_xyz_t265w_bias.tolist(), 
                                'stage': 'grasp'})
                print(f"grasp from {current_timestamp}")
            if hasattr(key, 'char') and key.char == 's':
                stages.append({'timestamp_ms': current_timestamp, 
                                't265_xyz_t265w_bias': t265_xyz_t265w_bias.tolist(), 
                                'stage': 'shave'})
                print(f"shave from {current_timestamp}")
            if hasattr(key, 'char') and key.char == 't':
                stages.append({'timestamp_ms': current_timestamp, 
                                't265_xyz_t265w_bias': t265_xyz_t265w_bias.tolist(), 
                                'stage': 'turn'})
                print(f"turn from {current_timestamp}")
            if hasattr(key, 'char') and key.char == 'm':
                minus = not minus
                print("bias minus" if minus else "bias plus")
            if hasattr(key, 'char') and key.char == 'x':
                t265_xyz_t265w_bias = t265_xyz_t265w_bias + np.array([0.005 if not minus else -0.005, 0., 0.])
                stage_idx = search_stage(current_timestamp, stages)
                stages[stage_idx]['t265_xyz_t265w_bias'] = t265_xyz_t265w_bias.tolist()
            if hasattr(key, 'char') and key.char == 'y':
                t265_xyz_t265w_bias = t265_xyz_t265w_bias + np.array([0., 0.005 if not minus else -0.005, 0.])
                stage_idx = search_stage(current_timestamp, stages)
                stages[stage_idx]['t265_xyz_t265w_bias'] = t265_xyz_t265w_bias.tolist()
            if hasattr(key, 'char') and key.char == 'z':
                t265_xyz_t265w_bias = t265_xyz_t265w_bias + np.array([0., 0., 0.005 if not minus else -0.005])
                stage_idx = search_stage(current_timestamp, stages)
                stages[stage_idx]['t265_xyz_t265w_bias'] = t265_xyz_t265w_bias.tolist()
            if key == keyboard.Key.delete:
                stages.pop()
                print("delete")
    def _on_release(key):
        pass
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    # create visualizer
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=1280, height=720, left=200, top=200, visible=True, window_name='data')

    # add l515 elements
    l515_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    visualizer.add_geometry(l515_frame)
    l515_color_img = cv2.imread(os.path.join(l515_path, 'color', f'{str(l515_current_idx).zfill(16)}.png'), cv2.IMREAD_COLOR)
    l515_color_img = cv2.cvtColor(l515_color_img, cv2.COLOR_BGR2RGB)                    # (H, W, 3), uint8
    l515_color_img = l515_color_img / 255.                                              # (H, W, 3), float64
    l515_depth_img = cv2.imread(os.path.join(l515_path, 'depth', f'{str(l515_current_idx).zfill(16)}.png'), cv2.IMREAD_ANYDEPTH)    # (H, W), uint16
    l515_depth_img = l515_depth_img * l515_depth_scale                                  # (H, W), float64
    l515_pc_xyz_l515, l515_pc_rgb = L515.img2pc(l515_depth_img, l515_intrinsics, l515_color_img)
    l515_pcd = o3d.geometry.PointCloud()
    l515_pcd.points = o3d.utility.Vector3dVector(l515_pc_xyz_l515)
    l515_pcd.colors = o3d.utility.Vector3dVector(l515_pc_rgb)
    visualizer.add_geometry(l515_pcd)
    # add t265 elements
    t265_xyz_t265w, t265_quat_t265w = t265_xyzs[t265_pose_current_idx], t265_quats[t265_pose_current_idx]
    t265_xyz_t265w = t265_xyz_t265w + t265_xyz_t265w_bias
    t265_pose_t265w = T265.raw2pose(t265_xyz_t265w, t265_quat_t265w)                    # c2w
    t265_pose_t2650 = np.linalg.inv(t265_initial_pose) @ t265_pose_t265w                # c2c0 = w2c0 @ c2w
    t265_pose_l515 = np.linalg.inv(L515_2_T265) @ t265_pose_t2650                       # c2l = c02l @ c2c0
    # t265_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    # t265_frame.transform(t265_pose_l515)
    # visualizer.add_geometry(t265_frame)
    ### t265_left_img = cv2.imread(os.path.join(t265_image_path, 'left', f'{str(t265_image_current_idx).zfill(16)}.png'), cv2.IMREAD_GRAYSCALE)         # (H, W), uint8
    ### cv2.namedWindow('t265_left', cv2.WINDOW_NORMAL)
    ### cv2.imshow('t265_left', t265_left_img)
    ### cv2.waitKey(1)
    ### t265_right_img = cv2.imread(os.path.join(t265_image_path, 'right', f'{str(t265_image_current_idx).zfill(16)}.png'), cv2.IMREAD_GRAYSCALE)       # (H, W), uint8
    ### cv2.namedWindow('t265_right', cv2.WINDOW_NORMAL)
    ### cv2.imshow('t265_right', t265_right_img)
    ### cv2.waitKey(1)
    # add pyft elements
    pyft_pose_l515 = t265_pose_l515 @ np.linalg.inv(T265_2_PYFT)                        # f2l = c2l @ f2c
    pyft_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    pyft_frame.transform(pyft_pose_l515)
    visualizer.add_geometry(pyft_frame)
    pyft_ft_pyft = pyft_fts[pyft_current_idx]
    pyft_pose_base = L515_2_BASE @ pyft_pose_l515                                       # f2b = l2b @ f2l
    pyft_ft_pyft = Pyft.raw2tare(pyft_ft_pyft, pyft_tare, pyft_pose_base[:3, :3])
    pyft_f_pyft, pyft_t_pyft = pyft_ft_pyft[:3], pyft_ft_pyft[3:]
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
    angler_width = angler_widths[angler_current_idx]
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
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    # visualize loop
    show_timestamps = np.arange(l515_start_timestamp+frame_interval_ms, l515_end_timestamp+1e-3, frame_interval_ms)
    with tqdm.tqdm(total=len(show_timestamps)) as pbar:
        show_idx = 0
        while show_idx < len(show_timestamps):
            current_timestamp = show_timestamps[show_idx]
            print(current_timestamp)
            stage_idx = search_stage(current_timestamp, stages)
            stage = stages[stage_idx]
            t265_xyz_t265w_bias = np.array(stage['t265_xyz_t265w_bias'])

            # update l515 variables
            l515_current_idx = np.searchsorted(l515_timestamps, current_timestamp)
            l515_current_idx = min(l515_current_idx, len(l515_timestamps)-1)
            l515_current_time = (l515_timestamps[l515_current_idx] - l515_start_timestamp) / 1000.
            # update t265 variables
            ### t265_image_current_idx = np.searchsorted(t265_image_timestamps, current_timestamp)
            ### t265_image_current_idx = min(t265_image_current_idx, len(t265_image_timestamps)-1)
            ### t265_image_current_time = (t265_image_timestamps[t265_image_current_idx] - l515_start_timestamp) / 1000.
            t265_pose_current_idx = np.searchsorted(t265_pose_timestamps, current_timestamp)
            t265_pose_current_idx = min(t265_pose_current_idx, len(t265_pose_timestamps)-1)
            t265_pose_current_time = (t265_pose_timestamps[t265_pose_current_idx] - l515_start_timestamp) / 1000.
            # update pyft variables
            pyft_current_idx = np.searchsorted(pyft_timestamps, current_timestamp)
            pyft_current_idx = min(pyft_current_idx, len(pyft_timestamps)-1)
            pyft_current_time = (pyft_timestamps[pyft_current_idx] - l515_start_timestamp) / 1000.
            # update angler variables
            angler_current_idx = np.searchsorted(angler_timestamps, current_timestamp)
            angler_current_idx = min(angler_current_idx, len(angler_timestamps)-1)
            angler_current_time = (angler_timestamps[angler_current_idx] - l515_start_timestamp) / 1000.
            
            # update l515 elements
            l515_color_img = cv2.imread(os.path.join(l515_path, 'color', f'{str(l515_current_idx).zfill(16)}.png'), cv2.IMREAD_COLOR)
            l515_color_img = cv2.cvtColor(l515_color_img, cv2.COLOR_BGR2RGB)            # (H, W, 3), uint8
            l515_color_img = l515_color_img / 255.                                      # (H, W, 3), float64
            l515_depth_img = cv2.imread(os.path.join(l515_path, 'depth', f'{str(l515_current_idx).zfill(16)}.png'), cv2.IMREAD_ANYDEPTH)    # (H, W), uint16
            l515_depth_img = l515_depth_img * l515_depth_scale                                     # (H, W), float64
            l515_pc_xyz_l515, l515_pc_rgb = L515.img2pc(l515_depth_img, l515_intrinsics, l515_color_img)
            l515_pcd.points = o3d.utility.Vector3dVector(l515_pc_xyz_l515)
            l515_pcd.colors = o3d.utility.Vector3dVector(l515_pc_rgb)
            visualizer.update_geometry(l515_pcd)
            # update t265 elements
            t265_xyz_t265w, t265_quat_t265w = t265_xyzs[t265_pose_current_idx], t265_quats[t265_pose_current_idx]
            t265_xyz_t265w = t265_xyz_t265w + t265_xyz_t265w_bias
            t265_pose_t265w = T265.raw2pose(t265_xyz_t265w, t265_quat_t265w)                    # c2w
            t265_pose_t2650 = np.linalg.inv(t265_initial_pose) @ t265_pose_t265w                # c2c0 = w2c0 @ c2w
            t265_pose_l515_last = t265_pose_l515.copy()
            t265_pose_l515 = np.linalg.inv(L515_2_T265) @ t265_pose_t2650                       # c2l = c02l @ c2c0
            # t265_frame.transform(np.linalg.inv(t265_pose_l515_last))
            # t265_frame.transform(t265_pose_l515)
            # visualizer.update_geometry(t265_frame)
            ### t265_left_img = cv2.imread(os.path.join(t265_image_path, 'left', f'{str(t265_image_current_idx).zfill(16)}.png'), cv2.IMREAD_GRAYSCALE)         # (H, W), uint8
            ### cv2.imshow('t265_left', t265_left_img)
            ### cv2.waitKey(1)
            ### t265_right_img = cv2.imread(os.path.join(t265_image_path, 'right', f'{str(t265_image_current_idx).zfill(16)}.png'), cv2.IMREAD_GRAYSCALE)       # (H, W), uint8
            ### cv2.imshow('t265_right', t265_right_img)
            ### cv2.waitKey(1)
            # update pyft elements
            pyft_pose_l515_last = pyft_pose_l515.copy()
            pyft_pose_l515 = t265_pose_l515 @ np.linalg.inv(T265_2_PYFT)
            pyft_frame.transform(np.linalg.inv(pyft_pose_l515_last))
            pyft_frame.transform(pyft_pose_l515)
            visualizer.update_geometry(pyft_frame)
            pyft_ft_pyft = pyft_fts[pyft_current_idx]
            ### print(pyft_ft_pyft)
            pyft_pose_base = L515_2_BASE @ pyft_pose_l515                                       # f2b = l2b @ f2l
            pyft_ft_pyft = Pyft.raw2tare(pyft_ft_pyft, pyft_tare, pyft_pose_base[:3, :3])
            pyft_f_pyft, pyft_t_pyft = pyft_ft_pyft[:3], pyft_ft_pyft[3:]
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
            angler_width = angler_widths[angler_current_idx]
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
            if not pause:
                show_idx += 1
                pbar.update(1)
            else:
                if forward:
                    show_idx += speed
                    pbar.update(speed)
                    forward = False
                elif backward:
                    show_idx -= speed
                    pbar.update(-speed)
                    backward = False
                else:
                    pass
            pbar.set_postfix(f=pyft_f_value, t=pyft_t_value, s=stage['stage'])

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
                pbar.update(-show_idx)
                show_idx = 0
                zero = False

    visualizer.destroy_window()
    listener.stop()

    if not has_annotation:
        print(stages)
        with open(annotation_path, 'w') as f:
            json.dump(stages, f, indent=4)


if __name__ == '__main__':
    args = config_parse()
    main(args)

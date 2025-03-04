import os
import time
import configargparse
import json
from pynput import keyboard
import numpy as np
import open3d as o3d

from r3kit.devices.camera.realsense.t265 import T265
from r3kit.devices.ftsensor.ati.pyati import PyATI as Pyft
from r3kit.algos.tare.linear import LinearMFTarer, LinearFTarer, LinearCTTarer
from configs.pose import *


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--t265_id', type=str)
    parser.add_argument('--pyft_id', type=str)
    parser.add_argument('--pyft_port', type=int)

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_path):
        # create devices
        t265 = T265(id=args.t265_id, image=False, name='t265')
        pyft = Pyft(id=args.pyft_id, port=args.pyft_port, fps=200, name='pyft')

        # special prepare for t265 to construct map
        start = input('Press enter to start T265 construct map')
        if start != '':
            del t265, pyft
            return
        print("Start T265 construct map")
        t265.collect_streaming(False)
        t265.start_streaming()

        # stable t265 pose
        stop_start = input('Press enter to stop T265 construct map and start stabling T265 pose')
        if stop_start != '':
            del t265, pyft
            return
        print("Stop T265 construct map and start stabling T265 pose")
        t265.collect_streaming(True)
        t265_pose_start_timestamp_ms = time.time() * 1000

        # mounting t265
        stop_start = input('Press enter to stop stabling T265 pose and start mounting T265')
        if stop_start != '':
            del t265, pyft
            return
        print("Stop stabling T265 pose and start mounting T265")
        t265_pose_end_timestamp_ms = time.time() * 1000

        # start streaming
        stop_start = input('Press enter to stop mounting T265 and start streaming')
        if stop_start != '':
            del t265, pyft
            return
        print("Stop mounting T265 and start streaming")
        pyft_fts, t265_poses = [], []
        print("Press enter to stop streaming")
        stop = False
        def _on_press(key):
            nonlocal stop
            if key == keyboard.Key.enter:
                stop = True
        def _on_release(key):
            pass
        listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        listener.start()
        while not stop:
            pyft_ft = pyft.get_mean_data(n=10, name='ft')
            _, _, t265_xyz, t265_quat = t265.get()
            t265_pose = T265.raw2pose(t265_xyz, t265_quat)
            pyft_fts.append(pyft_ft)
            t265_poses.append(t265_pose)

        # stop streaming
        print("Stop streaming")
        t265_data = t265.stop_streaming()
        listener.stop()

        # destroy devices
        del t265, pyft

        # NOTE: urgly code to save data
        os.makedirs(args.save_path, exist_ok=True)
        np.save(os.path.join(args.save_path, 't265_timestamp.npy'), np.array(t265_data['pose']["timestamp_ms"]))
        np.savetxt(os.path.join(args.save_path, 't265_stage_timestamps.txt'), np.array([t265_pose_start_timestamp_ms, t265_pose_end_timestamp_ms]))
        np.save(os.path.join(args.save_path, 't265_xyz.npy'), np.array(t265_data['pose']["xyz"]))
        np.save(os.path.join(args.save_path, 't265_quat.npy'), np.array(t265_data['pose']["quat"]))
        np.save(os.path.join(args.save_path, 't265_poses.npy'), np.array(t265_poses))
        np.save(os.path.join(args.save_path, 'pyft_fts.npy'), np.array(pyft_fts))
    else:
        # reload data
        t265_data = {
            'pose': {
                'timestamp_ms': np.load(os.path.join(args.save_path, 't265_timestamp.npy')), 
                'xyz': np.load(os.path.join(args.save_path, 't265_xyz.npy')), 
                'quat': np.load(os.path.join(args.save_path, 't265_quat.npy'))
            }
        }
        t265_pose_start_timestamp_ms, t265_pose_end_timestamp_ms = np.loadtxt(os.path.join(args.save_path, 't265_stage_timestamps.txt'))
        t265_poses = np.load(os.path.join(args.save_path, 't265_poses.npy'))
        pyft_fts = np.load(os.path.join(args.save_path, 'pyft_fts.npy'))

    # tare
    t265_all_poses = t265_data['pose']
    t265_initial_pose_mask = np.logical_and(np.array(t265_all_poses["timestamp_ms"]) > t265_pose_start_timestamp_ms, 
                                            np.array(t265_all_poses["timestamp_ms"]) < t265_pose_end_timestamp_ms)
    t265_initial_xyzs = np.array(t265_all_poses["xyz"])[t265_initial_pose_mask]
    t265_initial_quats = np.array(t265_all_poses["quat"])[t265_initial_pose_mask]
    t265_initial_xyz = np.median(t265_initial_xyzs, axis=0)
    t265_initial_quat = np.median(t265_initial_quats, axis=0)
    t265_initial_pose = T265.raw2pose(t265_initial_xyz, t265_initial_quat)  # c02w
    pyft_poses = []
    for t265_pose in t265_poses:                                            # c2w
        t265_pose = np.linalg.inv(t265_initial_pose) @ t265_pose            # c2c0 = w2c0 @ c2w
        t265_pose = T265_2_BASE @ t265_pose                                 # c2b = c02b @ c2c0
        pyft_pose = t265_pose @ np.linalg.inv(T265_2_PYFT)                  # f2b = c2b @ f2c
        pyft_poses.append(pyft_pose)
    mftarer = LinearMFTarer()
    for pyft_ft, pyft_pose in zip(pyft_fts, pyft_poses):
        mftarer.add_data(pyft_ft[:3], pyft_pose[:3, :3])
    result = mftarer.run()
    ftarer = LinearFTarer()
    ftarer.set_m(result['m'])
    for pyft_ft, pyft_pose in zip(pyft_fts, pyft_poses):
        ftarer.add_data(pyft_ft[:3], pyft_pose[:3, :3])
    result.update(ftarer.run())
    ctarer = LinearCTTarer()
    ctarer.set_m(result['m'])
    for pyft_ft, pyft_pose in zip(pyft_fts, pyft_poses):
        ctarer.add_data(pyft_ft[3:], pyft_pose[:3, :3])
    result.update(ctarer.run())
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
    print(result)

    # save data
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'tare_pyft.json'), 'w') as f:
        json.dump(result, f, indent=4)
    
    # visualize
    geometries = []
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(base_frame)
    for pyft_pose in pyft_poses[::(len(pyft_poses) // 200)]:
        pyft_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
        pyft_frame.transform(pyft_pose)
        geometries.append(pyft_frame)
    o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    args = config_parse()
    main(args)

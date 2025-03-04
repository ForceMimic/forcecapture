import os
import time
import configargparse
import json
from copy import deepcopy
from pynput import keyboard
import numpy as np

from r3kit.devices.camera.realsense.t265 import T265
from r3kit.devices.ftsensor.ati.pyati import PyATI as Pyft
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
    # read tare data
    with open(os.path.join(args.save_path, 'tare_pyft.json'), 'r') as f:
        tare_data = json.load(f)
        for key, value in tare_data.items():
            if isinstance(value, list):
                tare_data[key] = np.array(value)
    
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

    # NOTE: urgly code to get t265 initial pose
    t265.pose_streaming_mutex.acquire()
    t265_all_poses = deepcopy(t265.pose_streaming_data)
    t265.pose_streaming_mutex.release()
    t265_initial_pose_mask = np.logical_and(np.array(t265_all_poses["timestamp_ms"]) > t265_pose_start_timestamp_ms, 
                                            np.array(t265_all_poses["timestamp_ms"]) < t265_pose_end_timestamp_ms)
    t265_initial_xyzs = np.array(t265_all_poses["xyz"])[t265_initial_pose_mask]
    t265_initial_quats = np.array(t265_all_poses["quat"])[t265_initial_pose_mask]
    t265_initial_xyz = np.median(t265_initial_xyzs, axis=0)
    t265_initial_quat = np.median(t265_initial_quats, axis=0)
    t265_initial_pose = T265.raw2pose(t265_initial_xyz, t265_initial_quat)  # c02w

    # start streaming
    stop_start = input('Press enter to stop mounting T265 and start streaming')
    if stop_start != '':
        del t265, pyft
        return
    print("Stop mounting T265 and start streaming")
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
        t265_pose = T265.raw2pose(t265_xyz, t265_quat)                      # c2w
        t265_pose = np.linalg.inv(t265_initial_pose) @ t265_pose            # c2c0 = w2c0 @ c2w
        t265_pose = T265r_2_BASE @ t265_pose                                # c2b = c02b @ c2c0
        pyft_pose = t265_pose @ np.linalg.inv(T265r_2_PYFT)                 # f2b = c2b @ f2c
        pyft_ft = Pyft.raw2tare(raw_ft=pyft_ft, tare=tare_data, pose=pyft_pose[:3, :3])
        print(np.linalg.norm(pyft_ft[:3]), np.linalg.norm(pyft_ft[3:]))
        print(pyft_ft[:3], pyft_ft[3:])
        time.sleep(0.1)

    # stop streaming
    print("Stop streaming")
    t265.stop_streaming()
    listener.stop()

    # destroy devices
    del t265, pyft


if __name__ == '__main__':
    args = config_parse()
    main(args)

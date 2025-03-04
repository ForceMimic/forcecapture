import os
import shutil
import time
import configargparse
import json

from r3kit.devices.camera.realsense.t265 import T265
from r3kit.devices.camera.realsense.l515 import L515
from r3kit.devices.ftsensor.ati.pyati import PyATI as Pyft
from r3kit.devices.encoder.pdcd.angler import Angler


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--l515_id', type=str)
    parser.add_argument('--t265r_id', type=str)
    parser.add_argument('--t265l_id', type=str)
    parser.add_argument('--pyft_id', type=str)
    parser.add_argument('--pyft_port', type=int)
    parser.add_argument('--pyft_tare_path', type=str)
    parser.add_argument('--angler_id', type=str)
    parser.add_argument('--angler_index', type=int)

    args = parser.parse_args()
    return args


def main(args):
    # create devices
    l515 = L515(id=args.l515_id, name='l515')
    t265r = T265(id=args.t265r_id, image=False, name='t265r')
    t265l = T265(id=args.t265l_id, image=False, name='t265l')
    pyft = Pyft(id=args.pyft_id, port=args.pyft_port, fps=200, name='pyft')
    angler = Angler(id=args.angler_id, index=args.angler_index, fps=30, name='angler')

    # special prepare to close the gripper for angler to know bias
    start = input('Press enter to start closing gripper')
    if start != '':
        del l515, t265r, t265l, pyft, angler
        return
    print("Start closing gripper")

    # special prepare for t265 to construct map
    stop_start = input('Press enter to stop closing gripper and start T265 construct map')
    if stop_start != '':
        del l515, t265r, t265l, pyft, angler
        return
    print("Stop closing gripper and Start T265 construct map")
    t265r.collect_streaming(False)
    t265l.collect_streaming(False)
    t265r.start_streaming()
    t265l.start_streaming()

    # stable t265 pose
    stop_start = input('Press enter to stop T265 construct map and start stabling T265 pose')
    if stop_start != '':
        del l515, t265r, t265l, pyft, angler
        return
    print("Stop T265 construct map and start stabling T265 pose")
    t265r.collect_streaming(True)
    t265l.collect_streaming(True)
    t265_pose_start_timestamp_ms = time.time() * 1000

    # mounting t265
    stop_start = input('Press enter to stop stabling T265 pose and start mounting T265')
    if stop_start != '':
        del l515, t265r, t265l, pyft, angler
        return
    print("Stop stabling T265 pose and start mounting T265")
    t265_pose_end_timestamp_ms = time.time() * 1000

    # start streaming
    stop_start = input('Press enter to stop mounting T265 and start streaming')
    if stop_start != '':
        del l515, t265r, t265l, pyft, angler
        return
    print("Stop mounting T265 and start streaming")
    start_timestamp_ms = time.time() * 1000
    l515.start_streaming()
    pyft.start_streaming()
    angler.start_streaming()

    # collect data
    stop = input('Press enter to stop streaming')
    if stop != '':
        del l515, t265r, t265l, pyft, angler
        return
    print("Stop streaming")

    # stop streaming
    l515_data = l515.stop_streaming()
    t265r_data = t265r.stop_streaming()
    t265l_data = t265l.stop_streaming()
    pyft_data = pyft.stop_streaming()
    angler_data = angler.stop_streaming()

    # save data
    print("Start saving data")
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'stage_timestamp_ms.json'), 'w') as f:
        json.dump({
            't265_pose_start_timestamp_ms': t265_pose_start_timestamp_ms,
            't265_pose_end_timestamp_ms': t265_pose_end_timestamp_ms,
            'start_timestamp_ms': start_timestamp_ms
        }, f, indent=4)
    os.makedirs(os.path.join(args.save_path, 'l515'), exist_ok=True)
    l515.save_streaming(os.path.join(args.save_path, 'l515'), l515_data)
    os.makedirs(os.path.join(args.save_path, 't265r'), exist_ok=True)
    t265r.save_streaming(os.path.join(args.save_path, 't265r'), t265r_data)
    os.makedirs(os.path.join(args.save_path, 't265l'), exist_ok=True)
    t265l.save_streaming(os.path.join(args.save_path, 't265l'), t265l_data)
    os.makedirs(os.path.join(args.save_path, 'pyft'), exist_ok=True)
    shutil.copyfile(os.path.join(args.pyft_tare_path, "tare_pyft.json"), os.path.join(args.save_path, 'pyft', "tare_pyft.json"))
    pyft.save_streaming(os.path.join(args.save_path, 'pyft'), pyft_data)
    os.makedirs(os.path.join(args.save_path, 'angler'), exist_ok=True)
    angler.save_streaming(os.path.join(args.save_path, 'angler'), angler_data)
    print("Stop saving data")

    # destroy devices
    del l515, t265r, t265l, pyft, angler


if __name__ == '__main__':
    args = config_parse()
    main(args)

import os
import configargparse
import numpy as np
import tqdm
import h5py

from utils.transformation import delta_xyz, delta_quat


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--stage', type=str)
    parser.add_argument('--pc_mesh', action='store_true')
    parser.add_argument('--ft_coord', action='store_true')
    parser.add_argument('--num_o', type=int)
    parser.add_argument('--num_a', type=int)
    parser.add_argument('--num_aa', type=int)
    parser.add_argument('--pad_o', action='store_true')
    parser.add_argument('--pad_a', action='store_true')
    parser.add_argument('--pad_aa', action='store_true')
    parser.add_argument('--pyft_xyz_threshold', type=float)
    parser.add_argument('--pyft_quat_threshold', type=float)
    parser.add_argument('--pyft_f_threshold', type=float)
    parser.add_argument('--pyft_t_threshold', type=float)

    args = parser.parse_args()
    return args


def main(args):
    # general config
    data_path = args.data_path
    save_path = args.save_path
    stage = args.stage
    pc_mesh = args.pc_mesh
    ft_coord = args.ft_coord
    num_o = args.num_o
    num_a = args.num_a
    num_aa = args.num_aa
    pad_o = args.pad_o
    pad_a = args.pad_a
    pad_aa = args.pad_aa
    filter_thresholds = {
        'pyft_xyz_threshold': args.pyft_xyz_threshold,
        'pyft_quat_threshold': args.pyft_quat_threshold,
        'pyft_f_threshold': args.pyft_f_threshold,
        'pyft_t_threshold': args.pyft_t_threshold
    }
    hdf5_paths = []
    for trial_name in sorted(os.listdir(data_path)):
        if not os.path.isdir(os.path.join(data_path, trial_name)):
            continue
        for file_name in sorted(os.listdir(os.path.join(data_path, trial_name))):
            if not file_name.endswith('.hdf5'):
                continue
            if stage == 'all' or stage in file_name:
                hdf5_paths.append(os.path.join(data_path, trial_name, file_name))

    # hdf5 loop
    with h5py.File(save_path, 'w') as save_hdf5:
        save_hdf5_data_group = save_hdf5.create_group('data')
        save_hdf5_data_group.attrs['pc_mesh'] = pc_mesh
        save_hdf5_data_group.attrs['ft_coord'] = ft_coord
        save_hdf5_data_group.attrs['num_o'] = num_o
        save_hdf5_data_group.attrs['num_a'] = num_a
        save_hdf5_data_group.attrs['num_aa'] = num_aa
        save_hdf5_data_group.attrs['pad_o'] = pad_o
        save_hdf5_data_group.attrs['pad_a'] = pad_a
        save_hdf5_data_group.attrs['pad_aa'] = pad_aa
        for k, v in filter_thresholds.items():
            save_hdf5_data_group.attrs[k] = v

        total_samples = 0
        for hdf5_path in tqdm.tqdm(hdf5_paths):
            whole_demo_name = os.path.basename(os.path.dirname(hdf5_path))
            stage_name = os.path.splitext(os.path.basename(hdf5_path))[0]

            # load hdf5
            with h5py.File(hdf5_path, 'r') as data_hdf5:
                data_group = data_hdf5['data']
                num_samples = data_group.attrs['num_samples']

                data_o_group = data_group['o']
                data_o_attrs = dict(data_o_group.attrs)
                if pc_mesh:
                    l515_pc_xyzs_l515_o = data_o_group['l515_pc_xyzs_l515_mesh'][:].astype(np.float32)
                    l515_pc_rgbs_o = data_o_group['l515_pc_rgbs_mesh'][:].astype(np.float32)
                else:
                    l515_pc_xyzs_l515_o = data_o_group['l515_pc_xyzs_l515'][:].astype(np.float32)
                    l515_pc_rgbs_o = data_o_group['l515_pc_rgbs'][:].astype(np.float32)
                pyft_xyzs_l515_o = data_o_group['pyft_xyzs_l515'][:].astype(np.float32)
                pyft_quats_l515_o = data_o_group['pyft_quats_l515'][:].astype(np.float32)
                if ft_coord:
                    pyft_fs_o = data_o_group['pyft_fs_l515'][:].astype(np.float32)
                    pyft_ts_o = data_o_group['pyft_ts_l515'][:].astype(np.float32)
                else:
                    pyft_fs_o = data_o_group['pyft_fs_pyft'][:].astype(np.float32)
                    pyft_ts_o = data_o_group['pyft_ts_pyft'][:].astype(np.float32)
                angler_widths_o = data_o_group['angler_widths'][:].astype(np.float32)
            
            # delta filter
            if sum(filter_thresholds.values()) != 0:
                lidx, ridx = 0, 0
                selected_idxs = [0]
                while ridx < num_samples:
                    delta_pyft_xyz = delta_xyz(pyft_xyzs_l515_o[ridx], pyft_xyzs_l515_o[lidx])
                    delta_pyft_quat = delta_quat(pyft_quats_l515_o[ridx], pyft_quats_l515_o[lidx]) / np.pi * 180
                    delta_pyft_f = delta_xyz(pyft_fs_o[ridx], pyft_fs_o[lidx])
                    delta_pyft_t = delta_xyz(pyft_ts_o[ridx], pyft_ts_o[lidx])
                    if (delta_pyft_xyz > filter_thresholds['pyft_xyz_threshold'] and filter_thresholds['pyft_xyz_threshold'] != 0) or \
                        (delta_pyft_quat > filter_thresholds['pyft_quat_threshold'] and filter_thresholds['pyft_quat_threshold'] != 0) or \
                        (delta_pyft_f > filter_thresholds['pyft_f_threshold'] and filter_thresholds['pyft_f_threshold'] != 0) or \
                        (delta_pyft_t > filter_thresholds['pyft_t_threshold'] and filter_thresholds['pyft_t_threshold'] != 0):
                        selected_idxs.append(ridx)
                        lidx = ridx
                    ridx += 1
                l515_pc_xyzs_l515_o = l515_pc_xyzs_l515_o[selected_idxs]
                l515_pc_rgbs_o = l515_pc_rgbs_o[selected_idxs]
                pyft_xyzs_l515_o = pyft_xyzs_l515_o[selected_idxs]
                pyft_quats_l515_o = pyft_quats_l515_o[selected_idxs]
                pyft_fs_o = pyft_fs_o[selected_idxs]
                pyft_ts_o = pyft_ts_o[selected_idxs]
                angler_widths_o = angler_widths_o[selected_idxs]
                filter_ratio = len(selected_idxs) / num_samples
                print(filter_ratio)
                num_samples = len(selected_idxs)
            
            # save data
            demo_name = f'demo_{whole_demo_name}_{stage_name}'
            save_hdf5_demo_group = save_hdf5_data_group.create_group(demo_name)
            save_hdf5_demo_group.create_dataset('l515_pc_xyzs_l515', data=l515_pc_xyzs_l515_o, dtype=np.float32)
            save_hdf5_demo_group.create_dataset('l515_pc_rgbs', data=l515_pc_rgbs_o, dtype=np.float32)
            save_hdf5_demo_group.create_dataset('pyft_xyzs_l515', data=pyft_xyzs_l515_o, dtype=np.float32)
            save_hdf5_demo_group.create_dataset('pyft_quats_l515', data=pyft_quats_l515_o, dtype=np.float32)
            save_hdf5_demo_group.create_dataset('pyft_fs', data=pyft_fs_o, dtype=np.float32)
            save_hdf5_demo_group.create_dataset('pyft_ts', data=pyft_ts_o, dtype=np.float32)
            save_hdf5_demo_group.create_dataset('angler_widths', data=angler_widths_o, dtype=np.float32)

            # save sample
            o_idxs, a_idxs = [], []
            for current_idx in range(num_samples - 1):
                selected = True

                o_begin_idx = max(0, current_idx - num_o + 1)
                o_end_idx = min(num_samples, current_idx + 1)
                o_padding = num_o - (o_end_idx - o_begin_idx)
                if o_padding > 0:
                    if pad_o:
                        o_selected_idxs = [0] * o_padding + list(range(o_begin_idx, o_end_idx))
                    else:
                        selected = False
                else:
                    o_selected_idxs = list(range(o_begin_idx, o_end_idx))

                a_begin_idx = min(num_samples - 1, current_idx + 1)
                a_end_idx = min(num_samples, current_idx + 1 + num_a)
                a_padding = num_a - (a_end_idx - a_begin_idx)
                if a_padding > 0:
                    if pad_a and pad_aa:
                        if a_padding > num_a - num_aa:
                            selected = False
                        else:
                            a_selected_idxs = list(range(a_begin_idx, a_end_idx)) + [num_samples - 1] * a_padding
                    elif pad_a and not pad_aa:
                        a_selected_idxs = list(range(a_begin_idx, a_end_idx)) + [num_samples - 1] * a_padding
                    else:
                        selected = False
                else:
                    a_selected_idxs = list(range(a_begin_idx, a_end_idx))
                
                if selected:
                    o_idxs.append(o_selected_idxs)
                    a_idxs.append(a_selected_idxs)
            save_hdf5_demo_group.create_dataset('o', data=np.array(o_idxs, dtype=int), dtype=int)
            save_hdf5_demo_group.create_dataset('a', data=np.array(a_idxs, dtype=int), dtype=int)

            # save attributes
            save_hdf5_demo_group.attrs['num_samples'] = len(o_idxs)
            for k, v in data_o_attrs.items():
                save_hdf5_demo_group.attrs[k] = v
            total_samples += len(o_idxs)
            if sum(filter_thresholds.values()) != 0:
                save_hdf5_demo_group.attrs['filter_ratio'] = filter_ratio

        # save attributes
        save_hdf5_data_group.attrs['num_samples'] = total_samples
        print(f'{total_samples = }')


if __name__ == '__main__':
    args = config_parse()
    main(args)

data_dir='./peel_data'
save_dir='./post_peel_data'
data_list=`ls $data_dir`
for data_name in $data_list
do
data_path=$data_dir/$data_name
save_path=$save_dir/$data_name
echo $data_path, $save_path
python create_hdf5.py --data_path $data_path --save_path $save_path --clip_object --clip_pyft --clip_base --render_pyft_num 10000 --voxel_size 0.001 --pc_num 10000
done

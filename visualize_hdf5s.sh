hdf5_list=`ls ./post_peel_data/train/01/*.hdf5`
for hdf5_file in $hdf5_list
do
echo $hdf5_file
python visualize_hdf5.py --hdf5_path $hdf5_file --pc_mesh
done

python3 create_meshes_shinghei.py
python3 create_urdfs_shinghei.py
rosrun shape_servo_control collect_data_bimanual_physical_dvrk_shinghei.py --deformernet_data_path /home/baothach/Documents/shinghei_def_data --manipulation_point_data_path /home/baothach/Documents/shinghe_mani_data --save_data True
rosrun shape_servo_control collect_data_bimanual_physical_dvrk_shinghei.py --deformernet_data_path /home/baothach/Documents/shinghei_def_data --manipulation_point_data_path /home/baothach/Documents/shinghe_mani_data --save_data True --prim_name cylinder
python3 loop_deformernet_physical_dvrk_shinghei.py


cd /home/baothach/deformernet_core

cd single_deformernet
python3 process_data_single.py --obj_category {your_category}
python3 single_trainer_all_objects.py
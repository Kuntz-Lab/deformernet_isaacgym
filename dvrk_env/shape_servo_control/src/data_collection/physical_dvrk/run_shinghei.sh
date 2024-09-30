python3 create_meshes_shinghei.py
python3 create_urdfs_shinghei.py
rosrun shape_servo_control collect_data_bimanual_physical_dvrk_shinghei.py --deformernet_data_path /home/baothach/Documents/shinghei_def_data --manipulation_point_data_path /home/baothach/Documents/shinghe_mani_data --save_data True
rosrun shape_servo_control collect_data_bimanual_physical_dvrk_shinghei.py --deformernet_data_path /home/baothach/Documents/trash --manipulation_point_data_path /home/baothach/Documents/trash --save_data True --prim_name hemis
python3 loop_deformernet_physical_dvrk_shinghei.py

# rosrun shape_servo_control collect_data_bimanual_physical_dvrk_shinghei.py --deformernet_data_path /home/baothach/Documents/shinghei_def_data --manipulation_point_data_path /home/baothach/Documents/shinghe_mani_data --save_data True --prim_name hemis --stiffness 10k --obj_name 50


cd /home/baothach/deformernet_core/single_deformernet
python3 process_data_single.py --obj_category box_1kPa
python3 process_data_single.py --obj_category cylinder_1kPa
python3 single_trainer_all_objects.py

cd /home/baothach/deformernet_core/learn_manipulation_points
python3 process_data_dense_predictor_single.py --obj_category box_1kPa
python3 process_data_dense_predictor_single.py --obj_category cylinder_1kPa
python3 single_dense_predictor_trainer.py


rosrun shape_servo_control collect_data_bimanual_physical_dvrk.py --deformernet_data_path /home/baothach/Documents/shinghei_def_data_temp --manipulation_point_data_path /home/baothach/Documents/shinghe_mani_data_temp --save_data True

import sys
import os
import numpy as np
import timeit

pkg_path = "/home/baothach/dvrk_shape_servo"
os.chdir(pkg_path)

headless = "True"
start_time = timeit.default_timer()

prim_names = ["box", "cylinder"]
stiffnesses = ["1k"]  #["1k", "5k", "10k"] 
num_tissue = 100
for prim_name in prim_names:
    for stiffness in stiffnesses:
        object_category = f"{prim_name}_{stiffness}"
        deformernet_data_path = f"/home/baothach/Documents/shinghei_data/def/multi_{object_category}Pa/data" 
        manipulation_point_data_path = f"/home/baothach/Documents/shinghei_data/mani/multi_{object_category}Pa/data"
        for _ in range(0, 300):  # 500
            i = np.random.randint(0,num_tissue)
            os.system(f"rosrun shape_servo_control collect_data_bimanual_physical_dvrk_shinghei.py --headless {headless} "
                    f"--prim_name {prim_name} --stiffness {stiffness} --obj_name {i} --deformernet_data_path {deformernet_data_path} --manipulation_point_data_path {manipulation_point_data_path}  --save_data True")
        




print(f"Elapsed time (hours): {(timeit.default_timer() - start_time)/3600:.3f}")
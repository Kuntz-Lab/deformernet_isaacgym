import os
import pickle
import numpy as np



def get_random_balls(num_balls, xy_lower_bounds, xy_upper_bounds, min_separation):
    '''
    get num_balls random attachment points coordinates within the xy bounds. min_separation is the radius of the attachment point
    '''
    low_x, low_y = xy_lower_bounds
    high_x, high_y = xy_upper_bounds
    balls_xy = np.array([[np.random.uniform(low=low_x, high=high_x), np.random.uniform(low=low_y, high=high_y)]])

    i = 0
    while i < num_balls-1:
        x = np.random.uniform(low=low_x, high=high_x)
        y = np.random.uniform(low=low_y, high=high_y)
        new_ball = np.array([[x,y]])
        dists = np.linalg.norm(new_ball - balls_xy, axis=1)
        if np.sum(dists<min_separation)==0 and np.sum(y==balls_xy[:,1])==0:
            balls_xy = np.concatenate((balls_xy, new_ball), axis=0)
            i+=1

    return balls_xy

if __name__ == "__main__":
    mesh_name = "multi_box_1kPa" #multi_cylinders_1kPa
    shape_name = "box" #cylinder

    urdf_path = f"/home/baothach/sim_data/Custom_shinghei/Custom_urdf/physical_dvrk/{mesh_name}"
    os.makedirs(urdf_path,exist_ok=True)

    mesh_path = f"/home/baothach/sim_data/Custom_shinghei/Custom_mesh/physical_dvrk/{mesh_name}"
    mesh_relative_path = "/".join(mesh_path.split("/")[-3:])

    density = 100
    # youngs = 1e3
    poissons = 0.3
    scale = 1.0

    with open(os.path.join(mesh_path, f"primitive_dict_{shape_name}.pickle"), 'rb') as handle:
        data = pickle.load(handle)

    for i in range(1):
        object_name = shape_name + "_" + str(i)
        base_name = "base"+ "_" + str(i)
        height = data[object_name]["height"]
        width = data[object_name]["width"]
        thickness = data[object_name]["thickness"]
        youngs = round(data[object_name]["youngs"])
        attach_dist = data[object_name]["base_radius"]
        base_thickness =  data[object_name]["base_thickness"]

        cur_urdf_path = urdf_path + '/' + shape_name + "_" + str(i) + '.urdf'
        
        f = open(cur_urdf_path, 'w')
        # temp = width
        # width = height
        # height = temp

        max_num_balls = 3
        rand_num_balls = np.random.randint(low=1, high=max_num_balls+1)
        epsilon = min(height, width)/5
        balls_xy = get_random_balls(num_balls=rand_num_balls, xy_lower_bounds=[-(width/2),-(height/2-epsilon)], xy_upper_bounds=[width/4,height/2-epsilon], min_separation=attach_dist)
        balls_xyz = np.concatenate((balls_xy, np.ones((len(balls_xy), 1))*base_thickness/2), axis=1)
        print(len(balls_xyz))

        base_mesh_path = f"../../../{mesh_relative_path}"
        object_mesh_path = f"../../../{mesh_relative_path}"
        mesh_object_name = object_name

        print(base_mesh_path)

        # get urdf
        urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>    
        
        <robot name="{object_name}">
            <link name="{object_name}">    
                <fem>
                    <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                    <density value="{density}" />
                    <youngs value="{youngs}"/>
                    <poissons value="{poissons}"/>
                    <damping value="0.0" />
                    <attachDistance value="{attach_dist}"/>
                    <tetmesh filename="{os.path.join(object_mesh_path, mesh_object_name+".tet")}"/>
                    <scale value="{scale}"/>
                </fem>
            </link>
        """

        # for b in range(rand_num_balls):
            
        #     urdf_str +=f"""
        #         <link name="ball_{b}">
        #             <visual>
        #                 <origin xyz="{balls_xyz[b][0]} {balls_xyz[b][1]} {-(thickness+base_thickness)*scale/2+0.001:.5f}"/>              
        #                 <geometry>
        #                     <mesh filename="{os.path.join(base_mesh_path, base_name+".obj")}" scale="{scale} {scale} {scale}"/>
        #                 </geometry>
        #             </visual>
        #             <collision>
        #                 <origin xyz="{balls_xyz[b][0]} {balls_xyz[b][1]} {-(thickness+base_thickness)*scale/2+0.001:.5f}"/>           
        #                 <geometry>
        #                     <mesh filename="{os.path.join(base_mesh_path, base_name+".obj")}" scale="{scale} {scale} {scale}"/>
        #                 </geometry>
        #             </collision>
        #             <inertial>
        #                 <mass value="5000000000000"/>
        #                 <inertia ixx="10.0" ixy="0.0" ixz="0.0" iyy="10.0" iyz="0.0" izz="10.0"/>
        #             </inertial>
        #         </link>
                
        #         <joint name = "attach_{b}" type = "fixed">
        #             <origin xyz = "{0} {0} 0.0" rpy = "0 0 0"/>
        #             <parent link ="{object_name}"/>
        #             <child link = "ball_{b}"/>
        #         </joint>  

        #     """

        large_base_thickness = 0.001
        large_base_name = f"large_base_{i}"
        # add the large base to hold the back-half of the tissue still
        urdf_str +=f"""
                <link name="large_base">
                    <visual>
                        <origin xyz="{0} {-height/4 -attach_dist} {-(thickness+large_base_thickness)*scale/2+0.001:.5f}"/>              
                        <geometry>
                            <mesh filename="{os.path.join(base_mesh_path, large_base_name+".obj")}" scale="{scale} {scale} {scale}"/>
                        </geometry>
                    </visual>
                    <collision>
                        <origin xyz="{0} {-height/4 -attach_dist} {-(thickness+large_base_thickness)*scale/2+0.001:.5f}"/>           
                        <geometry>
                            <mesh filename="{os.path.join(base_mesh_path, large_base_name+".obj")}" scale="{scale} {scale} {scale}"/>
                        </geometry>
                    </collision>
                    <inertial>
                        <mass value="5000000000000"/>
                        <inertia ixx="10.0" ixy="0.0" ixz="0.0" iyy="10.0" iyz="0.0" izz="10.0"/>
                    </inertial>
                </link>
                
                <joint name = "attach_large_base" type = "fixed">
                    <origin xyz = "{0} {0} 0.0" rpy = "0 0 0"/>
                    <parent link ="{object_name}"/>
                    <child link = "large_base"/>
                </joint>  

            """


        urdf_str+="""
        </robot>
        """
        
        f.write(urdf_str)
        f.close()

         # save balls relative xyzs (x, y) are relative to center of the object, z is global
        data[object_name]['balls_relative_xyzs'] = balls_xyz
        with open(os.path.join(mesh_path, "primitive_dict_box.pickle"), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


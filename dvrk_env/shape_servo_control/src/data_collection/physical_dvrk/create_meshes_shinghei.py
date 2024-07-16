import numpy as np
import trimesh
import os
import pickle
import random
import sys
sys.path.append('../../')
from utils.mesh_utils import create_tet_mesh


def create_cylinder_mesh_datatset(save_mesh_dir, num_mesh=100, save_pickle=True, seed=0):
    np.random.seed(seed)
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        print(f"object {i}")
        min_dim, max_dim = 0.075, 0.15

        radius = np.random.uniform(low = min_dim/2, high = max_dim/2)
        height = 0.01 #np.random.uniform(low = 0.01, high = 0.02)
        mesh = trimesh.creation.cylinder(radius=radius, height=height)

        youngs_mean = 1000 #10000
        youngs_std = 200 #1000        
        youngs = np.random.normal(youngs_mean, youngs_std)

        shape_name = "cylinder"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet_mesh(save_mesh_dir, object_name)

        # attachment point mesh
        length = 2*radius
        base_radius = length/20
        base_thickness = 0.001
        mesh_base = trimesh.creation.cylinder(base_radius, height=base_thickness)
        base_name = f"base_{i}"
        mesh_base.export(os.path.join(save_mesh_dir, base_name+'.obj'))

        large_base_thickness = 0.001
        large_base = trimesh.creation.box((length/2-base_radius, length, large_base_thickness)) 
        large_base_name = f"large_base_{i}"
        x = np.random.uniform(low = -2, high = 2)
        while x <= 1e-3:
            x = np.random.uniform(low = -2, high = 2)
        y = np.random.uniform(low = -2, high = 2)
        origin_x = np.random.uniform(low = 0.005, high = -length/4)
        large_base = trimesh.intersections.slice_mesh_plane(mesh=large_base, plane_normal=[x,y,0], plane_origin=[origin_x,0,large_base_thickness/2], cap=True)

        large_base.export(os.path.join(save_mesh_dir, large_base_name+'.obj'))

        vis=True
        if vis:
            import copy
            coordinate_frame = trimesh.creation.axis()  
            coordinate_frame.apply_scale(0.2)    
            copied_mesh_obj = copy.deepcopy(mesh)
            T = trimesh.transformations.translation_matrix([0., 0, -0.1])
            copied_mesh_obj.apply_transform(T)
            meshes = [copied_mesh_obj, mesh_base, large_base]
            trimesh.Scene(meshes+[coordinate_frame]).show()
            trimesh.Scene(meshes).show()
        
        primitive_dict[object_name] = {'height': radius*2, 'width': radius*2, 'radius': radius, 'thickness': height, 'youngs': youngs, "base_radius":base_radius, "base_thickness":base_thickness}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_cylinder.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

def create_box_mesh_datatset(save_mesh_dir, type, num_mesh=100, save_pickle=True, seed=0):
    np.random.seed(seed) 
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        print(f"object {i}")

        thickness = 0.01 #np.random.uniform(low = 0.01, high = 0.02)

        min_dim, max_dim = 0.075, 0.15

        width = np.random.uniform(low = min_dim, high = max_dim)
        height = np.random.uniform(low = min_dim, high = max_dim)
        

        mesh = trimesh.creation.box((width, height, thickness))

        if type == '1k':
            youngs_mean = 1000
            youngs_std = 200        
        elif type == '5k':
            youngs_mean = 5000
            youngs_std = 1000  
        elif type == '10k':    
            youngs_mean = 10000
            youngs_std = 1000  

        youngs = np.random.normal(youngs_mean, youngs_std)

        shape_name = "box"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet_mesh(save_mesh_dir, object_name, mesh_extension='.stl')

        # attachment point mesh
        length = min(height, width)
        base_radius = length/20
        base_thickness = 0.001
        mesh_base = trimesh.creation.cylinder(base_radius, height=base_thickness)
        base_name = f"base_{i}"
        mesh_base.export(os.path.join(save_mesh_dir, base_name+'.obj'))


        
        large_base_thickness = 0.001
        large_base = trimesh.creation.box((width/2-base_radius, height, large_base_thickness)) 
        large_base_name = f"large_base_{i}"
        x = np.random.uniform(low = -2, high = 2)
        while x <= 1e-3:
            x = np.random.uniform(low = -2, high = 2)
        y = np.random.uniform(low = -2, high = 2)
        origin_x = np.random.uniform(low = 0.005, high = -width/4)
        large_base = trimesh.intersections.slice_mesh_plane(mesh=large_base, plane_normal=[x,y,0], plane_origin=[origin_x,0,large_base_thickness/2], cap=True)

        large_base.export(os.path.join(save_mesh_dir, large_base_name+'.obj'))

        vis=False
        if vis:
            import copy
            coordinate_frame = trimesh.creation.axis()  
            coordinate_frame.apply_scale(0.2)    
            copied_mesh_obj = copy.deepcopy(mesh)
            T = trimesh.transformations.translation_matrix([0., 0, -0.1])
            copied_mesh_obj.apply_transform(T)
            meshes = [copied_mesh_obj, mesh_base, large_base]
            trimesh.Scene(meshes+[coordinate_frame]).show()
            trimesh.Scene(meshes).show()
        
        primitive_dict[object_name] = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs, "base_radius":base_radius, "base_thickness":base_thickness}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_box.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 



## 1000-200, 5000-1000, 10000-1000

mesh_dir = "/home/baothach/sim_data/Custom_shinghei/Custom_mesh/physical_dvrk/multi_box_1kPa"
os.makedirs(mesh_dir, exist_ok=True)
create_box_mesh_datatset(mesh_dir, type='1k', num_mesh=1, seed=None) # seed=0

mesh_dir = "/home/baothach/sim_data/Custom_shinghei/Custom_mesh/physical_dvrk/multi_cylinders_1kPa"
os.makedirs(mesh_dir, exist_ok=True)
create_cylinder_mesh_datatset(mesh_dir,num_mesh=1)

# mesh_dir = "/home/baothach/sim_data/Custom/Custom_mesh/multi_hemis_10kPa"
# create_hemis_mesh_datatset(mesh_dir)





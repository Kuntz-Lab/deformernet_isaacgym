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
        height = np.random.uniform(low = 0.01, high = 0.015)
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

        ### when the object is rotated 90 degrees counterclockwise in the simulation
        # large_base_thickness = 0.001
        # large_base = trimesh.creation.box((length/2-base_radius, length, large_base_thickness)) 
        # large_base_name = f"large_base_{i}"
        # x = np.random.uniform(low = -2, high = 2)
        # while x <= 1e-3:
        #     x = np.random.uniform(low = -2, high = 2)
        # y = np.random.uniform(low = -2, high = 2)
        # origin_x = np.random.uniform(low = 0.005, high = -length/4)
        # large_base = trimesh.intersections.slice_mesh_plane(mesh=large_base, plane_normal=[x,y,0], plane_origin=[origin_x,0,large_base_thickness/2], cap=True)
        # large_base.export(os.path.join(save_mesh_dir, large_base_name+'.obj'))

        #### when the object is not rotated (same as world coordinate frame)
        large_base_thickness = 0.001
        large_base = trimesh.creation.box((length, length/2-base_radius, large_base_thickness)) 
        large_base_name = f"large_base_{i}"
        y = np.random.uniform(low = 0.1, high = 2)
        while y <= 1e-3:
            y = np.random.uniform(low = 0.1, high = 2)
        x = np.random.uniform(low = -2, high = 2)
        origin_y = np.random.uniform(low = -height/16, high = 0)
        large_base = trimesh.intersections.slice_mesh_plane(mesh=large_base, plane_normal=[-x,-y,0], plane_origin=[0,origin_y,large_base_thickness/2], cap=True)
        large_base.export(os.path.join(save_mesh_dir, large_base_name+'.obj'))

        # vis=True
        # if vis:
        #     import copy
        #     coordinate_frame = trimesh.creation.axis()  
        #     coordinate_frame.apply_scale(0.2)    
        #     copied_mesh_obj = copy.deepcopy(mesh)
        #     T = trimesh.transformations.translation_matrix([0., 0, -0.1])
        #     copied_mesh_obj.apply_transform(T)
        #     meshes = [copied_mesh_obj, mesh_base, large_base]
        #     trimesh.Scene(meshes+[coordinate_frame]).show()
        #     trimesh.Scene(meshes).show()
        
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

        thickness = np.random.uniform(low = 0.01, high = 0.015)

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


        ### when the object is rotated 90 degrees counterclockwise in the simulation
        # large_base_thickness = 0.001
        # large_base = trimesh.creation.box((width/2-base_radius, height, large_base_thickness)) 
        # large_base_name = f"large_base_{i}"
        # x = np.random.uniform(low = -2, high = 2)
        # while x <= 1e-3:
        #     x = np.random.uniform(low = -2, high = 2)
        # y = np.random.uniform(low = -2, high = 2)
        # origin_x = np.random.uniform(low = 0.005, high = -width/4)
        # large_base = trimesh.intersections.slice_mesh_plane(mesh=large_base, plane_normal=[x,y,0], plane_origin=[origin_x,0,large_base_thickness/2], cap=True)
        # large_base.export(os.path.join(save_mesh_dir, large_base_name+'.obj'))

        #### when the object is not rotated (same as world coordinate frame)
        large_base_thickness = 0.001
        large_base = trimesh.creation.box(( width, height/2-base_radius, large_base_thickness)) 
        large_base_name = f"large_base_{i}"
        y = np.random.uniform(low = 0.1, high = 2)
        while y <= 1e-3:
            y = np.random.uniform(low = 0.1, high = 2)
        x = np.random.uniform(low = -2, high = 2)
        #print("xy", x, y)
        origin_y = np.random.uniform(low = -height/16, high = 0)
        large_base = trimesh.intersections.slice_mesh_plane(mesh=large_base, plane_normal=[-x,-y,0], plane_origin=[0,origin_y,large_base_thickness/2], cap=True)
        large_base.export(os.path.join(save_mesh_dir, large_base_name+'.obj'))

        # vis=True
        # if vis:
        #     import copy
        #     coordinate_frame = trimesh.creation.axis()  
        #     coordinate_frame.apply_scale(0.2)    
        #     copied_mesh_obj = copy.deepcopy(mesh)
        #     T = trimesh.transformations.translation_matrix([0., 0, -0.1])
        #     copied_mesh_obj.apply_transform(T)
        #     meshes = [copied_mesh_obj, mesh_base, large_base]
        #     trimesh.Scene(meshes+[coordinate_frame]).show()
        #     trimesh.Scene(meshes).show()
        
        primitive_dict[object_name] = {'height': height, 'width': width, 'thickness': thickness, 'youngs': youngs, "base_radius":base_radius, "base_thickness":base_thickness}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_box.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_hemis_mesh_datatset(save_mesh_dir, num_mesh=100, save_pickle=True, seed=0):
    np.random.seed(seed)
    primitive_dict = {'count':0}
    for i in range(num_mesh):
        print(f"object {i}")
        
        youngs_mean = 10000
        youngs_std = 1000 
        min_dim, max_dim = 0.075, 0.15

        radius = np.random.uniform(low = min_dim/2, high = max_dim/2)
        ratio = np.random.uniform(low = 1.5, high = 4)
        origin = radius/(min_dim/2) * 0.01 
        youngs_mean = 10000 #10000
        youngs_std = 1000 #1000        
        youngs = np.random.normal(youngs_mean, youngs_std)

        mesh = trimesh.creation.icosphere(radius = radius)   # hemisphere
        mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=[0,0,1], plane_origin=[0,0,origin], cap=True)

        vertices_transformed = mesh.vertices * np.array([1./ratio,1,1])
        vertices_transformed[:,2] = vertices_transformed[:,2] - 2*origin
        mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh.faces)

        shape_name = "hemis"        
        object_name = shape_name + "_" + str(i)
        mesh.export(os.path.join(save_mesh_dir, object_name+'.stl'))
        create_tet_mesh(save_mesh_dir, object_name)

        #primitive_dict[object_name] = {'radius': radius, 'origin': origin, 'youngs': youngs}
        #primitive_dict['count'] += 1


        # attachment point mesh
        vertices = mesh.vertices
        height = np.max(vertices[:,1]) - np.min(vertices[:,1])
        width = np.max(vertices[:,0]) - np.min(vertices[:,0])
        thickness = np.max(vertices[:,2]) - np.min(vertices[:,2])

        length = min(height, width)
        base_radius = length/20
        base_thickness = 0.001
        mesh_base = trimesh.creation.cylinder(base_radius, height=base_thickness)
        base_name = f"base_{i}"
        mesh_base.export(os.path.join(save_mesh_dir, base_name+'.obj'))


        # #### when the object is not rotated (same as world coordinate frame)
        large_base_thickness = 0.001
        large_base = trimesh.creation.box((length, length/2-base_radius, large_base_thickness)) 
        large_base_name = f"large_base_{i}"
        y = np.random.uniform(low = 0.1, high = 2)
        while y <= 1e-3:
            y = np.random.uniform(low = 0.1, high = 2)
        x = np.random.uniform(low = -2, high = 2)
        origin_y = np.random.uniform(low = -height/16, high = 0)
        large_base = trimesh.intersections.slice_mesh_plane(mesh=large_base, plane_normal=[-x,-y,0], plane_origin=[0,origin_y,large_base_thickness/2], cap=True)
        large_base.export(os.path.join(save_mesh_dir, large_base_name+'.obj'))

        vis=False
        if vis:
            import copy
            coordinate_frame = trimesh.creation.axis()  
            coordinate_frame.apply_scale(0.2)    
            copied_mesh_obj = copy.deepcopy(mesh)
            T = trimesh.transformations.translation_matrix([0., 0, -0.0])
            copied_mesh_obj.apply_transform(T)
            meshes = [copied_mesh_obj, mesh_base, large_base]
            trimesh.Scene(meshes+[coordinate_frame]).show()
            trimesh.Scene(meshes).show()
        
        primitive_dict[object_name] = {'height': height, 'width': width, 'radius': radius, 'thickness': thickness, 'youngs': youngs, "base_radius":base_radius, "base_thickness":base_thickness, 'origin':origin}
        primitive_dict['count'] += 1
    
    if save_pickle == False:   
        return primitive_dict

    data = primitive_dict
    with open(os.path.join(save_mesh_dir, "primitive_dict_hemis.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 





## 1000-200, 5000-1000, 10000-1000

num_mesh = 100

# mesh_dir = "/home/baothach/sim_data/Custom_shinghei/Custom_mesh/physical_dvrk/multi_box_1kPa"
# os.makedirs(mesh_dir, exist_ok=True)
# create_box_mesh_datatset(mesh_dir, type='1k', num_mesh=num_mesh, seed=None) # seed=0

# mesh_dir = "/home/baothach/sim_data/Custom_shinghei/Custom_mesh/physical_dvrk/multi_cylinder_1kPa"
# os.makedirs(mesh_dir, exist_ok=True)
# create_cylinder_mesh_datatset(mesh_dir,num_mesh=num_mesh)

mesh_dir = "/home/baothach/sim_data/Custom_shinghei/Custom_mesh/physical_dvrk/multi_hemis_10kPa"
os.makedirs(mesh_dir, exist_ok=True)
create_hemis_mesh_datatset(mesh_dir)





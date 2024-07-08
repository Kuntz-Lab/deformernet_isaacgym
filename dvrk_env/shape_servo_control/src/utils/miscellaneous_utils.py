# import open3d
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle5 as pickle
import os
from .point_cloud_utils import pcd_ize




def get_object_particle_state(gym, sim, vis=False):
    from isaacgym import gymtorch
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    particles = particle_state_tensor.numpy()[:, :3]  
    
    if vis:
        pcd_ize(particles, vis=True)
    
    return particles.astype('float32')


def normalize_list(lst):
    minimum = min(lst)
    maximum = max(lst)
    value_range = maximum - minimum

    normalized_lst = [(value - minimum) / value_range for value in lst]

    return normalized_lst


def scalar_to_rgb(scalar_list, colormap='jet', min_val=None, max_val=None):
    if min_val is None:
        norm = plt.Normalize(vmin=np.min(scalar_list), vmax=np.max(scalar_list))
    else:
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.cm.get_cmap(colormap)
    rgb = cmap(norm(scalar_list))
    return rgb


def print_color(text, color="red"):

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

    if color == "red":
        print(RED + text + RESET)
    elif color == "green":
        print(GREEN + text + RESET)
    elif color == "yellow":
        print(YELLOW + text + RESET)
    elif color == "blue":
        print(BLUE + text + RESET)
    else:
        print(text)


def read_pickle_data(data_path):
    with open(data_path, 'rb') as handle:
        return pickle.load(handle)      

def write_pickle_data(data, data_path, protocol=3):
    with open(data_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=protocol)    
        

def read_youngs_value_from_urdf(urdf_file):
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    youngs_elem = root.find('.//fem/youngs')
    if youngs_elem is not None and 'value' in youngs_elem.attrib:
        return str(float(youngs_elem.attrib['value']))

    return None

def get_extents_object(tet_file):
    """Return [min_x, min_y, min_z], [max_x, max_y, max_z] for a tet mesh"""
    mesh_lines = list(open(tet_file, "r"))
    mesh_lines = [line.strip('\n') for line in mesh_lines]
    zs = []
    particles = []
    for ml in mesh_lines:
        sp = ml.split(" ")
        if sp[0] == 'v':
            particles.append([float(sp[j]) for j in range(1,4)])
                
    particles = np.array(particles)
    xs = particles[:,0]
    ys = particles[:,1]
    zs = particles[:,2]
    
    return [[min(xs), min(ys), min(zs)],\
            [max(xs), max(ys), max(zs)]]   
    
    
def print_lists_with_formatting(lists, decimals, prefix_str):
    print(prefix_str, end=' ')  # Print the prefix string followed by a space
    for lst in lists:
        print("[", end='')
        # Check if the iterable is not empty by checking its length
        if len(lst) > 0:
            for e in lst[:-1]:
                print(f"{e:.{decimals}f}" if isinstance(e, float) else e, end=', ')
            # Handle the last element to avoid a trailing comma
            print(f"{lst[-1]:.{decimals}f}" if isinstance(lst[-1], float) else lst[-1], end='] ')
        else:
            print("]", end=' ')
            
    print("\n")


def pad_rot_mat(rot_mat):
    """convert 3x3 to 4x4 rotation matrix for transformations package"""   
    new_mat = np.eye(4)  
    new_mat[:3,:3] = rot_mat  
    return new_mat 
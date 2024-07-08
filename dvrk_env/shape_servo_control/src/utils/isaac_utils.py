from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy
from isaacgym import gymtorch
from isaacgym import gymapi
from shape_servo_control.srv import *
import rospy
from typing import List

def setup_cam(gym, env, cam_width, cam_height, cam_pos, cam_target):
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height    
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
    return cam_handle, cam_props

def default_sim_config(gym, args):
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim_params.substeps = 4
    sim_params.dt = 1./60.
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 50
    sim_params.flex.relaxation = 0.7
    sim_params.flex.warm_start = 0.1
    sim_params.flex.shape_collision_distance = 5e-4
    sim_params.flex.contact_regularization = 1.0e-6
    sim_params.flex.shape_collision_margin = 1.0e-4
    sim_params.flex.deterministic_mode = True    
    # return gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params), sim_params

    gpu_physics = 0
    gpu_render = 0
    # if args.headless:
    #     gpu_render = -1
    return gym.create_sim(gpu_physics, gpu_render, sim_type,
                          sim_params), sim_params

def default_dvrk_asset(gym, sim):
    # dvrk asset
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.0005#0.0001

    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.max_angular_velocity = 40000.

    asset_root = "./src/dvrk_env"
    dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
    print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
    return gym.load_asset(sim, asset_root, dvrk_asset_file, asset_options)

def init_dvrk_joints(gym, env, dvrk_handles: List):
    for dvrk_handle in dvrk_handles:
        dvrk_dof_states = gym.get_actor_dof_states(env, dvrk_handle, gymapi.STATE_NONE)
        dvrk_dof_states['pos'][4] = 0.24
        dvrk_dof_states['pos'][8] = 1.5
        dvrk_dof_states['pos'][9] = 0.8
        gym.set_actor_dof_states(env, dvrk_handle, dvrk_dof_states, gymapi.STATE_POS)
        

# def init_dvrk_joints(gym, env, dvrk_handle):
#     dvrk_dof_states = gym.get_actor_dof_states(env, dvrk_handle, gymapi.STATE_NONE)
#     dvrk_dof_states['pos'][8] = 1.5
#     dvrk_dof_states['pos'][9] = 0.8
#     gym.set_actor_dof_states(env, dvrk_handle, dvrk_dof_states, gymapi.STATE_POS)

def isaac_format_pose_to_PoseStamped(body_states):
    ros_pose = PoseStamped()
    ros_pose.header.frame_id = 'world'
    ros_pose.pose.position.x = body_states["pose"]["p"]["x"]
    ros_pose.pose.position.y = body_states["pose"]["p"]["y"]
    ros_pose.pose.position.z = body_states["pose"]["p"]["z"]
    ros_pose.pose.orientation.x = body_states["pose"]["r"]["x"]
    ros_pose.pose.orientation.y = body_states["pose"]["r"]["y"]
    ros_pose.pose.orientation.z = body_states["pose"]["r"]["z"]
    ros_pose.pose.orientation.w = body_states["pose"]["r"]["w"]
    return ros_pose

def get_new_obj_pose(saved_object_states, num_recorded_poses, num_particles_in_obj):
    choice = np.random.randint(0, num_recorded_poses)   
    state = saved_object_states[choice*num_particles_in_obj : (choice+1)*num_particles_in_obj, :] 
    return state        # torch size (num of particles, 3)   


def isaac_format_pose_to_list(body_states):
    return [body_states["pose"]["p"]["x"], body_states["pose"]["p"]["y"], body_states["pose"]["p"]["z"],
        body_states["pose"]["r"]["x"], body_states["pose"]["r"]["y"], body_states["pose"]["r"]["z"], body_states["pose"]["r"]["w"]]

def fix_object_frame(object_world):
    object_world_fixed = deepcopy(object_world)
    object_size = [object_world.width, object_world.height, object_world.depth]
    quaternion =  [object_world_fixed.pose.orientation.x,object_world_fixed.pose.orientation.y,\
                                            object_world_fixed.pose.orientation.z,object_world_fixed.pose.orientation.w]  
    r = R.from_quat(quaternion)
    rot_mat = r.as_matrix()
    # print("**Before:", rot_mat)
    # x_axis = rot_mat[:3, 0]
    max_x_indices = []
    max_y_indices = []
    max_z_indices = []
    for i in range(3):
        column = [abs(value) for value in rot_mat[:, i]]
       
        if column.index(max(column)) == 0:
            max_x_indices.append(i)
        elif column.index(max(column)) == 1:
            max_y_indices.append(i)
        elif column.index(max(column)) == 2:
            max_z_indices.append(i)
    
    # print("indices: ", max_x_indices, max_y_indices, max_z_indices)
    if (not max_x_indices):
        z_values = [abs(z) for z in rot_mat[2, max_z_indices]]  
        max_z_idx = max_z_indices[z_values.index(max(z_values))]
        max_x_idx = max_z_indices[z_values.index(min(z_values))]
        max_y_idx = max_y_indices[0]
    elif (not max_y_indices):
        z_values = [abs(z) for z in rot_mat[2, max_z_indices]]  
        max_z_idx = max_z_indices[z_values.index(max(z_values))]
        max_y_idx = max_z_indices[z_values.index(min(z_values))]
        max_x_idx = max_x_indices[0]
    else:
        max_x_idx = max_x_indices[0]
        max_y_idx = max_y_indices[0]
        max_z_idx = max_z_indices[0]
        
    # print("indices:", max_x_idx, max_y_idx, max_z_idx)
    # x_values = [abs(x) for x in rot_mat[0, :]]
    # y_values = [abs(y) for y in rot_mat[1, :]]
    # z_values = [abs(z) for z in rot_mat[2, :]]
    # max_x_idx = x_values.index(max(x_values))
    # max_y_idx = y_values.index(max(y_values))
    # max_z_idx = z_values.index(max(z_values))

    fixed_x_axis = rot_mat[:, max_x_idx]
    fixed_y_axis = rot_mat[:, max_y_idx]
    fixed_z_axis = rot_mat[:, max_z_idx]
   
    fixed_rot_matrix = np.column_stack((fixed_x_axis, fixed_y_axis, fixed_z_axis))

    if (round(np.linalg.det(fixed_rot_matrix)) != 1):    # input matrices are not special orthogonal(https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_matrix.html)
        for i in range(3): 
            fixed_rot_matrix[i][0] = -fixed_rot_matrix[i][0]  # reverse x axis


    r = R.from_matrix(fixed_rot_matrix)
    fixed_quat = r.as_quat()
    print("**After:", fixed_rot_matrix)
    object_world_fixed.pose.orientation.x, object_world_fixed.pose.orientation.y, \
            object_world_fixed.pose.orientation.z, object_world_fixed.pose.orientation.w = fixed_quat
 

    r = R.from_quat(fixed_quat)
    print("matrix: ", r.as_matrix())

    
    object_world_fixed.width = object_size[max_x_idx]
    object_world_fixed.height = object_size[max_y_idx]
    object_world_fixed.depth = object_size[max_z_idx]

    return object_world_fixed

def get_pykdl_client(q_cur):
    '''
    get Jacobian matrix
    '''
    # rospy.loginfo('Waiting for service get_pykdl.')
    # rospy.wait_for_service('get_pykdl')
    # rospy.loginfo('Calling service get_pykdl.')
    try:
        pykdl_proxy = rospy.ServiceProxy('get_pykdl', PyKDL)
        pykdl_request = PyKDLRequest()
        pykdl_request.q_cur = q_cur
        pykdl_response = pykdl_proxy(pykdl_request) 
    
    except(rospy.ServiceException, e):
        rospy.loginfo('Service get_pykdl call failed: %s'%e)
    # rospy.loginfo('Service get_pykdl is executed.')    
    # print("np.reshape(pykdl_response.ee_pose_flattened, (4,4)", np.reshape(pykdl_response.ee_pose_flattened, (4,4)).shape)
    return np.reshape(pykdl_response.jacobian_flattened, tuple(pykdl_response.jacobian_shape)), np.reshape(pykdl_response.ee_pose_flattened, (4,4))
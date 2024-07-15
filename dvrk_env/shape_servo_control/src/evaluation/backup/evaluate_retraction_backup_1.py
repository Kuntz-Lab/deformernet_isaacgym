#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import

import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
import os
import numpy as np
import open3d
import transformations
from copy import deepcopy
import timeit
import pickle
import argparse


from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil

import rospy
from geometry_msgs.msg import PoseStamped, Pose

from core import Robot
from behaviors import MoveToPose, ResolvedRateControl
from utils.grasp_utils import GraspClient
from utils.isaac_utils import fix_object_frame, get_pykdl_client, init_dvrk_joints
from utils.miscellaneous_utils import get_object_particle_state, write_pickle_data, print_color, read_pickle_data
from utils.camera_utils import get_partial_pointcloud_vectorized, visualize_camera_views
from utils.point_cloud_utils import down_sampling, pcd_ize

from goal_plane import get_goal_plane


ROBOT_Z_OFFSET = 0.30
two_robot_offset = 0.86


def visualize_plane(plane_eq, x_range=[-0.1,0.2], y_range=0.6, z_range=0.15,num_pts = 10000):
    plane = []
    for i in range(num_pts):
        x = np.random.uniform(x_range[0], x_range[1])
        z = np.random.uniform(0.1, z_range)
        y = -(plane_eq[0]*x + plane_eq[2]*z + plane_eq[3])/plane_eq[1]
        if -y_range < y < 0:
            plane.append([x, y, z])     
    return plane   

def get_goal_projected_on_image(goal_pc, i):
    u_s =[]
    v_s = []
    for point in goal_pc:
        point = list(point) + [1]
        point = np.expand_dims(np.array(point), axis=0)
        point_cam_frame = point * np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], vis_cam_handles[0]))
        u_s.append(1/2 * point_cam_frame[0, 0]/point_cam_frame[0, 2])
        v_s.append(1/2 * point_cam_frame[0, 1]/point_cam_frame[0, 2])      
          
    centerU = vis_cam_width/2
    centerV = vis_cam_height/2    
    y_s = (centerU - np.array(u_s)*vis_cam_width).astype(int)
    x_s = (centerV + np.array(v_s)*vis_cam_height).astype(int)    
    return x_s, y_s


if __name__ == "__main__":

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="dvrk Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--object_type", "type": int, "default": 0, "help": "Type of bjects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"},
            {"name": "--headless", "type": bool, "default": False, "help": "headless mode"}])

    num_envs = args.num_envs
    
    # Configure sim
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    # sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
        sim_params.dt = 1./60.
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 10
        sim_params.flex.num_inner_iterations = 50
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.1
        sim_params.flex.shape_collision_distance = 5e-4
        sim_params.flex.contact_regularization = 1.0e-6
        sim_params.flex.shape_collision_margin = 1.0e-4
        sim_params.flex.deterministic_mode = True

    gpu_physics = 0
    gpu_render = 0
    sim = gym.create_sim(gpu_physics, gpu_render, sim_type, sim_params)


    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up ground
    gym.add_ground(sim, plane_params)


    # Create viewer
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()


    # Load robot assets
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, -two_robot_offset, ROBOT_Z_OFFSET) 
 
    pose_2 = gymapi.Transform()
    pose_2.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    pose_2.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.0001

    asset_root = "./src/dvrk_env"
    dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"

    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    if sim_type is gymapi.SIM_FLEX:
        asset_options.max_angular_velocity = 40000.

    print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
    dvrk_asset = gym.load_asset(sim, asset_root, dvrk_asset_file, asset_options)


    # Load kidney
    rigid_asset_root = "/home/baothach/sim_data/Custom/Custom_urdf"
    rigid_asset_file = "kidney_rigid.urdf"
    rigid_pose = gymapi.Transform()
    rigid_pose.p = gymapi.Vec3(0.00, 0.38-two_robot_offset, 0.03)
    rigid_pose.r = gymapi.Quat(0.0, 0.0, -0.707107, 0.707107)
    # rigid_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    # rigid_pose.r = gymapi.Quat( -0.7071068, 0.7071068, 0, 0 )
    # rigid_pose.r = gymapi.Quat(0.5, 0.5, 0.5, 0.5)  #kidney 2
    asset_options.thickness = 0.003 # 0.002
    rigid_asset = gym.load_asset(sim, rigid_asset_root, rigid_asset_file, asset_options)


    
    # Load soft objects' assets
    asset_root = "/home/baothach/sim_data/Custom/Custom_urdf"
    soft_asset_file = 'thin_tissue_layer_attached.urdf'
    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.035, 0.37-two_robot_offset, 0.081)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)  
    soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True
    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

        
    # set up the env grid
    num_envs = 1
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(np.sqrt(num_envs))
  

    # Cache some common handles for later use
    envs = []
    envs_obj = []
    dvrk_handles = []
    dvrk_handles_2 = []
    object_handles = []
    

    # Create environments and actors
    for i in range(num_envs):
        # Create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # Add dvrk_2
        dvrk_2_handle = gym.create_actor(env, dvrk_asset, pose_2, "dvrk_2", i, 2, segmentationId=11)        

        # add rigid obj
        rigid_actor = gym.create_actor(env, rigid_asset, rigid_pose, 'rigid', i, 0, segmentationId=11)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(env, rigid_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        
        # Add deformable object        
        # # env_obj = env
        env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs_obj.append(env_obj)                
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
        object_handles.append(soft_actor)

        dvrk_handles_2.append(dvrk_2_handle)


    # Set up position control mode for the robot
    dof_props_2 = gym.get_asset_dof_properties(dvrk_asset)
    dof_props_2["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props_2["stiffness"].fill(200.0)
    dof_props_2["damping"].fill(40.0)
    dof_props_2["stiffness"][8:].fill(1)
    dof_props_2["damping"][8:].fill(2)  
    vel_limits = dof_props_2['velocity']  
    print(f"Velocity limits: {vel_limits}")

    for env in envs:
        gym.set_actor_dof_properties(env, dvrk_handles_2[i], dof_props_2)
            

    # Camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(1, 0.5, 1)
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)
        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)


    # Camera for point cloud setup
    cam_positions = []
    cam_targets = []
    cam_handles = []
    cam_width = 256
    cam_height = 256
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height
    cam_positions.append(gymapi.Vec3(0.15, -0.6, 0.20))
    cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.05))
   
    for i, env_obj in enumerate(envs_obj):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])


    # Camera for visualization
    vis_cam_positions = []
    vis_cam_targets = []
    vis_cam_handles = []
    vis_cam_width = 400
    vis_cam_height = 400
    vis_cam_props = gymapi.CameraProperties()
    vis_cam_props.width = vis_cam_width
    vis_cam_props.height = vis_cam_height
    vis_cam_positions.append(gymapi.Vec3(0.15, 0.2-two_robot_offset, 0.2))
    vis_cam_targets.append(gymapi.Vec3(0.0, 0.4-two_robot_offset, 0.05))

    
    for i, env_obj in enumerate(envs_obj):
        vis_cam_handles.append(gym.create_camera_sensor(env_obj, vis_cam_props))
        gym.set_camera_location(vis_cam_handles[i], env_obj, vis_cam_positions[0], vis_cam_targets[0])
    

    '''
    Main simulation stuff starts from here
    '''
    rospy.init_node('shape_servo_control')


    # Initilize robots' joints
    init_dvrk_joints(gym, envs[0], [dvrk_handles_2[0]])  
    
    sample_count = 0
    frame_count = 0
    max_sample_count = 1000

    final_point_clouds = []
    final_desired_positions = []
    pc_on_trajectory = []
    poses_on_trajectory = []
    first_time = True
    save_intial_pc = True
    get_goal_pc = True
    state = "home"
    
    
    execute_count = 0
    max_execute_count = 3
    num_new_goal = 0
    max_num_new_goal = 10

    vis_frame_count = 0
    num_image = 0
    start_vis_cam = True
    prepare_vis_cam = True
    prepare_vis_goal_pc = True
    prepare_vis_shift_plane = True

    
    dc_client = GraspClient()
    # save_path = "/home/baothach/shape_servo_data/generalization/surgical_setup/plane_vis/1"
    


    # Set up DNN:
    import torch
    sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
    # from pointcloud_recon_2 import PointNetShapeServo, PointNetShapeServo2
    from architecture import DeformerNet

    device = torch.device("cuda")
    model = DeformerNet(normal_channel=False)
    model.load_state_dict(torch.load("/home/baothach/shape_servo_data/generalization/surgical_setup/weights/run1/epoch 150"))
    # model.load_state_dict(torch.load("/home/baothach/shape_servo_data/generalization/surgical_setup/weights/run2(on_ground)/epoch 128"))
    model.eval()


    
    start_time = timeit.default_timer()    

    close_viewer = False
    all_done = False

    robot = Robot(gym, sim, envs[0], dvrk_handles_2[0])
    segmentationId_dict = {"robot_2": 11}
    camera_args = [gym, sim, envs_obj[0], cam_handles[0], cam_props, 
                    segmentationId_dict, "deformable", None, 0.002, False, "cpu"]      

    constrain_plane = np.array([1, 1, 0, 0.45])  # 2
    shift_plane = constrain_plane.copy()
    shift_plane[3] += 0.03 

    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        t = gym.get_sim_time(sim)


        # if prepare_vis_cam:
        #     plane_points = visualize_plane(constrain_plane, num_pts=50000)
        #     plane_xs, plane_ys = get_goal_projected_on_image(plane_points, i, thickness = 0)
        #     valid_ind = []
        #     for t in range(len(plane_xs)):
        #         if 0 < plane_xs[t] < vis_cam_width and 0 < plane_ys[t] < vis_cam_height:
        #             valid_ind.append(t)
        #     plane_xs = np.array(plane_xs)[valid_ind]
        #     plane_ys = np.array(plane_ys)[valid_ind]
            


        #     prepare_vis_cam = False

        # if start_vis_cam: 
        #     if vis_frame_count % 20 == 0:
        #         gym.render_all_camera_sensors(sim)
        #         im = gym.get_camera_image(sim, envs_obj[i], vis_cam_handles[0], gymapi.IMAGE_COLOR).reshape((vis_cam_height,vis_cam_width,4))
        #         # goal_xs, goal_ys = get_goal_projected_on_image(data["full pcs"][1], i, thickness = 1)
                
        #         im[plane_xs, plane_ys, :] = [0,0,255,255]
             


        #         im = Image.fromarray(im)
                
        #         img_path =  os.path.join(save_path, "image" + f'{num_image:03}' + ".png")
                
        #         im.save(img_path)
        #         num_image += 1         
            
   

        #     vis_frame_count += 1


        if state == "home" :   
            frame_count += 1

            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk_2", "psm_main_insertion_joint"), 0.203)            
            
        
            
            if frame_count == 10:
                rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))

                if first_time:
                    frame_state = gym.get_actor_rigid_body_states(envs[i], object_handles[i], gymapi.STATE_POS)                
                    frame_state['pose']['p']['z'] -= 0.05                
                    gym.set_actor_rigid_body_states(envs[i], object_handles[i], frame_state, gymapi.STATE_ALL) 
                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles_2[i], gymapi.STATE_ALL))
                    first_time = False

                state = "generate preshape"
                
                frame_count = 0

                current_pc = get_object_particle_state(gym, sim)
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                pc_ros_msg = fix_object_frame(pc_ros_msg)
                

        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET,
                            0, 0.707107, 0.707107, 0]


            mtp_behavior = MoveToPose(target_pose, robot, sim_params.dt, 2)
            if mtp_behavior.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "reset"                
            else:
                rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                state = "move to preshape"
                # rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))


        if state == "move to preshape":         
            action = mtp_behavior.get_action()

            if action is not None:
                gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())      
                        
            if mtp_behavior.is_complete():
                state = "grasp object"   
                rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's fucking grasp it!!") 

        
        if state == "grasp object":             
            rospy.loginfo("**Current state: " + state)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk_2", "psm_tool_gripper1_joint"), -2.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk_2", "psm_tool_gripper2_joint"), -3.0)         

            g_1_pos = 0.35
            g_2_pos = -0.35
            dof_states = gym.get_actor_dof_states(envs[i], dvrk_handles_2[i], gymapi.STATE_POS)
            if dof_states['pos'][8] < 0.35:
                                       
                state = "get shape servo plan"
                    
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk_2", "psm_tool_gripper1_joint"), g_1_pos)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk_2", "psm_tool_gripper2_joint"), g_2_pos)         
        
                anchor_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles_2[i], gymapi.STATE_POS)[-3])
                start_vis_cam = True




        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)

            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles_2[i], gymapi.STATE_POS)[-3])
            print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] ) 

            current_pc = get_object_particle_state(gym, sim)
            pcd = pcd_ize(current_pc, color=[0,0,0]) 
            
            num_points = current_pc.shape[0]         
            if save_intial_pc:
                initial_pc = deepcopy(current_pc)
                # intial_pc_tensor = torch.from_numpy(np.swapaxes(intial_pc,0,1)).float() 
                save_initial_pc = False
            
            if get_goal_pc:
                delta = 0.00
                goal_pc_numpy = get_goal_plane(constrain_plane=constrain_plane, initial_pc=initial_pc)   
                pcd_goal = pcd_ize(goal_pc_numpy, color=[1,0,0])                      
                goal_pc = torch.from_numpy(np.swapaxes(goal_pc_numpy,0,1)).float() 
                # pcd_goal = open3d.geometry.PointCloud()
                # pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
                get_goal_pc = False

            current_pc = torch.from_numpy(np.swapaxes(current_pc,0,1)).float()     
            open3d.visualization.draw_geometries([pcd, pcd_goal])   

            with torch.no_grad():
                desired_position = model(current_pc.unsqueeze(0), goal_pc.unsqueeze(0))[0].detach().numpy()*(0.001)  
                # desired_position = model(intial_pc_tensor.unsqueeze(0), goal_pc.unsqueeze(0))[0].detach().numpy()*(0.001) 

            print("from model:", desired_position)
          
            delta_x = desired_position[0]   
            delta_y = desired_position[1] 
            delta_z = desired_position[2] 

          

            cartesian_pose = Pose()
            cartesian_pose.orientation.x = 0
            cartesian_pose.orientation.y = 0.707107
            cartesian_pose.orientation.z = 0.707107
            cartesian_pose.orientation.w = 0
            cartesian_pose.position.x = -current_pose["pose"]["p"]["x"] + delta_x
            cartesian_pose.position.y = -current_pose["pose"]["p"]["y"] + delta_y
            # cartesian_pose.position.z = current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + delta_z
            cartesian_pose.position.z = max(0.005- ROBOT_Z_OFFSET,current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + delta_z)
            dof_states = gym.get_actor_dof_states(envs[0], dvrk_handles_2[0], gymapi.STATE_POS)['pos']

            plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=dof_states)
            state = "move to goal"
            traj_index = 0

        if state == "move to goal":           
            # Does plan exist?
            if (not plan_traj):
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "get shape servo plan"
            else:            
                if frame_count % 10 == 0:
                    current_pc = get_object_particle_state(gym, sim)
                    num_points = current_pc.shape[0]                    
                    num_failed_points = len([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]]) 
                    rospy.logwarn(f"percentage passed: {1-float(num_failed_points)/float(num_points)}")
                    print("num, num failed:", num_points, num_failed_points)
                frame_count += 1

                # print(traj_index, len(plan_traj))
                dof_states = gym.get_actor_dof_states(envs[0], dvrk_handles_2[0], gymapi.STATE_POS)['pos']
                plan_traj_with_gripper = [plan+[0.15,-0.15] for plan in plan_traj]
                pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[0], dvrk_handles_2[0], pos_targets)                
                
                if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.01):
                    traj_index += 1 

        
                # if traj_index == 10 or traj_index == len(plan_traj):
                if traj_index == len(plan_traj):
                    traj_index = 0  
                    final_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles_2[i], gymapi.STATE_POS)[-3])
                    # print("***Final x, y, z: ", final_pose[" pose"]["p"]["x"], final_pose["pose"]["p"]["y"], final_pose["pose"]["p"]["z"] ) 
                    delta_x = -(final_pose["pose"]["p"]["x"] - anchor_pose["pose"]["p"]["x"])
                    delta_y = -(final_pose["pose"]["p"]["y"] - anchor_pose["pose"]["p"]["y"])
                    delta_z = final_pose["pose"]["p"]["z"] - anchor_pose["pose"]["p"]["z"]
                    print("delta x, y, z:", delta_x, delta_y, delta_z)
                    
                    state = "get shape servo plan" 
                    execute_count += 1

                    if execute_count >= max_execute_count:
                        rospy.logwarn("Shift goal plane")
                        new_goal = get_goal_plane(constrain_plane=constrain_plane, initial_pc=initial_pc, 
                                                  check=True, delta=delta, 
                                                  current_pc=get_partial_pointcloud_vectorized(*camera_args))
                        if new_goal is not None:
                            goal_pc_numpy = new_goal
                        # goal_pc_numpy = get_goal_plane(constrain_plane=constrain_plane, initial_pc=initial_pc, check=True, delta=delta, current_pc=[])
                        delta += 0.02
                        # delta = min(0.06, delta)
                        # shift_plane[3] = constrain_plane[3] + 0.03 + delta
                        # prepare_vis_goal_pc = True
                        # prepare_vis_shift_plane = True
                        
                        num_new_goal += 1
                        print("num_new_goal:", num_new_goal)
                        if goal_pc_numpy == 'success':
                            print("=====================SUCCESS================")
                            state = "get shape servo plan aaxa"
                        else:
                            goal_pc = torch.from_numpy(np.swapaxes(goal_pc_numpy,0,1)).float() 
                            execute_count = 0
                            frame_count = 0
                            state = "get shape servo plan" 
                                   

        if state == "reset":   
            rospy.loginfo("**Current state: " + state) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            pos_targets = np.array([0.,0.,0.,0.,0.05,0.,0.,0.,1.5,0.8], dtype=np.float32)
            gym.set_actor_dof_position_targets(envs[i], dvrk_handles[i], pos_targets)
            gym.set_actor_dof_position_targets(envs[i], dvrk_handles_2[i], pos_targets)
            dof_states_1 = gym.get_actor_dof_states(envs[0], dvrk_handles[0], gymapi.STATE_POS)['pos']
            dof_states_2 = gym.get_actor_dof_states(envs[0], dvrk_handles_2[0], gymapi.STATE_POS)['pos']
            if np.allclose(dof_states_1, pos_targets, rtol=0, atol=0.1) and np.allclose(dof_states_2, pos_targets, rtol=0, atol=0.1):
                # print("Scuesfully reset robot")
                gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
                print("Scuesfully reset robot and object")
                pc_on_trajectory = []
                poses_on_trajectory = []
                state = "home"
 
  


        
        if sample_count == max_sample_count:             
            all_done = True    

        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)




    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')

import numpy as np
import rospy
from copy import deepcopy
import rospy
import transformations

from shape_servo_control.srv import *
from behaviors import Behavior
from core import RobotAction


class ResolvedRateControl(Behavior):
    '''
    Resolved-rate control behavior. Move the robot's end-effector to a desired pose.
    '''

    def __init__(self, delta_xyz, robot, dt, traj_duration, vel_limits=None, init_pose=None, 
                pos_threshold = 3e-3, ori_threshold=1e-1, use_euler_target=True, open_gripper=True):
        super().__init__()

        self.name = "task velocity control"
        self.robot = robot
        self.dt = dt
        self.traj_duration = traj_duration
        self.action = RobotAction()
        self.open_gripper = open_gripper
        self.err_thres = [pos_threshold]*3 + [ori_threshold]*3
        self.dq = 10**-5 * np.ones(self.robot.n_arm_dof)
        self.init_pose = init_pose
        self.vel_limits = vel_limits


        self.use_euler_target = use_euler_target
        self.set_target_pose(delta_xyz)


    def get_action(self):
       
        if self.is_not_started():
            self.set_in_progress()

        q_cur = self.robot.get_arm_joint_positions()
        J, curr_trans = self.get_pykdl_client(q_cur)
        desired_trans = self.target_pose

        delta_ee = self.computeError(curr_trans, desired_trans).flatten()
   
        if np.any(abs(delta_ee) > self.err_thres):
            J_pinv = self.damped_pinv(J)
            q_vel = np.matmul(J_pinv, delta_ee)
            desired_q_vel = q_vel * 2

            if self.vel_limits is not None:
                exceeding_ratios = abs(np.divide(desired_q_vel, self.vel_limits[:self.robot.n_arm_dof]))
                if np.any(exceeding_ratios > 1.0):
                    scale_factor = max(exceeding_ratios)
                    desired_q_vel /= scale_factor

            self.action.set_arm_joint_position(np.array(desired_q_vel, dtype=np.float32))           
            return self.action

        else:
            self.set_success()
            return None

    def set_target_pose(self, goal_pose):
        if self.init_pose is not None:
            pose = deepcopy(self.init_pose)
        else:
            if self.use_euler_target:
                pose = transformations.euler_matrix(*goal_pose[3:])
                pose[:3, 3] = goal_pose[:3]
            else:
                pose = np.eye(4)
                pose[:3,:3] = goal_pose[3]
                pose[:3, 3] = goal_pose[:3]                
                    
        self.target_pose = pose
     
    def damped_pinv(self, A, rho=0.017):
        AA_T = np.dot(A, A.T)
        damping = np.eye(A.shape[0]) * rho**2
        inv = np.linalg.inv(AA_T + damping)
        d_pinv = np.dot(A.T, inv)
        return d_pinv

    def null_space_projection(self, q_cur, q_vel, J, J_pinv):
        identity = np.identity(self.robot.n_arm_dof)
        q_vel_null = \
            self.compute_redundancy_manipulability_resolution(q_cur, q_vel, J)
        q_vel_constraint = np.array(np.matmul((
            identity - np.matmul(J_pinv, J)), q_vel_null))[0]
        q_vel_proj = q_vel + q_vel_constraint
        return q_vel_proj    

    def compute_redundancy_manipulability_resolution(self, q_cur, q_vel, J):
        m_score = self.compute_manipulability_score(J)
        J_prime,_ = self.get_pykdl_client(q_cur + self.dq)
        m_score_prime = self.compute_manipulability_score(J_prime)
        q_vel_null = (m_score_prime - m_score) / self.dq
        return q_vel_null

    def compute_manipulability_score(self, J):
        return np.sqrt(np.linalg.det(np.matmul(J, J.transpose())))    

    def get_pykdl_client(self, q_cur):
        '''
        get Jacobian matrix
        '''
        try:
            pykdl_proxy = rospy.ServiceProxy('get_pykdl', PyKDL)
            pykdl_request = PyKDLRequest()
            pykdl_request.q_cur = q_cur
            pykdl_response = pykdl_proxy(pykdl_request) 
        
        except(rospy.ServiceException, e):
            rospy.loginfo('Service get_pykdl call failed: %s'%e)

        return np.reshape(pykdl_response.jacobian_flattened, tuple(pykdl_response.jacobian_shape)), np.reshape(pykdl_response.ee_pose_flattened, (4,4))   

    def matrixLog(self, Ae):
        # if(np.abs(np.trace(Ae)-1)>2.0):
        #     Ae = np.copy(Ae)
        #     Ae[:,0] /= np.linalg.norm(Ae[:,0]) #unit vectorize because numerical issues
        #     Ae[:,1] /= np.linalg.norm(Ae[:,1])
        #     Ae[:,2] /= np.linalg.norm(Ae[:,2])
        # if np.trace(Ae)-1 == 0:
        # print("Ae:",Ae)
        
        # if abs(np.trace(Ae)+1) <= 0.0001:    
        #     phi = np.pi
        #     w = 1/np.sqrt(2*(1+Ae[2,2])) * np.array([Ae[0,2], Ae[1,2], 1+Ae[2,2]])
        #     skew = np.array([[0, -w[2], w[1]], 
        #                     [w[2], 0, -w[0]],
        #                     [-w[1], w[0], 0]])
        #     print("case 2 mat log")                
        # else:
        phi = np.arccos(0.5*(np.trace(Ae)-1))
        # print("phi:", phi)
        skew = (1/(2*np.sin(phi)))*(Ae-np.transpose(Ae))
        # print("trace:", np.trace(Ae))
        # print("phi:", phi)
        return skew,phi

    def computeError(self,currentTransform,desiredTransform):
        errorTransform = np.dot(desiredTransform, np.linalg.inv(currentTransform))
        
        linearError = errorTransform[:3,3:]
        linearError = desiredTransform[:3,3:] - currentTransform[:3,3:]
        # print(desiredTransform[:3,3:] - currentTransform[:3,3:])
        skew,theta = self.matrixLog(errorTransform[:3,:3])
        if(theta == 0.0):
            rotationError = np.zeros((3,1))
        else:
            w_hat = self.skewToVector(skew)
            rotationError = w_hat * theta
        
        G = 1/theta*np.eye(3) - 1/2*skew + (1/theta - 1/2*(1/np.tan(theta/2))) *(skew @ skew) 
        
        # return np.concatenate((rotationError, theta * G @ linearError))
        return np.concatenate((theta * G @ linearError, rotationError))


    def skewToVector(self, skew):
        w = np.zeros((3,1))
        w[0,0] = skew[2,1]
        w[1,0] = skew[0,2]
        w[2,0] = skew[1,0]
        return w

    def convert_to_matrix(self, p, quat):
        rot_mat = transformations.quaternion_matrix(quat)
        rot_mat[:3,3] = p
        return rot_mat


    def get_transform(self, curr_trans, desired_trans, get_twist=False):
        '''
        Returns:
        p: xyz displacement vector from initial to current pose
        R: 3x3 rotation matrix from initial to current pose
        '''

        p = desired_trans[:3,3:] - curr_trans[:3,3:]
        R = np.dot(desired_trans, np.linalg.inv(curr_trans))[:3,:3]
        
        if get_twist:
            twist = self.computeError(curr_trans, desired_trans)
            return p, R, twist

        return p, R




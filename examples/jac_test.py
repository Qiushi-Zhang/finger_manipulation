import mujoco 
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

import mujoco.msh2obj_test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pinocchio as pin 
import numpy as np 
import mujoco_viewer
# from utils import pin_utils
import pinocchio as pin 
from utils import planner
from utils import pin_utils
from utils import visualizer
# from pinocchio.visualize import MeshcatVisualizer
# import example_robot_data
finger0_path = os.path.join(base_dir, 'finger_descriptions/trifinger/nyu_finger_triple0.urdf')
finger1_path = os.path.join(base_dir, 'finger_descriptions/trifinger/nyu_finger_triple1.urdf')
finger2_path = os.path.join(base_dir, 'finger_descriptions/trifinger/nyu_finger_triple2.urdf')



finger0_model = pin.buildModelFromUrdf(finger0_path)
finger0_data = finger0_model.createData()
eeid0 = finger0_model.getFrameId("finger0_tip_link")

# setup Mujoco env and viewer
mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/finger_manipulation/finger_descriptions/trifinger/trifinger.xml")
mj_data = mujoco.MjData(mj_model)
viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data) 

mj_model.opt.timestep = 0.005
# build pin models 

x_des = np.array([0,0, 0.2])
v_des = np.zeros(3)

T_world = np.eye(4)

P = 200
D = 5

R = 0.05
h = 0.2

theta0 = np.pi/3
theta1 = np.pi
theta2 = 5*np.pi/3

q_init = np.array([0,np.pi/4, -np.pi/2]*3)
mj_data.qpos[:9] = q_init


def compute_point_jacobian(model, data, body_name, local_pos):
    """
    Compute the Jacobian for a point specified in local (body) coordinates.
    
    Args:
        model (mujoco.MjModel): Loaded MuJoCo model.
        data (mujoco.MjData): MuJoCo data associated with the model.
        body_name (str): Name of the body on which the point lies.
        local_pos (np.ndarray): 3D coordinates of the point in the body's local frame.
        
    Returns:
        (J_pos, J_rot):
            J_pos: (3 x nv) translational Jacobian (maps joint velocities to the point's linear velocity).
            J_rot: (3 x nv) rotational Jacobian (maps joint velocities to the point's angular velocity).
    """
    # Get body id from name
    body_id = data.body(body_name).id 
    
    # Allocate space for Jacobians (3 x nv flattened)
    jacp = np.zeros((3, model.nv)) # translational part
    jacr = np.zeros((3, model.nv))  # rotational part
    
    # Compute the translational and rotational Jacobians for the point
    mujoco.mj_jac(model, data, jacp, jacr, local_pos,body_id)
    
    # Reshape from 1D [3*nv] to 2D [3, nv]
    J_pos = jacp.reshape((3, model.nv))
    J_rot = jacr.reshape((3, model.nv))
    
    return J_pos, J_rot

x_des = pin_utils.forward_kinematics(finger0_model, finger0_data, eeid0, np.array([0,np.pi/4, -np.pi/2]))[:3,3]
v_des = np.zeros(3)

while mj_data.time < 20:
    
    q0, v0 = mj_data.qpos[:3], mj_data.qvel[:3]
    q1, v1 = mj_data.qpos[3:6], mj_data.qvel[3:6]
    q2, v2 = mj_data.qpos[6:9], mj_data.qvel[6:9]
    t = mj_data.time 

 
    
    
    T_disk = visualizer.get_T_mj(mj_model, mj_data, "disk")
    T_0 = pin_utils.forward_kinematics(finger0_model, finger0_data, eeid0, q0)
    x_0 = T_0[:3,3]

    J_pos, J_rot =compute_point_jacobian(mj_model, mj_data, "finger0_lower_link" , x_0)

    # print(J_pos[:,:3])
    J_ref = pin_utils.compute_jacobian(finger0_model, finger0_data, eeid0, q0)[:3,:]
    # print(J_ref)

    print(np.linalg.norm(J_ref-J_pos[:,:3]))


    tau = J_pos.T@(P*(x_des-x_0)+D*(v_des-J_pos@mj_data.qvel))
    tau = tau[:9]
    print(mj_model.nv, mj_model.nq, len(mj_data.ctrl))
    mj_data.ctrl = tau 


    mujoco.mj_step(mj_model, mj_data)
    

    # visualizer.add_marker(viewer, x_des0, 0.01)
    # print(T_disk[:3,3])
    # visualizer.visualize_frame(viewer, T_disk, 0.1)
    # visualizer.visualize_frame(viewer, T_world, 0.1)
    # print(mj_data.qpos[-1]/np.pi*180)
    viewer.render()


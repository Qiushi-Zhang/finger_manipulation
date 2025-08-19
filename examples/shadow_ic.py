import mujoco 
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

# import mujoco.msh2obj_test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pinocchio as pin 
import numpy as np 
import mujoco_viewer
# from utils import pin_utils
# import pinocchio as pin 
from utils import planner
# from utils import pin_utils
from utils import visualizer
# from pinocchio.visualize import MeshcatVisualizer
# import example_robot_data

# mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/mujoco_menagerie/shadow_hand/right_hand.xml")
mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/finger_manipulation/finger_descriptions/trifinger/force_control_env.xml")
mj_data = mujoco.MjData(mj_model)
viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data) 

mj_model.opt.timestep = 0.002


P = 400
D = 10

R = 0.05
h = 0.2

theta0 = np.pi/3
theta1 = np.pi
theta2 = 5*np.pi/3

q_init = np.array([0,np.pi/4, -np.pi/2]*3)
mj_data.qpos[:9] = q_init


def compute_point_jacobian(model, data, body_name, local_pos):

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


epsilon = 10e-4
K = 100
alpha = 5
while mj_data.time < 20:
    
    

    # mj_data.qvel= mj_data.qvel*0

    mujoco.mj_step(mj_model, mj_data)
    

    # visualizer.add_marker(viewer, x_des0, 0.01)
    # print(T_disk[:3,3])
    # visualizer.visualize_frame(viewer, T_disk, 0.1)
    # visualizer.visualize_frame(viewer, T_world, 0.1)
    # print(mj_data.qpos[-1]/np.pi*180)
    viewer.render()


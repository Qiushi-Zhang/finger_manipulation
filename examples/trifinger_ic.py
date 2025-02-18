import mujoco 
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

import mujoco.msh2obj_test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin 
import numpy as np 
import mujoco_viewer
from utils import pin_utils
from utils import planner
from utils import visualizer
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data
finger0_path = os.path.join(base_dir, 'finger_descriptions/trifinger/nyu_finger_triple0.urdf')
finger1_path = os.path.join(base_dir, 'finger_descriptions/trifinger/nyu_finger_triple1.urdf')
finger2_path = os.path.join(base_dir, 'finger_descriptions/trifinger/nyu_finger_triple2.urdf')


# setup Mujoco env and viewer
mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/finger_manipulation/finger_descriptions/trifinger/trifinger.xml")
mj_data = mujoco.MjData(mj_model)
viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data) 

mj_model.opt.timestep = 0.005
# build pin models 
finger0_model = pin.buildModelFromUrdf(finger0_path)
finger0_data = finger0_model.createData()
eeid0 = finger0_model. getFrameId("finger0_tip_link")

finger1_model= pin.buildModelFromUrdf(finger1_path)
finger1_data = finger1_model.createData()
eeid1 = finger1_model. getFrameId("finger1_tip_link")

finger2_model = pin.buildModelFromUrdf(finger2_path)
finger2_data = finger2_model.createData()
eeid2 = finger2_model. getFrameId("finger2_tip_link")

x_des = np.array([0,0, 0.2])
v_des = np.zeros(3)

T_world = np.eye(4)

P = 100
D = 2

R = 0.05
h = 0.2

theta0 = np.pi/3
theta1 = np.pi
theta2 = 5*np.pi/3

q_init = np.array([0,np.pi/4, -np.pi/2]*3)
mj_data.qpos[:9] = q_init

x_init0 = pin_utils.forward_kinematics(finger0_model, finger0_data, eeid0, q_init[:3])[:3,3]
x_init1 = pin_utils.forward_kinematics(finger1_model, finger1_data, eeid1, q_init[:3])[:3,3]
x_init2 = pin_utils.forward_kinematics(finger2_model, finger2_data, eeid2, q_init[:3])[:3,3]



while mj_data.time < 20:
    
    q0, v0 = mj_data.qpos[:3], mj_data.qvel[:3]
    q1, v1 = mj_data.qpos[3:6], mj_data.qvel[3:6]
    q2, v2 = mj_data.qpos[6:9], mj_data.qvel[6:9]
    t = mj_data.time 
    x_des0 = planner.compute_trajectory(0, t, R, x_init0, h)
    x_des1 = planner.compute_trajectory(1, t, R, x_init1, h)
    # x_des1 = x_init1
    x_des2 = planner.compute_trajectory(2, t, R, x_init2, h)
    # x_des2 = x_init2

    tau0 = pin_utils.impedance_controller(finger0_model, finger0_data, eeid0, q0, v0, x_des0, v_des, P, D)
    tau1 = pin_utils.impedance_controller(finger1_model, finger1_data, eeid1, q1, v1, x_des1, v_des, P, D)
    tau2 = pin_utils.impedance_controller(finger2_model, finger2_data, eeid2, q2, v2, x_des2, v_des, P, D)
    mj_data.ctrl = np.hstack((tau0, tau1, tau2))
  
    mujoco.mj_step(mj_model, mj_data)
    T_disk = visualizer.get_T_mj(mj_model, mj_data, "disk")
    # print(pin_utils.forward_kinematics(finger0_model, finger0_data, eeid0, q0)[:3,3])
    visualizer.add_marker(viewer, x_des0, 0.01)
    # print(T_disk[:3,3])
    visualizer.visualize_frame(viewer, T_disk, 0.1)
    visualizer.visualize_frame(viewer, T_world, 0.1)
    print(mj_data.qpos[-1]/np.pi*180)
    viewer.render()
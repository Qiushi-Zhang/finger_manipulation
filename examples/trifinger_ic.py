import mujoco 
import sys
import os
import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path)

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
import mujoco
import matplotlib.pyplot as plt

version_string = mujoco.mj_versionString()
print("MuJoCo version:", version_string)

# import mujoco.msh2obj_test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin 
import numpy as np 
import mujoco_viewer
from utils import pin_utils
from utils import planner
from utils import visualizer
from pinocchio.visualize import MeshcatVisualizer
# import example_robot_data
finger0_path = os.path.join(base_dir, 'finger_descriptions/trifinger/nyu_finger_triple0.urdf')
finger1_path = os.path.join(base_dir, 'finger_descriptions/trifinger/nyu_finger_triple1.urdf')
finger2_path = os.path.join(base_dir, 'finger_descriptions/trifinger/nyu_finger_triple2.urdf')


def compute_point_jacobian(model, data, body_name, local_pos):
    body_id = data.body(body_name).id 
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, local_pos, body_id)
    return jacp, jacr

# — First figure: torque errors (unchanged) —
plt.ion()  # turn on interactive mode
fig, ax = plt.subplots()
line0, = ax.plot([], [], label='τ[0]')
line1, = ax.plot([], [], label='τ[1]')
line2, = ax.plot([], [], label='τ[2]')
ax.set_xlim(0, 5)                # assuming simulation runs up to t=20s
ax.set_ylim(-0.5, .5)             # adjust to expected torque range
ax.set_xlabel('Time (s)')
ax.set_ylabel('Torque (Nm)')
ax.legend()

# — Second figure: contact‐force error & magnitudes —
fig2, ax2 = plt.subplots()
line_f_err, = ax2.plot([], [], label='‖F_ref − F‖')
line_f,     = ax2.plot([], [], label='‖F‖')
line_f_ref, = ax2.plot([], [], label='‖F_ref‖')
ax2.set_xlim(0, 5)
ax2.set_ylim(0, 3)   # adjust if needed
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Force magnitude (N)')
ax2.legend()

times              = []
taus               = [[], [], []]
cf_mags            = []
cfr_mags           = []
error_norms        = []            # lists to store torque history


# setup Mujoco env and viewer
mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/finger_manipulation/finger_descriptions/trifinger/trifinger.xml")
mj_data = mujoco.MjData(mj_model)
viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data) 
# "mjVIS_CONTACTPOINT" = show the contact points
# "mjVIS_CONTACTFORCE"  = draw vectors showing contact force direction and magnitude
viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
mj_model.vis.scale.forcewidth = 0.08   # thickness of force arrow
# mj_model.vis.scale.forceheight = 0.1 
mj_model.vis.scale.contactwidth = 0.03  # thickness of contact sphere
mj_model.vis.scale.contactheight = 0.5 # length of contact arrow


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

P = 300
D = 5

R = 0.05
h = 0.2

theta0 = np.pi/3
theta1 = np.pi
theta2 = 5*np.pi/3

q_init = np.array([0,np.pi/4, -np.pi/2]*3)
q_init_stay = np.array([0, np.pi/4, -np.pi/2]*2)
mj_data.qpos[:9] = q_init


x_init0 = pin_utils.forward_kinematics(finger0_model, finger0_data, eeid0, q_init[:3])[:3,3]
x_init1 = pin_utils.forward_kinematics(finger1_model, finger1_data, eeid1, q_init[:3])[:3,3]
x_init2 = pin_utils.forward_kinematics(finger2_model, finger2_data, eeid2, q_init[:3])[:3,3]





while mj_data.time < 20:
    
    q0, v0 = mj_data.qpos[:3], mj_data.qvel[:3]
    q1, v1 = mj_data.qpos[3:6], mj_data.qvel[3:6]
    q2, v2 = mj_data.qpos[6:9], mj_data.qvel[6:9]
    t = mj_data.time 
    
    q, v, a = mj_data.qpos[:3], mj_data.qvel[:3], mj_data.qacc[:3]
    tau = mj_data.ctrl[:3]
    tau_ref = pin.rnea(finger0_model, finger0_data, q,v,a)
    err = tau-tau_ref 



    x_des0 = planner.compute_trajectory(0, t, R, x_init0, h)
    x_des1 = planner.compute_trajectory(1, t, R, x_init1, h)
    # x_des1 = x_init1
    x_des2 = planner.compute_trajectory(2, t, R, x_init2, h)
    # x_des2 = x_init2


    times.append(t)
    taus[0].append(err[0])
    taus[1].append(err[1])
    taus[2].append(err[2])

    # --- update plot data ---
    line0.set_data(times, taus[0])
    line1.set_data(times, taus[1])
    line2.set_data(times, taus[2])
    ax.relim()               # recalculate limits
    ax.autoscale_view()      # autoscale
    fig.canvas.draw()
    fig.canvas.flush_events()


    if mj_data.ncon > 0:
        # 1) force+torque buffer
        ft_buf = np.zeros((6,1), dtype=np.float64)
        mujoco.mj_contactForce(mj_model, mj_data, 0, ft_buf)
        F_world = ft_buf[:3,0].copy()      # Fx, Fy, Fz

        # 2) position from mj_data.contact
        contact_struct = mj_data.contact[0]
        P_world = contact_struct.pos.copy()  # x, y, z in world frame

    else:
        F_world = np.zeros(3)
        P_world = np.zeros(3)


    J_pos, _ = compute_point_jacobian(mj_model, mj_data,
                                      "finger0_lower_link",
                                      P_world)
    J_pos = J_pos[:,:3]
    # print(J_pos.shape)
    # solve J^T F_ref = err  →  F_ref = (J^T)^{-1} err
    contact_force_ref = np.linalg.inv(J_pos.T).dot(err)

    # log magnitudes & error norm
    cf_mags.append(np.linalg.norm(F_world))
    cfr_mags.append(np.linalg.norm(contact_force_ref))
    error_norms.append(np.linalg.norm(contact_force_ref - F_world))

    # update second plot
    line_f.set_data(times, cf_mags)
    line_f_ref.set_data(times, cfr_mags)
    line_f_err.set_data(times, error_norms)
    ax2.relim(); ax2.autoscale_view()
    fig2.canvas.draw(); fig2.canvas.flush_events()




    viewer.render()

    tau0 = pin_utils.impedance_controller(finger0_model, finger0_data, eeid0, q0, v0, x_des0, v_des, P, D)
    tau1 = pin_utils.impedance_controller(finger1_model, finger1_data, eeid1, q1, v1, x_des1, v_des, P, D)
    tau2 = pin_utils.impedance_controller(finger2_model, finger2_data, eeid2, q2, v2, x_des2, v_des, P, D)
    mj_data.ctrl = np.hstack((tau0, tau1, tau2))

    times.append(t)
  
    mujoco.mj_step(mj_model, mj_data)
    T_disk = visualizer.get_T_mj(mj_model, mj_data, "disk")
    visualizer.add_marker(viewer, x_des0, 0.01)

    visualizer.visualize_frame(viewer, T_disk, 0.1)
    visualizer.visualize_frame(viewer, T_world, 0.1)

    
    
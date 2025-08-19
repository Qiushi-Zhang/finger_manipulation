#!/usr/bin/env python3
import mujoco
import mujoco_viewer
import numpy as np
import os, sys
print("mujoco version", mujoco.__version__)

# --------------------------------------------------------------------- #
# 0.  Your original set‑up (paths, model, viewer, gains, initial state)
# --------------------------------------------------------------------- #
XML_PATH = "/home/qiushi/workspace/finger_manipulation/finger_descriptions/trifinger/force_control_env.xml"

mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data  = mujoco.MjData(mj_model)
viewer   = mujoco_viewer.MujocoViewer(mj_model, mj_data)

mj_model.opt.timestep = 0.002

# initial finger configuration
q_init = np.array([0, np.pi/4, -np.pi/2] * 3)
mj_data.qpos[:9] = q_init

# --------------------------------------------------------------------- #
# 1.  Helper: build "sensor id  →  slice()" lookup once
# --------------------------------------------------------------------- #
sensor_slice = {}
adr = 0
for sid in range(mj_model.nsensor):
    n   = mj_model.sensor_dim[sid]   # number of floats this sensor writes
    sensor_slice[sid] = slice(adr, adr + n)
    adr += n

# --------------------------------------------------------------------- #
# 2.  Find every touch‑grid sensor (or any sensor you care about)
# --------------------------------------------------------------------- #
touch_grid_sensors = []
for sid in range(mj_model.nsensor):
    stype = mj_model.sensor_type[sid]
    # mjSENS_PLUGIN catches touch_grid (and any other plugin sensor)
    if stype == mujoco.mjtSensor.mjSENS_PLUGIN:
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
        touch_grid_sensors.append((sid, name))

if not touch_grid_sensors:
    raise RuntimeError("No touch‑grid plugin sensors found in the model!")

print("Touch‑grid sensors detected:")
for sid, name in touch_grid_sensors:
    print(f"  id={sid:2d}  name='{name}'  dim={mj_model.sensor_dim[sid]}")

# --------------------------------------------------------------------- #
# 3.  Main simulation loop
# --------------------------------------------------------------------- #
while mj_data.time < 20.0:

    mujoco.mj_step(mj_model, mj_data)

    # # ---- get forces from every grid ----------------------------------
    # for sid, name in touch_grid_sensors:
    #     block = mj_data.sensordata[sensor_slice[sid]]

    #     # Infer grid shape from sensor dim (channels = 3 → Fx,Fy,Fz)
    #     nvals   = mj_model.sensor_dim[sid]
    #     rows    = cols = int(round(np.sqrt(nvals / 3)))
    #     forces  = block.reshape(rows, cols, 3)   # (row, col, [Fz, Fx, Fy])

    #     # Re‑order to world (Fx, Fy, Fz) if you prefer
    #     Fx, Fy, Fz = forces[:, :, 1], forces[:, :, 2], forces[:, :, 0]

    #     if np.any(Fz > 0):            # print only when contact exists
    #         total_f = (Fx.sum(), Fy.sum(), Fz.sum())
    #         print(f"t={mj_data.time:6.3f}  {name}: ΣFx,ΣFy,ΣFz  = {total_f}")

    # ------------------------------------------------------------------
    for i in range(mj_data.ncon):           # iterate over all contacts
        con = mj_data.contact[i]

        # (optional) filter by geom or body id
        # if con.geom1 != fingertip_id and con.geom2 != plate_id: continue

        world_pos = con.pos           # 3-vector, metres
        frame     = con.frame.reshape(3, 3)    # contact frame

        # 6-vector force in contact frame
        cf = np.zeros(6)
        mujoco.mj_contactForce(mj_model,mj_data, i, cf)

        # convert to world coordinates if you like
        f_world = frame.T @ cf[:3]     # linear part
        τ_world = frame.T @ cf[3:]     # torque part

        print(f"t={mj_data.time:5.3f}  p={world_pos}  F={f_world} index={i}")

    viewer.render()

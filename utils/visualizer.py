import mujoco
import numpy as np



# Define a function to add a marker
def add_marker(viewer, pos, size=0.02, color=(1, 0, 0, 1)):
    viewer.add_marker(
        pos=pos,          # Position of the marker
        size=[size]*3,    # Marker size
        rgba=color,       # Marker color (r, g, b, a)
        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # Shape of the marker
    )
  


def visualize_ee(mj_data, viewer, frame_id):
    pos = mj_data.xpos[frame_id]
    add_marker(viewer,pos)





def visualize_frame(viewer, T, size=0.1):
   
    colors = {
        'x': (1, 0, 0, 1),
        'y': (0, 1, 0, 1),
        'z': (0, 0, 1, 1)
    }
    pos = T[:3,3]
    R = T[:3,:3]


    viewer.add_marker(
        pos=pos,                          # Start position of the arrow
        size=[0.01,0.01,0.2],               # (radius, length)
        rgba=[1,0,0,1],                       # Arrow color
        type=mujoco.mjtGeom.mjGEOM_ARROW, # Shape of the marker
        mat=R@np.array([[0,0,1],[0,1,0],[-1,0,0]])
    )
    viewer.add_marker(
        pos=pos,                          # Start position of the arrow
        size=[0.01,0.01,0.2],               # (radius, length)
        rgba=[0,1,0,1],                       # Arrow color
        type=mujoco.mjtGeom.mjGEOM_ARROW, # Shape of the marker
        mat=R@np.array([[1,0,0],[0,0,1],[0,-1,0]])
    )
    viewer.add_marker(
        pos=pos,                          # Start position of the arrow
        size=[0.01,0.01,0.2],               # (radius, length)
        rgba=[0,0,1,1],                       # Arrow color
        type=mujoco.mjtGeom.mjGEOM_ARROW, # Shape of the marker
        mat=R@np.array([[1,0,0],[0,1,0],[0,0,1]])
    )
    
def get_T_mj(model, data, body_name):
    body_id = model.body_parentid
    # print(body_id.shape)
    body_id = 10

    print(data.body("disk").id, body_id)
    
    # 2. Extract the position (xpos) and 3×3 rotation (xmat)
    #    MuJoCo stores them in flat arrays, so we reshape xmat.
    pos = data.xpos[body_id]               # shape (3,)
    mat_3x3 = data.xmat[body_id].reshape(3, 3)  # shape (3,3)
    
    # 3. Build the 4×4 transformation
    T = np.eye(4)
    T[:3, :3] = mat_3x3
    T[:3, 3] = pos
    
    return T


def visualize_trajectory(viewer, trajectory, color=(0, 1, 1, 1)):
    """
    Draws the trajectory as a series of connected line segments and markers in green.
    """
    # viewer.marker_count = 0 
    for i in range(trajectory.shape[1] - 1):
        # Draw line segments between consecutive points
        viewer.add_marker(
            pos=trajectory[:, i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0.02, 0.02],  # Sphere marker size
            rgba=color,
        )
        # viewer.add_marker(
        #     pos=trajectory[:, i + 1],
        #     type=mujoco.mjtGeom.mjGEOM_LINE,
        #     size=[0.01, 0.01, 0.01],  # Line thickness
        #     rgba=color,
        #     dir=trajectory[:, i + 1] - trajectory[:, i],
        # )





import pinocchio as pin
import numpy as np

def forward_kinematics(model,data,frame_id,q):
    pin.forwardKinematics(model,data,q)
    pin.updateFramePlacements(model,data)
    T = data.oMf[frame_id].homogeneous
    return T 

def compute_frame_err(T1,T2):
    T1 = pin.SE3(T1)
    T2 = pin.SE3(T2)
    err = pin.log(T1.actInv(T2)).vector

    
    return err

def compute_jacobian(model,data,frame_id,q):
    J = pin.computeFrameJacobian(model,data, q,frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J 


def impedance_controller(model, data, frame_id, q, v, x_des, v_des, P, D):
    T_ee = forward_kinematics(model,data,frame_id, q)
    x_ee = T_ee[:3,3]
    R_ee = T_ee[:3,:3]
    

    J_ee = compute_jacobian(model, data, frame_id, q) 

    V_ee = J_ee@v
    x_err = x_des- x_ee 
    # x_err = pin_utils.compute_frame_err(T_ee,T_goal)[:3]
   
    v_err = v_des - V_ee[:3]
    
    

    F = P * x_err + D * v_err 
    tau =  J_ee[:3,:].T @ F

    return tau 

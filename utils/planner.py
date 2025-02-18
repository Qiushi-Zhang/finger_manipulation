import numpy as np 
def compute_trajectory(i, t, R, x_init, h):
    x_des1 = np.array([(R+0.02)*np.cos(np.pi+i*2*np.pi/3),(R+0.02)*np.sin(np.pi+i*2*np.pi/3), h])
    x_des2 = np.array([(R-0.000)*np.cos(np.pi+i*2*np.pi/3),(R-0.000)*np.sin(np.pi+i*2*np.pi/3), h])
    x_des3 = np.array([(R-0.000)*np.cos(np.pi+i*2*np.pi/3+np.pi/6),(R-0.000)*np.sin(np.pi+i*2*np.pi/3+np.pi/6), h])
    x_des4 = np.array([(R+0.01)*np.cos(np.pi+i*2*np.pi/3+np.pi/6),(R+0.01)*np.sin(np.pi+i*2*np.pi/3+np.pi/6), h])
    if t < 1:
        x_des = x_init + t*(x_des1-x_init)
    if t >= 1 and t < 2:
        x_des = x_des1 + (t-1)*(x_des2-x_des1)
    if t >= 2 and t < 3: 
        theta = np.pi/6*(t-2)
        x_des = [(R-0.000)*np.cos(np.pi+i*2*np.pi/3+theta),(R-0.000)*np.sin(np.pi+i*2*np.pi/3+theta), h]
    if t >= 3 and t < 5: 
        x_des = x_des3 
    if t >= 5 and t < 7:
        x_des = x_des3 + (x_des4-x_des3)*(t-3)/2
        
    return x_des 
        

    
    
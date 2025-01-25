#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# import jax.numpy as np
from pr3_utils import *
from scipy.linalg import expm, block_diag
import scipy.linalg
from pr3_utils import *
from tqdm import tqdm
import cProfile
import pstats
# from jax import jit
import matplotlib.pyplot as plt


# In[2]:


filename = "../data/10.npz"
t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
  
R1 = [[1,0,0,0],
      [0,-1,0,0],
      [0,0,-1,0],
      [0,0,0,1]]

imu_T_cam = R1 @ imu_T_cam
cam_T_imu = np.linalg.inv(imu_T_cam)
# cam_T_imu = R1 @ cam_T_imu


# In[3]:


def skew_symmetric(v):
    """Convert a vector to a skew-symmetric matrix."""
    v = np.asarray(v).reshape(-1)
    
    # Case for a 3D vector: so(3) representation
    if v.size == 3:
        return np.array([[0, -v[2], v[1]], 
                         [v[2], 0, -v[0]], 
                         [-v[1], v[0], 0]])
    
    # Case for a 6D vector: se(3) representation
    elif v.size == 6:
        # The top-left 3x3 block is the skew-symmetric matrix for angular velocity.
        skew_sym = np.array([[0, -v[5], v[4]], 
                             [v[5], 0, -v[3]], 
                             [-v[4], v[3], 0]])
        # The top-right 3x1 block is the linear velocity.
        lin_vel = v[:3].reshape(-1, 1)
        # Combine them into the se(3) matrix.
        return np.block([[skew_sym, lin_vel], 
                         [np.zeros((1, 3)), 0]])
    else:
        raise ValueError("Input vector must be 3D or 6D.")


# In[4]:


def skew2(x):
    
    x = x.reshape(-1)
    return np.block([[skew_symmetric(x[3:]), skew_symmetric(x[:3])], [np.zeros((3, 3)), skew_symmetric(x[3:])]])


# In[5]:


def cdot(x):
    x = x.reshape(-1)
    return np.block([[np.eye(3), -skew_symmetric(x[:3])], [np.zeros((1, 6))]])


# In[6]:


def projection_(p):
    return p / p[2, :]


# In[7]:


import numpy as np

def derivative(a):
    """
    Compute the derivative of the perspective division with respect to the homogeneous coordinates of a point.

    Args:
        q (np.ndarray): The homogeneous coordinates of a point in the camera frame, shape (4,).

    Returns:
        np.ndarray: The Jacobian matrix of the perspective division operation, shape (4, 4).
    """
    # Ensure q is a column vector and extract its components for readability
    a = a.reshape(-1)  # Ensure q is a flat array
    x, y, z, w = a
    
    # Calculate the common factor for the derivative terms related to x and y
    common_factor = 1 / z
    
    # Construct the Jacobian matrix
    J = np.array([
        [common_factor, 0, -x / (z * z), 0],
        [0, common_factor, -y / (z * z), 0],
        [0, 0, 0, 0],
        [0, 0, -w / (z * z), common_factor]
    ])
    
    return J


# In[8]:



def initialize_landmarks(z, valid_obs, T_op_frame):
    global already_intialized 
    global n_landmarks 
    global max_index_initialized 
    global landmarks_pos
    global M
    # print("intialize_landmarks function called successfully")
    # Identify columns in z that do not sum up to -4 (assuming -1 for each non-observed element)
    # valid_observations = np.any(z != -1, axis=0)
    # valid_obs = np.array(np.where(z.sum(axis=0) > -4))

    # print("valid_obs before mark", valid_obs)

    valid_obs = valid_obs[~(already_intialized[valid_obs])]


    n_new_land = valid_obs.size
    # print("n_new_land", n_new_land)
    if n_new_land > 0:
        T_world_frame = np.linalg.inv(T_op_frame)

        already_intialized[valid_obs] = True
        # print("It is updated here")
        z = z[:,valid_obs]
        points_world = np.ones((4, n_new_land))
        points_world[0, :] = (z[0, :] - M[0, 2]) * b / (z[0, :] - z[2, :])
        points_world[1, :] = (z[1, :] - M[1, 2]) * (-M[2, 3]) / (M[1, 1] * (z[0, :] - z[2, :]))
        points_world[2, :] = -M[2, 3] / (z[0, :] - z[2, :])
        points_world = T_world_frame @ points_world
        landmarks_pos[valid_obs,:] = points_world[:3, :].T

        no_initialized_landmarks = np.sum(already_intialized)
        max_index_initialized = max(valid_obs.max()+1, max_index_initialized)
        # print("max_index_initialized", max_index_initialized)
        
        

    
    # Placeholder for actual landmark initialization logic
    # For example, you might want to store initial positions for new landmarks
    # Note: Actual initialization might require more context, such as camera calibration parameters


# In[9]:


def predict_z(valid_obs,T_op_frame):
    global landmarks_pos
    global M
    no_obs = valid_obs.size
    temp_land_pos = np.hstack([landmarks_pos[valid_obs, :], np.ones((no_obs, 1))])
    zp = M @ projection_(T_op_frame @ temp_land_pos.T)

    return zp.reshape(-1,1, order='F')
    


# In[10]:



def update(z,T_op_frame, T_next, cam_T_imu1):

     
    global landmarks_pos
    global max_index_initialized 
    global M
    global P_matrix 
    global V 

    valid_obs = np.array(np.where(z.sum(axis=0) > -4), dtype=np.int32).reshape(-1)
    no_landmarks_seen = valid_obs.size
    if no_landmarks_seen > 0:
        initialize_landmarks(z, valid_obs, T_op_frame)




        #-------------- H matrix ----------------------------------------------------------------------------------------------------------------------------------
        # no_updates = max_index_initialized
        no_updates = no_landmarks_seen
        Proj = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        
        upd_landmarks_pos = np.hstack([landmarks_pos[valid_obs,:],np.ones((no_landmarks_seen,1))])
        H = np.zeros((no_landmarks_seen*4,no_landmarks_seen*3 + 6))
        

        for i in range(no_landmarks_seen):
            # landmark_pos_homogeneous = np.hstack([landmarks_pos[valid_obs, :], [1]])
            # H_block = M @ derivative(T_op_frame @ upd_landmarks_pos[i,:].reshape(-1, 1)) @ T_op_frame
            # H[i*4:(i+1)*4, i*3:(i+1)*3] = M @ derivative(T_op_frame @ upd_landmarks_pos[i,:].reshape(-1, 1)) @ T_op_frame
            H[i*4:(i+1)*4, 6+i*3 : 6+ (i+1)*3] = M @ derivative((T_op_frame @ upd_landmarks_pos[i,:].reshape(-1,1)))  @ T_op_frame @ Proj.T

       
            # H[i*4:(i+1)*4, valid_obs[i]*3:(valid_obs[i]+1)*3] = M @ derivative(T_op_frame @ upd_landmarks_pos[i,:].reshape(-1,1)) @ T_op_frame @ Proj.T
        
        for i in range(no_landmarks_seen):
            H[i * 4:(i + 1) * 4, :6] = M @ derivative(T_op_frame @ upd_landmarks_pos[i, :].reshape(-1, 1)) @ cam_T_imu1 @ cdot(
                T_next@ upd_landmarks_pos[i, :].reshape(-1, 1))


        #-------------- landmark_positions and covariance -----------------------------------------------------------------------------------------------------
        # no_updates = max_index_initialized
        no_updates = no_landmarks_seen
        # print("no_updates", no_updates)
        # print("no_landmarks_seen", no_landmarks_seen)
        temp2_landmarks_pos = landmarks_pos[valid_obs, :]
        # P = P_matrix[:no_updates*3, :no_updates*3]
        # expanded_indices = np.hstack([valid_obs*3 + i for i in range(3)])
        # expanded_indices = np.concatenate([np.arange(idx * 3, (idx + 1) * 3) for idx in valid_obs])
        # expanded_indices = np.concatenate([np.arange((valid_obs[k]*3) , ((valid_obs[k]+1) * 3)) for k in range(6+no_landmarks_seen)])

        # print(expanded_indices)
        # print(np.ix_(expanded_indices, expanded_indices))
        # Extract the submatrix from P_matrix corresponding to the observed landmarks
        # P = P_matrix[np.ix_(expanded_indices, expanded_indices)]

        # Create an array of indices for the first 6 rows and columns
        first_indices = np.arange(6)
        # Create an array of indices for the landmarks seen
        landmark_indices = np.concatenate([np.arange(idx * 3+6, (idx + 1) * 3+6) for idx in valid_obs])

        # Concatenate the indices arrays
        expanded_indices = np.concatenate([first_indices, landmark_indices])
        # print(expanded_indices)
        P = P_matrix[np.ix_(expanded_indices, expanded_indices)]
        # P = P_matrix[:6+no_landmarks_seen*3,:6+no_landmarks_seen*3]
        # print("H shape",H.shape)
        # print("P shape",P.shape)

        #--------------Predict----------------------------------------
        z_pred = predict_z(valid_obs, T_op_frame)
        z = z[:, valid_obs].reshape(-1, 1, order='F')
        # print(z - z_pred)

        # print("before np.kron and no of landmarks seen", no_landmarks_seen)
        # print(V)
        noise_obs = np.kron(np.eye(no_landmarks_seen), V)
        # print("after np.kron")
        # print("before kalman operations and no of landmarks seen", no_landmarks_seen)
        cross_corr = P @ H.T
        # kalman_gain = np.linalg.solve((H @ cross_corr + noise_obs).T, cross_corr.T).T
        # H_PHT = H @ cross_corr
        # to_inv_ = H_PHT + noise_obs
        # inv_ = np.linalg.pinv(to_inv_)
        kalman_gain = cross_corr @ np.linalg.pinv((H @ cross_corr + noise_obs))
        # kalman_gain = cross_corr @ inv_
        update1 = kalman_gain @ (z-z_pred)
        temp2_landmarks_pos += update1[6:,0].reshape(-1,3)
        # print((kalman_gain @ (z - z_pred)).reshape(-1,3))
        P_updated = (np.eye(kalman_gain.shape[0]) - kalman_gain @ H) @ P
        # P_updated =  (P_updated+P_updated.T)/2
        # print(P_updated)
        # print("after kalman operations")
        
        #---------------------Update values--------------------------------------   
        # no_updates = max_index_initialized
        no_updates = no_landmarks_seen

        T_next = scipy.linalg.expm(skew_symmetric(update1[:6,0])) @ T_next

        landmarks_pos[valid_obs,:] = temp2_landmarks_pos
       
        P_matrix[np.ix_(expanded_indices, expanded_indices)] = P_updated
        # P = P_matrix[:6+no_landmarks_seen*3,:6+no_landmarks_seen*3] = P_updated
        # print("P_matrix[expanded_indices][:, expanded_indices]", P_matrix[expanded_indices][:, expanded_indices])
        # print(P_matrix)
        # print(valid_obs)
        # print(P_matrix)
    return T_next

        


# In[11]:


land_cov =np.eye(3) * 5e-3
position_cov = np.eye(6) * 1e-3
# position_cov = np.diag([1e-3,1e-3,1e-3,1e-4,1e-4,1e-4])
# position_cov[:2, :2] = np.eye(2) * 1e-6  # Assign small values for x and y
# position_cov[2, 2] = 1e-8  # Assign a small value for theta
obsv_noise_cov = np.eye(4) * 100 
# process_noise_cov = np.eye(6) * 1e-3
process_noise_cov = np.diag([1e-1,1e-2,1e-2,1e-5,1e-5,1e-4])
# process_noise_cov[0, 0] = 1e-3
# process_noise_cov[1, 1] = 1e-3
# process_noise_cov[5, 5] = 1e-5
# new_landmarks
n_landmarks = features.shape[1]
already_intialized = np.zeros(n_landmarks, dtype=bool)
landmarks_pos = np.zeros((n_landmarks, 3)) #xm
no_initialized_landmarks = 0
max_index_initialized = 0
# P_matrix = np.kron(np.eye(n_landmarks),land_cov)
P_matrix = np.block([[position_cov, np.zeros((6, 3 * n_landmarks))],
                           [np.zeros((3 * n_landmarks, 6)),
                            np.kron(np.eye(n_landmarks), land_cov)]])

V = obsv_noise_cov
W = process_noise_cov

imu_T_cam = imu_T_cam
M = np.block([[K[:2, :], np.array([[0, 0]]).T], [K[:2, :], np.array([[-K[0, 0] * b, 0]]).T]])


# In[12]:


# P1 = np.block([[position_cov, np.zeros((6, 3 * 2))],
#                            [np.zeros((3 * 2, 6)),
#                             np.kron(np.eye(2), land_cov)]])
# P1


# In[13]:


# Initial pose and covariance

# global max_index_initialized
# max_index_initialized = 0
n_landmarks = features.shape[1]
already_intialized = np.zeros(n_landmarks, dtype=bool)
landmarks_pos = np.zeros((n_landmarks, 3)) #xm
no_initialized_landmarks = 0
max_index_initialized = 0

T_prev = np.eye(4)
poses = [T_prev]
u = np.vstack([linear_velocity, angular_velocity])


for i in tqdm(range(1, linear_velocity.shape[-1])):
    dt = t[0, i] - t[0, i-1]
    # omega = angular_velocity[:, i]
    # v = linear_velocity[:, i]
    
    # Predict the next pose
    # M_exp = se3_exp(omega, v, dt)
    M_exp = scipy.linalg.expm(-dt * skew_symmetric(u[:, i]))
    T_next = M_exp @ T_prev
    
    # T_next = T_prev @ M_exp 

    M_exp1 = scipy.linalg.expm(-dt * skew2(u[:, i]))
    P_matrix[:6,:] = M_exp1 @ P_matrix[:6,:]
    P_matrix[:,:6] = P_matrix[:,:6] @ M_exp1.T 
    P_matrix[:6,:6] += W
    T_op_frame = cam_T_imu @ T_prev
    
   
    
    # # Create a Profile object
    # profiler = cProfile.Profile()
    # # Start profiling
    # profiler.enable()
    

    T_upd = update(features[:, :, i], T_op_frame, T_next, cam_T_imu)
    T_prev = T_upd

    # print(landmarks_pos)
    # landmarks_current = landmark_upd
    # P_current = P_upd
    # print(P_matrix)

    # profiler.disable()
    # # Create Stats object
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # # Print the statistics
    # stats.print_stats()

    
    poses.append(np.linalg.inv(T_prev))
    
    # print(landmarks_pos[:,0])
    # break
    # if(i==2):
    #     break

    if (i + 1) % 5 == 0:
        visualize_trajectory_2d(np.stack(poses,-1), landmarks_pos, path_name="IMU Trajectory", show_ori=True, save_fig_name=f'./fig/0027/{i}.jpg',show_navigation=True)

    

    # Fx = jacobian_se3_exp(omega, v, dt)

    # P_next = Fx @ P_prev @ Fx.T + Q
    # T_prev, P_prev = T_next, P_next

    
# print(poses)

# Convert poses to a format suitable for visualization
poses = np.stack(poses,axis=-1) 
# plt.figure
# plt.scatter(landmarks_pos[:,0], landmarks_pos[:,1])
visualize_trajectory_2d(poses, landmarks_pos, path_name="IMU Trajectory", show_ori=True)


# In[14]:


# poses = np.stack(poses,axis=-1) 
# plt.figure
# plt.scatter(landmarks_pos[:,0], landmarks_pos[:,1])
# visualize_trajectory_2d(poses, path_name="IMU Trajectory", show_ori=True)
# poses = np.stack(poses,axis=-1) 
# # plt.figure
# plt.scatter(landmarks_pos[:,0], landmarks_pos[:,1])
# plt.xlim([-1500, 800])
# plt.ylim([-1000, 600])
# %matplotlib notebook
visualize_trajectory_2d(poses, landmarks_pos, path_name="IMU Trajectory", show_ori=True)
print(P_matrix)


# In[15]:


plt.figure
plt.scatter(landmarks_pos[:,0], landmarks_pos[:,1])
plt.plot(poses[0,3,:],poses[1,3,:],'r-',label='IMU_Trajectory')
plt.scatter(poses[0,3,0],poses[1,3,0],marker='s',label="start")
plt.scatter(poses[0,3,-1],poses[1,3,-1],marker='o',label="end")


# In[16]:


visualize_trajectory_2d(poses, landmarks_pos, path_name="IMU Trajectory", show_ori=True)


# In[17]:


print(P_matrix)


# In[18]:


import numpy as np

# Save poses to a file
np.save('poses_VISLAM.npy', poses)

# Save landmarks positions to a file
np.save('landmarks_pos_VISLAM.npy', landmarks_pos)


# In[19]:


import numpy as np
# import jax.numpy as np
from pr3_utils import *
from scipy.linalg import expm, block_diag
import scipy.linalg
from pr3_utils import *
from tqdm import tqdm
import cProfile
import pstats
# from jax import jit
import matplotlib.pyplot as plt
# Load poses from file

poses_VISLAM = np.load('poses_LSLAM.npy')

# Load landmarks positions from file
landmarks_pos_VISLAM = np.load('landmarks_pos_LSLAM.npy')

poses_LSLAM = np.load('poses_VISLAM.npy')

# Load landmarks positions from file
landmarks_pos_LSLAM = np.load('landmarks_pos_VISLAM.npy')

plt.figure
# plt.scatter(landmarks_pos_LSLAM[:,0], landmarks_pos_LSLAM[:,1])
plt.plot(poses_LSLAM[0,3,:],poses_LSLAM[1,3,:],label='IMU_Trajectory')
plt.scatter(poses_LSLAM[0,3,0],poses_LSLAM[1,3,0],label="start")
plt.scatter(poses_LSLAM[0,3,-1],poses_LSLAM[1,3,-1],label="end")
# plt.scatter(landmarks_pos_VISLAM[:,0], landmarks_pos_VISLAM[:,1])
plt.plot(poses_VISLAM[0,3,:],poses_VISLAM[1,3,:],label='SLAM_Trajectory')
plt.scatter(poses_VISLAM[0,3,0],poses_VISLAM[1,3,0])
plt.scatter(poses_VISLAM[0,3,-1],poses_VISLAM[1,3,-1])

# plt.set_xlabel('x')
# plt.set_ylabel('y')
plt.axis('equal')
plt.grid(False)
plt.legend()


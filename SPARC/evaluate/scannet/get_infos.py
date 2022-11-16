from glob import glob
from socket import INADDR_ALLHOSTS_GROUP
import cv2
import json
import os
import scipy.ndimage
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rot
from tqdm import tqdm

def make_dir_check(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def make_M_from_tqs(t, q, s):
    # q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    # R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    q = [q[1],q[2],q[3],q[0]]
    R[0:3, 0:3] = scipy_rot.from_quat(q).as_matrix()
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M,R[0:3, 0:3],S[0:3, 0:3]

def load_4by4_from_txt(path):
    M = np.zeros((4,4))
    with open(path,'r') as f:
        content = f.readlines()
        for i in range(4):
            line = content[i].split()
            for j in range(4):
                M[i,j] = np.float32(line[j])
        return M

def get_scene_pose(R_and_T,scene_trs):
    # R_and_T = load_4by4_from_txt(dir_path + scene['id_scan'] + '/pose/' + frame + '.txt')
    R_pose = R_and_T[:3,:3].copy()
    T_pose = R_and_T[:3,3].copy()

    T_pose = np.concatenate((T_pose,np.ones((1))),axis=0)
    

    # Mscene,_,_ = make_M_from_tqs(scene["trs"]['translation'],scene["trs"]['rotation'],scene["trs"]['scale'])
    Mscene,_,_ = make_M_from_tqs(scene_trs['translation'],scene_trs['rotation'],scene_trs['scale'])

    T_scene_pose = np.matmul(Mscene,T_pose)[:3]

    R_scene_pose = np.matmul(Mscene[:3,:3],R_pose).copy()
    R_scene_pose = np.linalg.inv(R_scene_pose)
    
    T_final_pose = - np.matmul(R_scene_pose,T_scene_pose)
    R_final_pose = np.linalg.inv(R_scene_pose)

    return R_scene_pose,T_final_pose



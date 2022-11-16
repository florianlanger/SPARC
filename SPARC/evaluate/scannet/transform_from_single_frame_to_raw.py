import csv
import json
from unicodedata import category
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from SPARC.evaluate.scannet.get_infos import get_scene_pose
import quaternion
from SPARC.evaluate.scannet import SE3,CSVHelper
import sys
from operator import itemgetter

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)


def category_to_id(category):
    top = {}
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"

    inv_map = {v: k for k, v in top.items()}
    return inv_map[category]

def transform_single_raw_invert_own_no_invert_roca(per_frame,scan2cad_full_annotations,output_path,scannet_poses,not_invert_previous_predictions=True):
    print('Transform single raw')

    idscan2trs = {}
    for r in scan2cad_full_annotations:
        id_scan = r["id_scan"]
        idscan2trs[id_scan] = r["trs"]

    scores_frame = []
    for image in per_frame:
        for detection in per_frame[image]:
            scores_frame.append(str(detection["score"]))

    # scores_raw = list(df["object_score"])

    all_infos = {}
    counter = 0
    for image in tqdm(per_frame):
        # if image != 'scene0653_00/color/001300.jpg':
        #     continue
        if counter > 100000:
            break
        for detection in per_frame[image]:
            counter += 1
            


            # index_raw = scores_raw.index(scores_frame[0])

            for scene_info in scan2cad_full_annotations:
                if scene_info["id_scan"] == image.split('/')[0]:
                    break

            if type(detection["scene_cad_id"]) == list:
                cad_id = detection["scene_cad_id"][-1]
            else:
                cad_id = detection["scene_cad_id"]

            if cad_id == None:
                continue

            models = scene_info['aligned_models']
            for model in models:
                if model['id_cad'] == cad_id:
                    object_center = model['center']


            scene = image.split('/')[0]
            frame = image.split('/')[2].replace('.jpg','.txt')

            scene_trs = scene_info["trs"]
            R_and_T = np.array(scannet_poses[scene + '_' + frame])
            R_scene_pose,T_scene_pose = get_scene_pose(R_and_T,scene_trs)


            # if detection["own_prediction"] == True:
            #     invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
            if not_invert_previous_predictions == True and detection["own_prediction"] == False:
                invert = np.array([[1,0,0],[0,1,0],[0,0,1.]])
            else:
                invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
            # print('Enable inversion')
            # Get T 
            t_start = detection['t']
            # T = np.array(t_start)
            T = np.matmul(invert,t_start)
            t_cad = np.matmul(np.linalg.inv(R_scene_pose),T - T_scene_pose)

            # print('t_cad',t_cad)


            # GET R
            q = np.quaternion(detection['q'][0], detection['q'][1], detection['q'][2], detection['q'][3])
            R = quaternion.as_rotation_matrix(q)
            R = np.matmul(invert,R)
            R = np.matmul(np.linalg.inv(R_scene_pose),R)
            q = quaternion.from_rotation_matrix(R)
            r_cad = quaternion.as_float_array(q)

            # IDEA go from predictions back to kind of like T,R,S of CAD model, thans apply same transformation as in EvalBenchmark script

            # scan_tran = -np.array(idscan2trs[id_scan]["translation"])
            # Mscan = SE3.compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])
            Mscan = SE3.compose_mat4(idscan2trs[scene]["translation"], idscan2trs[scene]["rotation"], idscan2trs[scene]["scale"])
            Mcad = SE3.compose_mat4(t_cad, r_cad, detection['s'])#,-np.array(object_center))

            t_pred, q_pred, s_pred = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))
            # print('t pred',t_pred)
            q_pred = quaternion.as_float_array(q_pred)

            # t_final = [df.iloc[index_raw]['tx'],df.iloc[index_raw]['ty'],df.iloc[index_raw]['tz']]
            # q_final = [df.iloc[index_raw]['qw'],df.iloc[index_raw]['qx'],df.iloc[index_raw]['qy'],df.iloc[index_raw]['qz']]
            # s_final = [df.iloc[index_raw]['sx'],df.iloc[index_raw]['sy'],df.iloc[index_raw]['sz']]

            infos = (scene,category_to_id(detection['category']),cad_id,t_pred[0],t_pred[1],t_pred[2],q_pred[0],q_pred[1],q_pred[2],q_pred[3],s_pred[0],s_pred[1],s_pred[2],detection['score'],image,detection["detection"])
            # infos = (scene,category_to_id(detection['category']),cad_id,t_pred[0],t_pred[1],t_pred[2],q_pred[0],q_pred[1],q_pred[2],q_pred[3],s_pred[0],s_pred[1],s_pred[2],detection['score'])
            
            if scene not in all_infos:
                all_infos[scene] = []

            all_infos[scene].append(infos)

    for scene in all_infos:
        all_infos[scene] = sorted(all_infos[scene], key=itemgetter(13),reverse=True)
        # all_infos[scene] = sorted(all_infos[scene], key=itemgetter(-1),reverse=True)


    all_infos_list = []
    for scene in sorted(all_infos):
        for prediction in all_infos[scene]:
            all_infos_list.append(prediction)

    all_infos_list.insert(0,('id_scan','objectCategory','alignedModelId','tx','ty','tz','qw','qx','qy','qz','sx','sy','sz','object_score','image','detection'))
    CSVHelper.write(output_path,all_infos_list)

    
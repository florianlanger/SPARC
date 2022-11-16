import numpy as np
np.warnings.filterwarnings('ignore')
import pathlib
import subprocess
import os
import collections
import shutil
import quaternion
import operator
import glob
import csv
import re
from SPARC.evaluate.scannet import SE3,CSVHelper,JSONHelper
import argparse
np.seterr(all='raise')
import argparse
import json
from tqdm import tqdm
import sys



# helper function to calculate difference between two quaternions 
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:                                                                                                                                                                                                                                                                                                                      
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation

def alignment_row_to_info(alignment,cad2sym):
    
    catid_cad = str(alignment[0]).zfill(8)
    id_cad = alignment[1]

    if (catid_cad, id_cad) not in cad2sym:
        print((catid_cad, id_cad))
        sym = 'Object does not exist'
        print(alignment)
        print(sym)
    else:
        sym = cad2sym[(catid_cad, id_cad)]
    t = np.asarray(alignment[2:5], dtype=np.float64)
    q0 = np.asarray(alignment[5:9], dtype=np.float64)
    q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
    s = np.asarray(alignment[9:12], dtype=np.float64)
    score = np.asarray(alignment[12], dtype=np.float64)

    return catid_cad,id_cad,q,t,s,sym,score


def rot_error(q_1,q_2,sym):
    if sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [calc_rotation_diff(q_2, q_1*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [calc_rotation_diff(q_2, q_1*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [calc_rotation_diff(q_2, q_1*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    else:
        error_rotation = calc_rotation_diff(q_2, q_1)
    return error_rotation

def check_same_detections(alignment_1,alignment_2,threshold_translation,threshold_rotation,threshold_scale,cad2sym):
    # take alignment_1 to be the already detected 

    catid_cad_1,id_cad_1,q_1,t_1,s_1,sym_1,score_1 = alignment_row_to_info(alignment_1,cad2sym)
    catid_cad_2,id_cad_2,q_2,t_2,s_2,sym_2,score_2 = alignment_row_to_info(alignment_2,cad2sym)

    if sym_2 == 'Object does not exist':
        # return True so that object will be ignored
        print(dfd)
        return True

    assert score_2 <= score_1 + 0.000001, (catid_cad_1,id_cad_1,score_1,catid_cad_2,id_cad_2,score_2)

    if catid_cad_1 != catid_cad_2:
        return False

    else:
        error_translation = np.linalg.norm(t_1 - t_2, ord=2)
        # error_scale = 100.0*np.abs(np.mean(s_2/s_1) - 1)
        error_scale = 100.0*np.abs(np.mean(s_1/s_2) - 1)
        # error_rotation = rot_error(q_1,q_2,sym_2)
        error_rotation = rot_error(q_1,q_2,sym_1)

        cat_correct = str(catid_cad_1).zfill(8) == str(catid_cad_2).zfill(8)

        # if np.abs(float(alignment_2[12]) - 0.9909141063) < 0.00001:
        #     print(error_translation,error_scale,error_rotation)

        is_same_detection = error_translation <= threshold_translation and error_rotation <= threshold_rotation and error_scale <= threshold_scale and cat_correct
        # if np.abs(float(alignment_2[12]) - 0.9909141063) < 0.00001:
        #     print(is_same_detection)
        return is_same_detection


def cluster_3d(in_dir,out_dir,filename_annotations):

    # -> define Thresholds
    threshold_translation = 0.4 # <-- in meter
    threshold_rotation = 60 # <-- in deg
    threshold_scale = 60 # <-- in %
    # <-

    print('NOTE SCALE error depends which way round define it')
    print('Threshold translation: ',threshold_translation)
    print('Threshold rotation: ',threshold_rotation)
    print('Threshold scale: ',threshold_scale)

    cad2sym = {}

    for r in JSONHelper.read(filename_annotations):
        for model in r["aligned_models"]:
            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]
            cad2sym[(catid_cad, id_cad)] = model["sym"]
    # print(cad2sym)
    # for key in cad2sym:
    #     if key[0] == '02871439':
    #         print(key)
    #     if key[1] == 'd6da5457b0682e24696b74614952b2d0':
    #         print(key)

    for file0 in tqdm(glob.glob(in_dir + "/*.csv")):

        # if "scene0187_01" not in file0:
        #     continue

        # print(file0)
        filtered_objects = []
        alignments = CSVHelper.read(file0)
        id_scan = os.path.basename(file0.rsplit(".", 1)[0])
        for new_alignment in alignments: # <- multiple alignments of same object in scene
            # print(new_alignment)
            already_detected = False
            for existing_alignment in filtered_objects:
                already_detected = check_same_detections(existing_alignment,new_alignment,threshold_translation,threshold_rotation,threshold_scale,cad2sym)
                if already_detected:
                    break
            
            if not already_detected:
                filtered_objects.append(tuple(new_alignment))
        # print(len(filtered_objects),'/',len(alignments))
        # print('dont save')
        CSVHelper.write(file0.replace(in_dir,out_dir),filtered_objects)

if __name__ == "__main__":

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]

    scan2cad_dir = global_config["dataset"]["dir_path_images"] + '../scan2cad_annotations/'
    # in_dir = target_folder + '/global_stats/eval_scannet/results_per_scene'
    # out_dir = target_folder + '/global_stats/eval_scannet/results_per_scene_filtered'
    in_dir = target_folder + '/global_stats/eval_scannet/results_per_scene'
    out_dir = target_folder + '/global_stats/eval_scannet/results_per_scene_filtered'
    # print('FILTERING THRESHOLDS CHANGED ')
    cluster_3d(in_dir,out_dir,scan2cad_dir + "full_annotations.json")
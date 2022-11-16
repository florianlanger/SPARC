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
import sys



# get top8 (most frequent) classes from annotations. 
def get_top8_classes_scannet():                                                                                                                                                                                                                                                                                           
    top = collections.defaultdict(lambda : "other")
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"
    return top



def number_per_model_to_number_per_category(dic):
    new_dict = {}
    for key in dic:
        cat_id = key.split('_')[0]
        if cat_id in new_dict:
            new_dict[cat_id] += dic[key]
        else:
            new_dict[cat_id] = dic[key]
    return new_dict

def scan2cad_constraint(projectdir,outputdir,filename_cad_appearance, filename_annotations):

    appearances_cad = JSONHelper.read(filename_cad_appearance)

    appearances_cad_total = {}
    for scene in appearances_cad:
        appearances_cad_total[scene] = 0
        for model in appearances_cad[scene]:
            appearances_cad_total[scene] += appearances_cad[scene][model]

    benchmark_per_scan = collections.defaultdict(lambda : collections.defaultdict(lambda : 0)) # <-- benchmark_per_scan
    benchmark_per_class = collections.defaultdict(lambda : collections.defaultdict(lambda : 0)) # <-- benchmark_per_class

    catid2catname = get_top8_classes_scannet()
    
    groundtruth = {}
    cad2info = {}
    idscan2trs = {}
    
    testscenes = [os.path.basename(f).split(".")[0] for f in glob.glob(projectdir + "/*.csv")]
    
    testscenes_gt = []
    for r in JSONHelper.read(filename_annotations):
        id_scan = r["id_scan"]

        # NEED THIS
        if id_scan not in testscenes:
            continue

        testscenes_gt.append(id_scan)

        idscan2trs[id_scan] = r["trs"]
        
        for model in r["aligned_models"]:
            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]
            catname_cad = catid2catname[catid_cad]
            model["n_total"] = len(r["aligned_models"])
            groundtruth.setdefault((id_scan, catid_cad),[]).append(model)
            cad2info[(catid_cad, id_cad)] = {"sym" : model["sym"], "catname" : catname_cad}
            if catname_cad != 'other':
                benchmark_per_class[catname_cad]["n_total"] += 1
                benchmark_per_scan[id_scan]["n_total"] += 1

    for file0 in glob.glob(projectdir + "/*.csv"):

        alignments = CSVHelper.read(file0)
        id_scan = os.path.basename(file0.rsplit(".", 1)[0])
        if id_scan not in testscenes_gt:
            print('id_scan',id_scan)
            print('continue')
            continue
        benchmark_per_scan[id_scan]["seen"] = 1

        appearance_counter = {}
        kept_alignments = []

        appearance_per_cat = number_per_model_to_number_per_category(appearances_cad[id_scan])


        for alignment in alignments: # <- multiple alignments of same object in scene
        
            # -> read from .csv file
            catid_cad = str(alignment[0]).zfill(8)
            id_cad = alignment[1]
            cadkey = catid_cad + "_" + id_cad


            if catid_cad in appearance_per_cat:
                n_appearances_allowed = appearance_per_cat[catid_cad] # maximum number of appearances allowed
            else:
                n_appearances_allowed = 0
            appearance_counter.setdefault(catid_cad, 0)

            if appearance_counter[catid_cad] >= n_appearances_allowed:
                continue
            appearance_counter[catid_cad] += 1
            kept_alignments.append(tuple(alignment))


            
        CSVHelper.write(outputdir + id_scan + '.csv',kept_alignments)

                

        


if __name__ == "__main__":
    print('Scan2CAD constraints')

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    scan2cad_dir = global_config["dataset"]["dir_path_images"] + '../scan2cad_annotations/'
    # projectdir = global_config["general"]["target_folder"] + '/global_stats/eval_scannet/results_per_scene_filtered_no_rotation/'
    # outputdir = global_config["general"]["target_folder"] + '/global_stats/eval_scannet/results_per_scene_scan2cad_constraints_filtered_no_rotation/'

    projectdir = global_config["general"]["target_folder"] + '/global_stats/eval_scannet/results_per_scene_filtered/'
    outputdir = global_config["general"]["target_folder"] + '/global_stats/eval_scannet/results_per_scene_scan2cad_constraints/'

    print('Overwrite in Scan2CAD constraints')
    # if os.path.exists(outputdir):
    #     assert os.listdir(outputdir) == []
    # else:
    #     os.mkdir(outputdir)

    scan2cad_constraint(projectdir,outputdir, scan2cad_dir +  "cad_appearances.json", scan2cad_dir + "full_annotations.json")
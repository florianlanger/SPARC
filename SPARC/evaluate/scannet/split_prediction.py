import os
import glob
import pandas as pd
import sys
import json



def split(raw_results,out_dir):

    scenes = {}
    for line in raw_results:
        scene = line.split(',')[0]
        if scene == 'id_scan':
            continue

        line = line.replace(line.split(',')[0] + ',','')
        if scene not in scenes:
            scenes[scene] = [line]
        else:
            scenes[scene].append(line)

    for scene in scenes:
        with open(out_dir + scene + '.csv', "w") as f:
            for line in scenes[scene]:
                f.write(line)


def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]
    with open(target_folder + '/global_stats/eval_scannet/raw_results.csv','r') as file:
        raw_results = file.readlines()

    out_dir = target_folder + '/global_stats/eval_scannet/results_per_scene/'


    split(raw_results,out_dir)

if __name__ == '__main__':
    print('Split Predictions csv files')
    main()


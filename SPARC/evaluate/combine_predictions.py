from enum import Flag
import json
import numpy as np

from SPARC.utilities import get_model_to_infos_scannet_just_id



def combine_predictions_with_predictions_all_images(eval_path,seg_and_retrieval_info_path,config,ending_file='.jpg'):

    print('un normalise predictions here')

    with open(eval_path,'r') as f:
        predictions_own = json.load(f)

    with open(seg_and_retrieval_info_path,'r') as f:
        seg_and_retrieval_info = json.load(f)

    model_to_infos_scannet = get_model_to_infos_scannet_just_id(config["data"]["dir_path_scan2cad_anno"])

    for detection in predictions_own:
        detection_id = int(detection.split('_')[-1].split('.')[0])




        if ending_file == '.jpg':
            gt_name = detection.split('-')[0] + '/color/' + detection.split('-')[1].split('_')[0] + ending_file
        elif ending_file == '.json':
            # print('detection',detection)
            # print('')
            gt_name = detection.split('-')[0] + '-' + detection.split('-')[1].split('_')[0] + ending_file
            # print('gt_name',gt_name)
        # for key in ['q','t','s']:

        for key in ['q','t']:
            seg_and_retrieval_info[gt_name][detection_id][key] = predictions_own[detection][key]

        factor = np.array(model_to_infos_scannet[predictions_own[detection]["model_id"]]['bbox']) * 2
        seg_and_retrieval_info[gt_name][detection_id]['s'] = (predictions_own[detection]['s'] / factor).tolist()

        seg_and_retrieval_info[gt_name][detection_id]["scene_cad_id"][1] = predictions_own[detection]["model_id"]
        seg_and_retrieval_info[gt_name][detection_id]["own_prediction"] = True


    for img in seg_and_retrieval_info:
        # print('img',img)
        for i in range(len(seg_and_retrieval_info[img])):
            if "own_prediction" not in seg_and_retrieval_info[img][i]:
                seg_and_retrieval_info[img][i]["own_prediction"] = False
            if seg_and_retrieval_info[img][i]["category"] == 'bookcase':
                seg_and_retrieval_info[img][i]["category"] = 'bookshelf'

            if ending_file == '.jpg':
                seg_and_retrieval_info[img][i]['detection'] = img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '_' + str(i).zfill(2)
                

    return seg_and_retrieval_info


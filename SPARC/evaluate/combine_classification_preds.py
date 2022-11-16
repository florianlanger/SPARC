
import json
import os
import sys


exp_dir = sys.argv[1]

eval_names = os.listdir(exp_dir )

add_on = '/predictions/epoch_000050/translation_pred_scale_pred_rotation_init_for_classification_retrieval_roca_all_images_True/our_single_predictions.json'


# out_path = exp_dir + '/' + eval_names[0] + add_on.replace('our_single_predictions', 'best_rotation_index')
out_path = exp_dir + '/best_rotation_index.json'
assert os.path.exists(out_path) == False

predictions = {}
for file in eval_names:
    rotation_index = int(file.split('_')[-1])
    with open(exp_dir + '/' + file + add_on, 'r') as f:
        predictions[rotation_index] = json.load(f)

for key in predictions:
    assert set(predictions[0]) == set(predictions[key])

rotation_index_per_detection_with_max_score = {}

for detection in predictions[0]:
    max_score = -1
    max_rotation_index = -1
    for rotation_index in predictions:
        score = predictions[rotation_index][detection]["classification_score"][0]
        if score > max_score:
            max_score = score
            max_rotation_index = rotation_index
    rotation_index_per_detection_with_max_score[detection] = max_rotation_index

with open(out_path, 'w') as f:
    json.dump(rotation_index_per_detection_with_max_score, f)

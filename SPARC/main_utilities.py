
import torch
import torch.optim as optim

import os
import numpy as np

import torch_optimizer as optim_special

import json
from scipy.spatial.transform import Rotation as scipy_rot
import random

from SPARC.model.image_network import Classification_network
from SPARC.dataset.dataset_points import Dataset_points
from SPARC.utilities import dict_replace_value
from SPARC.model.perceiver_pytorch import Perceiver


def save_predictions(all_metrics,N_refinements,epoch,exp_path,eval_params,use_all_images):


    extra_infos_combined = {}
    for key in all_metrics['all_extra_infos'][0]:
        extra_infos_combined[key] = np.concatenate([all_metrics['all_extra_infos'][i][key] for i in range(len(all_metrics['all_extra_infos']))])

    # for key in all_metrics['all_extra_infos'][0]:
    #     print(key)

    iter_refinement = np.array(all_metrics['iter_refinement'])

    mask = iter_refinement == N_refinements - 1
    # detection_names = np.array(all_metrics['detection_names'])[mask]
    # s_pred = np.array(all_metrics['s_pred'])[mask,:]
    # t_pred = np.array(all_metrics['t_pred'])[mask,:]

    detection_names = np.array(all_metrics['detection_names'])
    s_pred = np.array(all_metrics['s_pred'])
    t_pred = np.array(all_metrics['t_pred'])
    r_pred = np.array(all_metrics['r_pred'])

    assert detection_names.shape[0] == s_pred.shape[0] == t_pred.shape[0]
    output = {}
    for i in range(detection_names.shape[0]):
        if mask[i] == True:
            output[detection_names[i]] = {}
            output[detection_names[i]]['s'] = (extra_infos_combined['S'][i,:] * (1 + s_pred[i])).tolist()
            output[detection_names[i]]['t'] = (extra_infos_combined['T'][i,:] + t_pred[i]).tolist()

            r_offset_pred = scipy_rot.from_quat(r_pred[i])
            q = (scipy_rot.from_matrix(extra_infos_combined['R'][i])*r_offset_pred).as_quat()
            # change because different convention
            q = [q[3],q[0],q[1],q[2]]
    
            output[detection_names[i]]['q'] = q
            output[detection_names[i]]["model_id"] = extra_infos_combined['model_3d_name'][i].split('_')[1].split('.')[0]
            output[detection_names[i]]["classification_score"] = all_metrics['all_predictions'][i]

    out_dir = exp_path + '/predictions/epoch_{}'.format(str(epoch).zfill(6))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir += '/translation_{}_scale_{}_rotation_{}_retrieval_{}_all_images_{}'.format(eval_params["what_translation"],eval_params["what_scale"],eval_params["what_rotation"],eval_params["what_retrieval"],str(use_all_images))
    os.mkdir(out_dir)

    out_path = out_dir + '/our_single_predictions.json'
    with open(out_path, 'w') as f:
        json.dump(output, f)

    return out_dir



def get_N_refinements_and_sample_for_classification(kind,config):
    if kind == 'val_roca':
        N_refinements = config['training']['refinement_per_object']
        sample_for_classification = False
    else:
        N_refinements = int(np.random.choice([1,2,3,4], 1, p=[0.6, 0.2, 0.1, 0.1]))
        sample_for_classification = False
        if N_refinements == 1 and np.random.rand() > 0.5:
            sample_for_classification = True
    return N_refinements,sample_for_classification

def get_N_refinements_and_sample_for_classification_v2(kind,config):
    if kind == 'val_roca':
        N_refinements = config['training']['refinement_per_object']
        sample_for_classification = False
    else:
        # N_refinements = int(np.random.choice([1,3], 1, p=[0.5,0.5]))
        p_classifier = config['training']['p_classifier']
        N_refinements = int(np.random.choice([1,config['training']['refinement_per_object']], 1, p=[p_classifier,1-p_classifier]))
        sample_for_classification = False
        if N_refinements == 1:
            sample_for_classification = True
    return N_refinements,sample_for_classification

def get_N_refinements_and_sample_for_classification_v3(kind,config):
    if kind == 'val_roca':
        N_refinements = config['training']['refinement_per_object']
        sample_for_classification = False
    else:
        p_classifier = config['training']['p_classifier']
        sample_for_classification = np.random.choice([True,False], 1, p=[p_classifier,1-p_classifier])
        if sample_for_classification == True:
            N_refinements = 1
        elif sample_for_classification == False:
            N_refinements = config['training']['refinement_per_object']
    return N_refinements,sample_for_classification



def update_running_metrics(all_metrics,latest_metrics,extra_infos,loss,iter_refinement):


    all_metrics['all_predictions'] += latest_metrics['probabilities'].tolist()
    all_metrics['all_labels'] += latest_metrics['labels'].tolist()
    all_metrics['all_categories'] += extra_infos['category'].tolist()
    all_metrics['all_roca_bbox'] += extra_infos['roca_bbox'].tolist()
    all_metrics['all_losses'] += loss.tolist()
    all_metrics['all_extra_infos'].append(extra_infos)
    all_metrics['t_distance'] += latest_metrics['t_distance'].tolist()
    all_metrics['t_offset'] += extra_infos['offset_t'].tolist()
    all_metrics['s_distance'] += latest_metrics['s_distance'].tolist()
    all_metrics['s_offset'] += extra_infos['offset_s'].tolist()
    all_metrics['r_distance'] += latest_metrics['r_distance'].tolist()
    all_metrics['r_offset'] += extra_infos['offset_r'].tolist()
    all_metrics['t_correct'] += latest_metrics['t_correct'].tolist()
    all_metrics['s_correct'] += latest_metrics['s_correct'].tolist()
    all_metrics['iter_refinement'] += [iter_refinement] * len(latest_metrics['probabilities'])

    all_metrics['weighted_classification_loss'] += latest_metrics['weighted_classification_loss'].tolist()
    all_metrics['weighted_t_loss'] += latest_metrics['weighted_t_loss'].tolist()
    all_metrics['weighted_s_loss'] += latest_metrics['weighted_s_loss'].tolist()
    all_metrics['weighted_r_loss'] += latest_metrics['weighted_r_loss'].tolist()

    all_metrics['counter_examples'] += loss.shape[0]
    all_metrics['running_loss'] += torch.sum(loss).item()
    all_metrics['n_correct_total'] += torch.sum(latest_metrics['correct']).item()

    all_metrics['detection_names'] += extra_infos['detection_name'].tolist()
    all_metrics['s_pred'] += latest_metrics['s_pred'].tolist()
    all_metrics['t_pred'] += latest_metrics['t_pred'].tolist()
    all_metrics['r_pred'] += latest_metrics['r_pred'].tolist()

    return all_metrics




def set_network_state(net,kind):
    if kind == 'train':
        net.train()
    elif kind == 'val' or kind == 'val_roca':
        net.eval()

def set_device(config):
    if torch.cuda.is_available():
        print('config["general"]["gpu"]',config['general']['gpu'])
        device = torch.device("cuda:{}".format(config["general"]["gpu"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print('on cpu')
    return device

def create_data_loaders(config):


    if config['data']['targets'] == 'labels':

        if config['training']['only_eval_roca'] == False:

            train_dataset = Dataset_points(config,kind='train')

            if config["training"]["validate"] == True:
                # val_dataset = Dataset_points(config,kind='val')
                # val_roca_dataset = Dataset_points(config,kind='val_roca')
                val_dataset = None
                val_roca_dataset = None
                val_roca_dataset_all_images = Dataset_points(config,kind='val_roca',use_all_images=True)
            elif config["training"]["validate"] == False:
                val_dataset = None
                val_roca_dataset = None
                val_roca_dataset_all_images = None
                print('No val roca dataset')
            
        elif config['training']['only_eval_roca'] == True:
            train_dataset = None
            val_dataset = None
            val_roca_dataset = None
            val_roca_dataset_all_images = Dataset_points(config,kind='val_roca',use_all_images=True)
            print('Only eval roca dataset')
        
    print('Loaded datasets')
    return train_dataset, val_dataset,val_roca_dataset, val_roca_dataset_all_images

def create_shuffled_indices(n,config,kind):
    all_indices = np.arange(n)
    if kind == 'train' or kind == 'val':
        random.shuffle(all_indices)
    all_indices = np.repeat(all_indices,config['training']["n_same_objects_per_batch"])
    return all_indices

def log_hparams(writer,config,metric_dict):

    hparam_dict = {'bs': config["training"]["batch_size"],'lr': config["training"]["learning_rate"],'optimiser':config["training"]['optimiser']}
    writer.add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)

def get_optimiser(config,network):
    if config["training"]["optimiser"] == 'Adam':
        optimizer = optim.Adam([{'params': network.parameters()}], lr=config["training"]["learning_rate"])
    elif config["training"]["optimiser"] == 'SGD':
        optimizer = optim.SGD([{'params': network.parameters()}], lr=config["training"]["learning_rate"],momentum=config["training"]["momentum_sgd"])
    elif config['training']['optimiser'] == 'Lamb':
        optimizer = optim_special.Lamb([{'params': network.parameters()}], lr=config["training"]["learning_rate"])
    return optimizer

def save_checkpoint(dir_path,epoch,network,optimizer,config):
    # torch.save(network.state_dict(), dir_path + '/network_last_epoch.pth')
    # torch.save(optimizer.state_dict(), dir_path + '/optimizer_last_epoch.pth')
    if epoch % config["training"]["save_interval"] == 0:
        torch.save(network.state_dict(), dir_path + '/network_epoch_{}.pth'.format(str(epoch).zfill(6)))
        torch.save(optimizer.state_dict(), dir_path + '/optimizer_epoch_{}.pth'.format(str(epoch).zfill(6)))



def add_missing_keys_to_config(config):

    if "regress_offsets" not in config['model']:
        config['model']["regress_offsets"] = False
    if not "sample_wrong_R_percentage" in config["data_augmentation"]:
        config["data_augmentation"]["sample_wrong_R_percentage"] = 0.0
    if 'type' not in config['data']:
        config['data']['type'] = 'lines'
    if "name_norm_folder" not in config["data"]:
        config["data"]["name_norm_folder"] = "norm"

    if "train_only_classifier" not in config["loss"]:
        config["loss"]["train_only_classifier"] = False
    if "evaluate" not in config:
        config["evaluate"] = {"rotation_index": 0}

    if 'fourier_encode_data' not in config['model']['perceiver_config']:
        config['model']['perceiver_config']['fourier_encode_data'] = True

    if "N_rotations" not in config["data_augmentation"]:
        config["data_augmentation"]["N_rotations"] = 4
    

    default_false = ['use_all_points_2d','add_random_points','input_rgb',"input_RST_and_CAD_ID","add_history"]
    default_true = ["use_3d_points","rerender_points"]

    for key in default_false:
        if key not in config['data']:
            config['data'][key] = False
        
    for key in default_true:
        if key not in config['data']:
            config['data'][key] = True

    return config

def process_config(config):

    config = add_missing_keys_to_config(config)

    
    keys = ["dir_path_3d_points_and_normals","dir_path_shapes_for_vis","dir_path_2d_train","dir_path_2d_val","dir_path_2d_val_roca","dir_path_2d_val_roca_all_images","path_roca_all_preds","dir_path_scan2cad_anno","path_scan2cad_cad_appearances","path_scannet_poses","path_scaling_limits"]
    values = ["data_3d/points_normals_1000/","data_3d/shapenet_models_rescaled/","train","no_path","val","val","val/all_detection_infos_all_images_normalised_scale.json","data_scannet/scan2cad_full_annotations.json","data_scannet/cad_appearances.json","data_scannet/combined_poses.json","data_3d/scaling/scaling_limits_increased_display_z.json"]
    for key in keys:
        if key not in config["data"]:
            config["data"][key] = config["general"]["dataset_dir"] + values[keys.index(key)]

    config["data"]["objects_train"] = "valid_objects_normalised_scale"
    config["data"]["objects_val"] = "valid_objects_normalised_scale"

    n_rgb = (np.sum([config["data"]["use_rgb"],config["data"]["use_normals"],config["data"]["use_alpha"]]) + 1)
    if "n_same_objects_per_batch" in config["training"]:
        assert config["training"]["batch_size"] % config["training"]["n_same_objects_per_batch"] == 0
        assert config["training"]["batch_size_val"] % config["training"]["n_same_objects_per_batch"] == 0


    if config['model']["regress_offsets"] == False:
        config["model"]["n_outputs"] = 1
    elif config['model']["regress_offsets"] == True:
        # config["model"]["n_outputs"] = 7
        config["model"]["n_outputs"] = 11
        # change back to 11 had 16 before even with quaternion


    assert config["data_augmentation"]["change_R_angle_degree"][1] < 180 / config["data_augmentation"]["N_rotations"]


    if config["model"]["type"] == "vgg16" or config["model"]["type"] == "resnet50" or config["model"]["type"] == "resnet18":
        config["training"]["optimiser"] = "Adam"
        config["training"]["learning_rate"] = 0.00005
        # print('lr times 2')
        # config["training"]["learning_rate"] = 0.00005 * 2
    elif config["model"]["type"] == "perceiver":
        config["training"]["optimiser"] = "Lamb"
        config["training"]["learning_rate"] = 0.001

    if config["data"]["what_models"] == "lines":
        config["data"]["dims_per_pixel"] = 3
    elif config["data"]["what_models"] == "points_and_normals" and config["data"]["input_3d_coords"] ==  False and config["data"]["input_rgb"] ==  False:
        config["data"]["dims_per_pixel"] = 7
        config["data"]["indices_rgb"] = (None,None)
        config["data"]["indices_3d"] = (None,None)
    elif config["data"]["what_models"] == "points_and_normals" and config["data"]["input_3d_coords"] ==  True and config["data"]["input_rgb"] ==  False:
        config["data"]["dims_per_pixel"] = 10
        config["data"]["indices_rgb"] = (None,None)
        config["data"]["indices_3d"] = (7,10)
    elif config["data"]["what_models"] == "points_and_normals" and config["data"]["input_3d_coords"] ==  False and config["data"]["input_rgb"] ==  True:
        config["data"]["dims_per_pixel"] = 10
        config["data"]["indices_rgb"] = (7,10)
        config["data"]["indices_3d"] = (None,None)
    elif config["data"]["what_models"] == "points_and_normals" and config["data"]["input_3d_coords"] ==  True and config["data"]["input_rgb"] ==  True:
        config["data"]["dims_per_pixel"] = 13
        config["data"]["indices_rgb"] = (10,13)
        config["data"]["indices_3d"] = (7,10)

    if config["data"]["sample_what"] == 'T':
        # config["data"]["sample_T"] = {"percent_small": 0.4,"percent_large": 0.5,"threshold_correct_T": 0.2}
        config["data"]["sample_T"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": True,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}
        config["data"]["sample_S"] = {"use_gt": True,"ignore_prediction": True,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": True,"ignore_prediction": True}

        config["loss"]["constants_multiplier"]["classification"] = 0.0
        config["loss"]["constants_multiplier"]["s"] = 0.0
        config["loss"]["constants_multiplier"]["r"] = 0.0

    elif config["data"]["sample_what"] == 'T_and_R':
        # config["data"]["sample_T"] = {"percent_small": 0.4,"percent_large": 0.5,"threshold_correct_T": 0.2}
        config["data"]["sample_T"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": True,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}
        config["data"]["sample_S"] = {"use_gt": True,"ignore_prediction": True,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": False,"ignore_prediction": False}

        config["loss"]["constants_multiplier"]["classification"] = 0.0
        config["loss"]["constants_multiplier"]["s"] = 0.0

    elif config["data"]["sample_what"] == 'T_and_S':
        config["data"]["sample_T"] = {"use_gt": False,"percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": False,"percent_small": 0.0,"limit_small": 0.2,"percent_large": 0.0,"limit_large": 0.5}
        # config["data"]["sample_T"] = {"percent_small": 0.7,"percent_large": 0.2,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": False,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}
        config["data"]["sample_S"] = {"use_gt": False,"percent_small": 0.0,"percent_large": 0.0}

    elif config["data"]["sample_what"] == 'T_and_R_and_S':
        # config["data"]["sample_T"] = {"percent_small": 0.4,"percent_large": 0.5,"threshold_correct_T": 0.2}
        config["data"]["sample_T"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        # config["data"]["sample_S"] = {"use_gt": True,"percent_small": 0.7,"limit_small": 0.2,"percent_large": 0.2,"limit_large": 0.5}
        config["data"]["sample_S"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": False,"ignore_prediction": False}



    elif config["data"]["sample_what"] == 'S':
        config["data"]["sample_T"] = {"use_gt": True,"ignore_prediction": True, "percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        config["data"]["sample_S"] = {"use_gt": False,"ignore_prediction": False,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": True,"ignore_prediction": True}
        config["loss"]["constants_multiplier"]["classification"] = 0.0
        config["loss"]["constants_multiplier"]["t"] = 0.0
        config["loss"]["constants_multiplier"]["r"] = 0.0

    elif config["data"]["sample_what"] == 'R':
        config["data"]["sample_T"] = {"use_gt": True,"ignore_prediction": True, "percent_small": 0.0,"percent_large": 0.0,"threshold_correct_T": 0.2}
        config["data"]["sample_S"] = {"use_gt": True,"ignore_prediction": True,"percent_small": 0.0,"percent_large": 0.0}
        config["data"]["sample_R"] = {"use_gt": False,"ignore_prediction": False}
        config["loss"]["constants_multiplier"]["classification"] = 0.0
        config["loss"]["constants_multiplier"]["t"] = 0.0
        config["loss"]["constants_multiplier"]["s"] = 0.0


    return config

def load_network(config,device):
    print('loading network')

    if config["model"]["type"] == 'perceiver':
        network = Perceiver(
        input_channels = config['data']["dims_per_pixel"],          # number of channels for each token of the input
        input_axis = 1,              # number of axis for input data (2 for images, 3 for video)
        num_freq_bands = config['model']['perceiver_config']['num_freq_bands'],  # number of freq bands, with original value (2 * K + 1
        max_freq = config['model']['perceiver_config']['max_freq'],              # maximum frequency, hyperparameter depending on how fine the data is
        depth = config['model']['perceiver_config']['depth'],                   # depth of net. The shape of the final attention mechanism will be:
                                    #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents = config['model']['perceiver_config']['num_latents'],           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = config['model']['perceiver_config']['latent_dim'],            # latent dimension
        cross_heads = config['model']['perceiver_config']['cross_heads'],             # number of heads for cross attention. paper said 1
        latent_heads = 8,            # number of heads for latent self attention, 8
        cross_dim_head = 64,         # number of dimensions per cross attention head
        latent_dim_head = 64,        # number of dimensions per latent self attention head
        num_classes = config["model"]["n_outputs"],          # output number of classes
        attn_dropout = config['model']['perceiver_config']['attn_dropout'],
        ff_dropout = config['model']['perceiver_config']['ff_dropout'],
        weight_tie_layers = config['model']['perceiver_config']['weight_tie_layers'],   # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data = config['model']['perceiver_config']['fourier_encode_data'],  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn = 2,      # number of self attention blocks per cross attention
        final_classifier_head = True
    )
        print('sending to device')
        print(device)
        network = network.to(device)

    else:
        network = Classification_network(config,device)
    print('loaded network')
    return network



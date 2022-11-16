from shutil import register_unpack_format
import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data.sampler import Sampler
import numpy as np

from scipy.spatial.transform import Rotation as scipy_rot



def get_index_infos(outputs,extra_infos,config,n_refinement):
    outputs = outputs.cpu().detach().numpy()

    list_of_dicts = []
    for i in range(len(outputs)):
        single_dict = {}
        for key in extra_infos:
            if key == 'T':
                single_dict[key] = extra_infos[key][i] + outputs[i][1:4]
            elif key == 'offset_t':
                single_dict[key] = extra_infos[key][i] - outputs[i][1:4]
            elif key == 'S':
                # single_dict[key] = extra_infos[key][i] * (outputs[i][4:7] + 1)
                single_dict[key] = extra_infos[key][i] + outputs[i][4:7]
            elif key == 'offset_s':
            
                single_dict[key] =  extra_infos['S_gt'][i] - (extra_infos['S'][i] + outputs[i][4:7])

            elif key == 'R':
                # offset_r = outputs[i][7:16].reshape(3,3)
                if 'R' in config["data"]["sample_what"]:
                    offset_r = scipy_rot.from_quat(outputs[i][7:11]).as_matrix()
                    single_dict[key] = np.matmul(extra_infos[key][i],offset_r)
                
                # if dont sample and learn R just keep gt R, so that dont add on existing augmentations 
                else:
                    single_dict[key] = np.array(extra_infos['R_gt'][i])


            elif key == 'offset_r':
                if 'R' in config["data"]["sample_what"]:
                    # offset_r_new = outputs[i][7:16].reshape(3,3)
                    # inv_offset_r_new = np.linalg.inv(offset_r_new)
                    # previous_offset = extra_infos['offset_r'][i].reshape(3,3)
                    # single_dict[key] = np.matmul(inv_offset_r_new,previous_offset).reshape(9)
                    offset_r_new = scipy_rot.from_quat(outputs[i][7:11])
                    previous_offset = scipy_rot.from_quat(extra_infos['offset_r'][i])
                    out = (offset_r_new.inv() * previous_offset).as_quat()
                    if out[3] < 0:
                        out = out * -1
                    single_dict[key] = out
                else:
                    single_dict[key] = np.array([0,0,0,1.])


            else:
                single_dict[key] = extra_infos[key][i]
        list_of_dicts.append(single_dict)

    return list_of_dicts


def get_batch_from_dataset(dataset,indices,sample_just_classifier):
    batch = get_simple_batch_from_dataset(dataset,indices,sample_just_classifier)

    padded_info_all = torch.stack([item[0] for item in batch])
    target_all = torch.stack([item[1] for item in batch])

    extra_infos_all = {}
    for key in batch[0][2]:
        extra_infos_all[key] = np.stack([item[2][key] for item in batch])
    
    return padded_info_all,target_all,extra_infos_all

def get_simple_batch_from_dataset(dataset,indices,sample_just_classifier):
    batch = []

    for counter,i in enumerate(indices):
        # debug here replace output with 0 s 
        tuple_index_just_classifier = (i,sample_just_classifier)
        batch.append(dataset[tuple_index_just_classifier])
    return batch


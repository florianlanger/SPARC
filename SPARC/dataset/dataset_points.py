from asyncio import run_coroutine_threadsafe
import re
from xml.parsers.expat import model
from xml.sax.handler import DTDHandler
import torch
from torch.utils import data
import numpy as np
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import imageio
import os
import random
import cv2
from scipy.spatial.transform import Rotation as scipy_rot
from SPARC.utilities import load_json
import time
from torch.utils.data import DataLoader
from torch.multiprocessing import Pool, Process, set_start_method
import torchvision.transforms as transforms


from SPARC.dataset.dataset_combine_classification_and_regression import load_numpy_dir,get_gt_infos,get_gt_infos_by_detection,bbox_to_lines,reproject_3d_lines,check_config,compute_lines_resized_torch,Dataset_lines,compute_lines_resized


def augment_depth(depth,kind,device,config):
    depth_aug_infos = config['data_augmentation']['depth']
    random_number = np.random.rand()
    if depth_aug_infos['use_depth_augmentation'] == True and kind == 'train' and random_number < depth_aug_infos['probability_augmentation']:
        noise =  (2 * torch.rand(depth.shape,device=device) - 1) * depth_aug_infos['max_augmentation']
        depth = depth + noise
        depth = torch.clamp(depth,min=0,max=None)
    
    return depth 

def flatten(t):
    return [item for sublist in t for item in sublist]

def get_angle(m1,m2):
    assert m1.shape == m2.shape

    m = torch.matmul(m1,torch.transpose(m2,-1,-2))

    value = m.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    value = (value - 1 )/ 2

    clipped_value = torch.clip(value,-0.9999999,0.999999)

    angle = torch.arccos(clipped_value)

    return angle * 180 / np.pi

def get_device(a):

    if a.is_cuda:
        gpu_n = a.get_device()
        device = torch.device("cuda:{}".format(gpu_n))
    else:
        device = torch.device("cpu")
    return device

def points_3d_to_pixel_from_calibration(points_3D,R,T,K):
    t1 = time.time()
    cc1 = torch.transpose(torch.matmul(R,torch.transpose(points_3D,-1,-2)),-1,-2) + T
    nc1 = cc1 / torch.abs(cc1[:,2:3])

    mask = cc1[:,2] > 0

    nc1 = nc1 * torch.Tensor([-1,-1,1]).to(get_device(nc1))

    K = K[:3,:3].to(get_device(nc1))
    K[2,2] = 1

    pixel = torch.transpose(torch.matmul(K,torch.transpose(nc1,-1,-2)),-1,-2)
    t2 = time.time()
    # pixel = torch.cat([pixel[:,1:2],pixel[:,0:1]],dim=1)

    return pixel[:,:2],mask

def sample_points_from_lines(lines,points_per_line):

    device = get_device(lines)

    n_lines = lines.shape[0]
    lines = torch.repeat_interleave(lines,points_per_line,dim=0)
    interval = torch.linspace(0,1,points_per_line).repeat(n_lines).to(device)
    interval = interval.unsqueeze(1).repeat(1,2)
    points = lines[:,:2] + (lines[:,2:4]-lines[:,:2]) * interval
    points = points.view(n_lines,points_per_line,2)
    return points

def get_offsets(n_offsets,distance_to_center):
    s = distance_to_center
    if n_offsets == 1:
        offsets = torch.Tensor([[0,0]]).long()
    elif n_offsets == 5:
        offsets = torch.Tensor([[-s,0],[0,-s],[0,0],[0,s],[s,0]]).long()
    elif n_offsets == 9:
        offsets = torch.Tensor([[-s,-s],[-s,0],[-s,s],[0,-s],[0,0],[0,s],[s,-s],[s,0],[s,s]]).long()
    elif n_offsets == 25:
        offsets = torch.Tensor([[-2*s,-2*s],[-2*s,-s],[-2*s,0],[-2*s,s],[-2*s,2*s],[-s,-2*s],[-s,-s],[-s,0],[-s,s],[-s,2*s],[0,-2*s],[0,-s],[0,0],[0,s],[0,2*s],[s,-2*s],[s,-s],[s,0],[s,s],[s,2*s],[2*s,-2*s],[2*s,-s],[2*s,0],[2*s,s],[2*s,2*s]]).long()
    return offsets


def expand_indices(indices,n_locations_per_pixel,distance_to_center,device,target_size):
    assert indices.shape[1] == 2, indices.shape
    n_indices_orig = indices.shape[0]

    indices = torch.repeat_interleave(indices,n_locations_per_pixel,dim=0)
    offsets = get_offsets(n_locations_per_pixel,distance_to_center)

    indices = indices + offsets.to(device).repeat((n_indices_orig,1))

    mask = torch.all((indices >= torch.zeros(2).to(device)) & (indices < torch.Tensor([target_size[0],target_size[1]]).to(device)),dim=1)
    indices = indices[mask,:]

    # for i in range(2):
    #     indices[:,i] = indices[:,i].clamp(min=0,max=target_size[i]-1)
    return indices


def get_cc_from_points_and_depth(points,depth,img_size,K,device):
    assert points.shape[1] == 2, points.shape

    reshaped_orig_size = torch.Tensor(img_size).unsqueeze(0).repeat((points.shape[0],1)).to(device)
    points = points * reshaped_orig_size
    # print(points.shape)
    # print(depth.shape)
    # print('points[:3]',points[:3])

    K = K[:3,:3].to(device)
    K[2,2] = 1
    K_inv = torch.inverse(K)

    pixel = torch.cat([points[:,:],points[:,:1] * 0 + 1],dim=1)
    nc1 = torch.transpose(torch.matmul(K_inv,torch.transpose(pixel,-1,-2)),-1,-2)

    nc1 = nc1 * torch.Tensor([-1,-1,1]).to(device)

    # print('nc1[:3]',nc1.shape)
    # print('depth[:3]',depth.shape)
    # print('torch.linalg.norm(nc1[:,:],dim=1)',torch.linalg.norm(nc1[:,:],dim=1).shape)

    cc1 = nc1 / torch.linalg.norm(nc1[:,:],dim=1).unsqueeze(1).repeat(1,3) * depth.repeat(1,3)

    # print(cc1.shape)
    return cc1

def add_samples_from_within_bbox(points_2d,bbox,n_samples,device):
    assert points_2d.shape[1] == 2, points_2d.shape
    x_coords = torch.rand((n_samples,),device=device) * (bbox[2]-bbox[0]) + bbox[0]
    y_coords = torch.rand((n_samples,),device=device) * (bbox[3]-bbox[1]) + bbox[1]
    points_sampled = torch.stack([x_coords,y_coords],dim=1)
    all_points = torch.cat([points_2d,points_sampled],dim=0)
    return all_points




def create_base_points(config,target_size,n_points_per_line,lines_2d=None,bbox=None,device=torch.device('cpu'),normals=None,depth=None,img_rgb=None,use_canny=False,reprojected_points=None,ratio_preloaded_original=1.0,orig_img_size=None,K=None,kind=None):

    points = []
    # print('create base points')
    t1 = time.time()

    if config["data"]["use_bbox"] == True:
        # img = draw_boxes(img,np.array([bbox]),thickness=1,color=(0,0,255))
        bbox_lines = bbox_to_lines(bbox)
        # t2 = time.time()
        points_3d_bbox = get_points_3d_torch(bbox_lines.to(device),target_size,bbox_lines.shape[0],channel=1,n_samples=n_points_per_line).squeeze(0)
        assert points_3d_bbox.shape == (bbox_lines.shape[0]*n_points_per_line,3), (points_3d_bbox.shape,bbox_lines.shape,n_points_per_line)
        points_3d_bbox_extra_channels = torch.cat([points_3d_bbox,torch.zeros((points_3d_bbox.shape[0],config["data"]["dims_per_pixel"]-3),dtype=torch.float32).to(device)],dim=1)
        assert points_3d_bbox_extra_channels.shape == (bbox_lines.shape[0]*n_points_per_line,config["data"]["dims_per_pixel"]), (points_3d_bbox_extra_channels.shape,bbox_lines.shape,n_points_per_line)
        # t3 = time.time()
        reshaped_size = torch.Tensor([target_size + (config["data"]["dims_per_pixel"] - 2) * [1]]).to(device).repeat((points_3d_bbox_extra_channels.shape[0],1))
        points_3d_bbox_extra_channels = points_3d_bbox_extra_channels / reshaped_size
        assert torch.all(points_3d_bbox_extra_channels[:,2] < 2.5),points_3d_bbox_extra_channels[:,2]
        points.append(points_3d_bbox_extra_channels)
    # t4 = time.time()

    t2 = time.time()
    if config["data"]["use_lines_2d"] == True:
        if use_canny == False:
            points_2d = get_points_3d_torch(lines_2d.to(device),target_size,lines_2d.shape[0],channel=0,n_samples=n_points_per_line).squeeze(0)
            assert points_2d.shape == (lines_2d.shape[0]*n_points_per_line,3), (points_2d.shape,lines_2d.shape,n_points_per_line)
        else:
            points_2d = lines_2d.to(device)
            assert len(points_2d.shape) == 2 and points_2d.shape[1] == 2, points_2d.shape
        t3 = time.time()

        if config["data"]["add_samples_from_within_bbox"] == True:
            points_2d = add_samples_from_within_bbox(points_2d,bbox,config["data"]["number_of_samples_from_within_bbox"],device)
        indices = points_2d[:,:2].long()
        indices = expand_indices(indices,config["data"]["number_locations_per_pixel_2d"],config["data"]["distance_to_center_2d"],device,target_size)
        t4 = time.time()
        channels = torch.zeros((indices.shape[0],1),dtype=torch.float32,device=device)

        # print('channels 0 shape',channels.shape)
        t5 = time.time()
        if reprojected_points != None:
            indices_reprojected = torch.round(reprojected_points).long()
            t6 = time.time()
            indices_reprojected = expand_indices(indices_reprojected,config["data"]["number_locations_per_pixel_reprojected"],config["data"]["distance_to_center_reprojected"],device,target_size)
            t7 = time.time()
            # print('indices_reprojected shape channel 4',indices_reprojected.shape)
            indices = torch.cat([indices,indices_reprojected],dim=0)
            channels = torch.cat([channels,4*torch.ones((indices_reprojected.shape[0],1),dtype=torch.float32,device=device)],dim=0)
        t8 = time.time()

        # normalise 2d points to be in range [0,1], dont use original ones as had to clamp some to be in range
        reshaped_size = torch.Tensor(target_size).unsqueeze(0).repeat((indices.shape[0],1)).to(device)
        # points = points / reshaped_size
        points_2d_normalised = indices.float() / reshaped_size

        points_and_channel = torch.cat([points_2d_normalised,channels],dim=1)
        # assert torch.all(points_and_channel[:,2] < 2.5),combined_points[:,2]
        # indices_accessing_normal_depth = indices
        indices_accessing_normal_depth = torch.round(indices.float() * ratio_preloaded_original).long()
        max_for_clamp = (torch.Tensor(target_size) * ratio_preloaded_original).long().to(device) -1
        indices_accessing_normal_depth = torch.clamp(indices_accessing_normal_depth,min=torch.zeros(2,device=device,dtype=torch.long),max=max_for_clamp)

        normals_selected = normals[indices_accessing_normal_depth[:,1],indices_accessing_normal_depth[:,0],:]
        depth_selected = depth[indices_accessing_normal_depth[:,1],indices_accessing_normal_depth[:,0]].unsqueeze(1)
        depth_selected = augment_depth(depth_selected,kind,device,config)

        t9 = time.time()

        channels_for_concat = [points_and_channel,normals_selected,depth_selected]

        if config["data"]["input_3d_coords"] == True:
            cc = get_cc_from_points_and_depth(points_2d_normalised,depth_selected,orig_img_size,K,device)
            channels_for_concat.append(cc)

        if config["data"]["input_rgb"] == True:
            rgb_selected = img_rgb[indices_accessing_normal_depth[:,1],indices_accessing_normal_depth[:,0],:]
            channels_for_concat.append(rgb_selected)

        combined_points = torch.cat(channels_for_concat,dim=1)

        assert torch.all(combined_points[:,2] < 4.5),combined_points[:,2]
        points.append(combined_points)
        
        # print(df)
   
    points = torch.cat(points,dim=0)
    # assert torch.all(points[:,2] < 2.5),points[:,2]
    t10 = time.time()

    
    # print('create base points time:',t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t8-t7,t9-t8,t10-t9)

    return points


def get_points_3d_torch(lines_2d_batched,base_img_shape,n_lines_2d,samples_per_pixel=1,channel=0,n_samples=10):
        assert lines_2d_batched.shape[1] == 4 and len(lines_2d_batched.shape) == 2, lines_2d_batched.shape
    
        assert lines_2d_batched.shape[0] % n_lines_2d == 0, (lines_2d_batched,n_lines_2d)
        t1 = time.time()
        n_images = lines_2d_batched.shape[0] // n_lines_2d

        device = get_device(lines_2d_batched)

        # max_length_lines = torch.max(torch.linalg.norm(lines_2d_batched[:,2:4] - lines_2d_batched[:,0:2],dim=1))
        # max_length_lines = np.linalg.norm(base_img_shape[0:2])
        # n_samples = int(np.round(max_length_lines.item() * samples_per_pixel))

        t2 = time.time()
        points = sample_points_from_lines(lines_2d_batched,n_samples)
        # points = torch.round(points).long()

        assert points.shape == (lines_2d_batched.shape[0],n_samples,2), (points.shape,[lines_2d_batched.shape[0],n_samples,2])
        t3 = time.time()

        channel_index = channel
        channel_points = torch.ones((points.shape[0],n_samples,1),dtype=torch.float32).to(device) * channel_index
        points = torch.cat([points,channel_points],dim=2)        

        # points = points.view(lines_2d_batched.shape[0]*n_samples,2)

        points = points.view(n_images,n_lines_2d*n_samples,3)

        assert points.shape == (n_images,n_lines_2d*n_samples,3), (n_images,n_lines_2d*n_samples,3)

        return points


class Dataset_points(Dataset_lines):


    def __init__(self,config,kind,use_all_images=False):
        super().__init__(config,kind,use_all_images)

        if self.config['data']["use_preloaded_depth_and_normals"] == True:
            img_size_preloaded = self.config['data']["size_preloaded"]
            print('load depth 2d')
            self.depth_2d = self.load_image_dir(self.dir_path_2d + '/{}_{}_{}'.format(config["data"]["name_depth_folder"],img_size_preloaded[0],img_size_preloaded[1]),max_n=4*config["training"]['max_number'],type='depth')
            print('load normals 2d')
            self.norm_2d = self.load_image_dir(self.dir_path_2d + '/{}_{}_{}'.format(config["data"]["name_norm_folder"],img_size_preloaded[0],img_size_preloaded[1]),max_n=4*config["training"]['max_number'],type='normal')
            if self.config["data"]["input_rgb"] == True:
                self.rgb = self.load_image_dir(self.dir_path_2d + '/images_{}_{}'.format(img_size_preloaded[0],img_size_preloaded[1]),max_n=4*config["training"]['max_number'],type='image')

            assert img_size_preloaded[0] / img_size_preloaded[1] == self.config["data"]["img_size"][0] / self.config["data"]["img_size"][1]
            self.ratio_preloaded_original = float(img_size_preloaded[0] / self.config["data"]["img_size"][0])
        else:
            img_size = self.config["data"]["img_size"]
            self.norm_2d_dir = self.dir_path_2d + '/{}_{}_{}/'.format(config["data"]["name_norm_folder"],img_size[0],img_size[1])
            self.rgb_dir = self.dir_path_2d + '/images_{}_{}/'.format(img_size[0],img_size[1])
            self.depth_2d_dir = self.dir_path_2d + '/{}_{}_{}/'.format(config["data"]["name_depth_folder"],img_size[0],img_size[1])
            self.ratio_preloaded_original = 1

            print('self.ratio_preloaded_original:',self.ratio_preloaded_original)
        if self.config['data']["mark_visible_points"] == True:
            # self.visibility_3d_points = self.load_numpy_dir_compressed(config["data"]["dir_path_3d_points_and_normals"] + 'masks',max_n=1)
            print('Load visibility')
            # self.visibility_3d_points = np.load(config["data"]["dir_path_3d_points_and_normals"] + 'masks_combined.npz')
            self.visibility_3d_points = torch.load(config["data"]["dir_path_3d_points_and_normals"] + 'masks_combined.pt')
            

        self.loading_times = [[],[],[],[],[],[],[],[],[]]
        self.times = {}
        self.epoch = 0

    def create_points(self,lines_2d,augmented_bbox,bbox,orig_img_size,target_size,n_points_per_line,normals,depth,reprojected_points=None):
        lines_2d,augmented_bbox,bbox = compute_lines_resized([lines_2d.cpu(),augmented_bbox,bbox],orig_img_size,target_size)
        points_base = create_base_points(self.config,target_size,n_points_per_line,lines_2d=lines_2d,bbox=augmented_bbox,device=self.device,normals=normals,depth=depth,reprojected_points=reprojected_points,ratio_preloaded_original=self.ratio_preloaded_original)

        return points_base

    def create_points_canny(self,points_2d,augmented_bbox,bbox,orig_img_size,target_size,n_points_per_line,normals,depth,K,reprojected_points=None,img_rgb=None):
        t17 = time.time()
        augmented_bbox,bbox = compute_lines_resized([augmented_bbox,bbox],orig_img_size,target_size)
        t18 = time.time()
        points_base = create_base_points(self.config,target_size,n_points_per_line,lines_2d=points_2d,bbox=augmented_bbox,device=self.device,normals=normals,depth=depth,img_rgb=img_rgb,use_canny=True,reprojected_points=reprojected_points,ratio_preloaded_original=self.ratio_preloaded_original,orig_img_size=orig_img_size,K=K,kind=self.kind)
        t19 = time.time()
        # self.add_times('t18-t17',t18-t17)
        # self.add_times('t19-t18',t19-t18)
        return points_base

    

    def get_index_to_names_canny(self):
        index_to_names = {}
        counter = 0
        for gt_name in sorted(self.gt_infos):

            for i in range(len(self.gt_infos[gt_name]['objects'])):
                model = self.gt_infos[gt_name]['objects'][i]['model']
                model_3d_name = model.split('/')[1] + '_' + model.split('/')[2] + '.npz'
                
                detection_number = self.gt_infos[gt_name]['objects'][i]['index']
                detection_name = gt_name.split('.')[0] + '_{}.npy'.format(str(detection_number).zfill(2))
                index_to_names[counter] = [detection_name,gt_name]
                counter += 1

        return index_to_names



    def get_index_to_names_roca_all_preds_all_images(self):
        index_to_names = {}
        counter = 0
        for gt_name in sorted(self.roca_all_preds):
            for i in range(len(self.roca_all_preds[gt_name])):
                detection_name = self.roca_all_preds[gt_name][i]['detection'] + '.npy'
                index_to_names[counter] = [detection_name,gt_name]
                counter += 1

        return index_to_names

    def get_index_to_names(self, kind):

        if kind == 'val_roca':
            return self.get_index_to_names_roca_all_preds_all_images()

        else:
            if self.config["data"]["use_canny"] == False:
                # return self.get_index_to_names_no_canny(kind)
                return self.get_index_to_names_no_canny_v2()
            elif self.config["data"]["use_canny"] == True:
                return self.get_index_to_names_canny()

    def create_info_RST(self,R,S,T):
        info = torch.zeros((15,self.config["data"]["dims_per_pixel"]))
        # channel 3
        info[:,2] = info[:,2] * 0 + 5
        R = scipy_rot.from_matrix(R).as_quat()
        info[0,3:7] = torch.tensor(R)
        info[1,3:6] = torch.tensor(T)
        info[2,3:6] = torch.tensor(S)
        return info

    def create_info_RST_plus_modelid(self,R,S,T,model_name,sym,just_classifier):
        info = torch.zeros((15,self.config["data"]["dims_per_pixel"]))
        # channel 3

        if self.config["rotation_infos"]["random_rotate_R_before_input"] == True and just_classifier == True:
            R = self.random_rotate_R_input_symmetry(R,sym)

        info[:,2] = info[:,2] * 0 + 5
        R = scipy_rot.from_matrix(R).as_quat()
        info[0,3:7] = torch.tensor(R)
        info[1,3:6] = torch.tensor(T)
        info[2,3:6] = torch.tensor(S)

        index = self.model_name_to_index[model_name.split('.')[0]]
        binary_string_12_digits = '{0:012b}'.format(index)
        binary_as_tensor = torch.tensor([float(x) for x in binary_string_12_digits])
        info[3:15,0] = binary_as_tensor
        info[3:15,2] = info[3:15,2] * 0 + 6

        # print('info',info)
        return info
    
    def random_rotate_R_input_symmetry(self,R,sym):
        n_rots = self.config["data_augmentation"]["N_rotations"]
        # index = np.random.randint(0,4)

        if sym == '__SYM_NONE':
            probs = [1] + [0] * (n_rots -1)
        elif sym == '__SYM_ROTATE_UP_2':
            n_correct = 2
            probs = [0] * n_rots
            probs[0] = 1 / n_correct
            probs[int(n_rots/2)] = 1 / n_correct
        elif sym == '__SYM_ROTATE_UP_4':
            n_correct = 4
            probs = [0] * n_rots
            probs[0] = 1 / n_correct
            probs[int(n_rots/4)] = 1 / n_correct
            probs[int(n_rots/2)] = 1 / n_correct
            probs[int(3*n_rots/4)] = 1 / n_correct
        else:
            probs = [1 / (n_rots)] * n_rots

        index = int(np.random.choice(n_rots, 1, p=probs))

        R = np.matmul(R,self.transform_Rs[index])
        return R


    def sample_all_points(self):

        t1 = time.time()
        x = torch.linspace(0,self.config["data"]["size_preloaded"][0]-1,self.config["data"]["size_preloaded"][0]).to(self.device)
        y = torch.linspace(0,self.config["data"]["size_preloaded"][1]-1,self.config["data"]["size_preloaded"][1]).to(self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        points_2d = torch.stack([grid_x,grid_y],dim=2)

        points_2d = points_2d.view(-1,2) * self.config["data"]["img_size"][0] / self.config["data"]["size_preloaded"][0]
        t2 = time.time()
        points_2d = self.shuffle_tensor_first_dim(points_2d)
        points_2d = self.mask_random(points_2d,self.config["data_augmentation"]["percentage_lines_2d"])
        t3 = time.time()
        # points_2d = torch.zeros(size=(0,2)).to(self.device)
        return points_2d

    def load_normals_and_depth(self,gt_name):

        if self.config['data']["use_preloaded_depth_and_normals"] == True:
            normals = self.norm_2d[gt_name.replace('.json','.png')]
            depth = self.depth_2d[gt_name.replace('.json','.png')]
        else:
            normals = self.load_single_image(self.norm_2d_dir + gt_name.replace('.json','.png'),type='normal')
            depth = self.load_single_image(self.depth_2d_dir + gt_name.replace('.json','.png'),type='depth')

        if self.config['data']['use_normals'] == False:
            normals = normals * 0
        
        if self.config['data']['use_depth'] == False:
            depth = depth * 0

        normals = normals.to(self.device)
        depth = depth.to(self.device)

        return normals,depth

    def load_rgb(self,gt_name):

        if self.config['data']['input_rgb'] == True:
            if self.config['data']["use_preloaded_depth_and_normals"] == True:
                bgr = self.rgb[gt_name.replace('.json','.jpg')]
            else:
                bgr = self.load_single_image(self.rgb_dir + gt_name.replace('.json','.jpg'),type='image')
            bgr = bgr.to(self.device)
            rgb = bgr[:,:,[2, 1, 0]]
            rgb = rgb.to(float) / 255.
            
            if self.kind == 'train':
                rgb = self.augment_image(rgb)

            return rgb
        elif self.config['data']['input_rgb'] == False:
            return None

    def augment_image(self,image):
        image = image.permute(2,1,0)
        transform=transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1),
            transforms.GaussianBlur(3)])
        image = transform(image)
        image = image.permute(2,1,0)
        return image

    def update_hard_examples(self,losses,extra_infos):
        if self.config["hard_example_mining"]["use"] == True:
            extra_infos = self.reformat_extra_infos(extra_infos)
            losses = torch.Tensor(losses)
            n_hard = np.round(self.config["hard_example_mining"]["percentage_hard_examples"] * self.__len__()).astype(int)
            _,hard_indices = torch.sort(losses,descending=True)
            hard_indices = hard_indices[:n_hard]
            for key in extra_infos:
                extra_infos[key] = np.array(extra_infos[key])[hard_indices]

            self.hard_examples = extra_infos
            self.hard_example_indices = np.arange(n_hard).tolist()

    def reformat_extra_infos(self,extra_infos):
        n = len(extra_infos)

        reformatted_extra_infos = {}

        for key in ['detection_name','gt_name','model_3d_name','category','sym','name_orientation']:
            lists = [extra_infos[i][key] for i in range(n)]
            reformatted_extra_infos[key] = flatten(lists)
        for key in ['R','S','T','offset_t','offset_s','r_correct','roca_bbox','correct','S_normalised']:
            reformatted_extra_infos[key] = torch.cat([extra_infos[i][key] for i in range(n)],dim=0)
        
        return reformatted_extra_infos


    def get_infos_from_index(self,index,just_classifier):

        detection_name,gt_name = self.index_to_names[index]


        gt_infos = self.gt_infos_by_detection[detection_name]
        model_3d_name = gt_infos['model'].split('/')[1] + '_' + gt_infos['model'].split('/')[2] + '.npz'
        K = torch.Tensor(self.gt_infos[gt_name.replace('.png','.json')]['K'])
        bbox = self.gt_infos_by_detection[detection_name]['bbox']
        img_size = gt_infos['img_size']

        use_roca_bbox = self.get_info_roca_bbox(gt_infos)


        assert len(bbox) == 4,bbox
        augmented_bbox = self.augment_bbox(bbox,img_size)

        model_path = gt_infos['model']
        gt_scaling_orig = gt_infos['scaling']
        category = gt_infos['category']
        f = gt_infos['focal_length']

        R_gt = gt_infos['rot_mat']
        T_gt = gt_infos['trans_mat']
        S_gt = gt_infos['scaling']
        img_size = gt_infos['img_size']

        R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,bbox_3d = self.get_R_and_T_and_S(augmented_bbox,R_gt,T_gt,f,img_size,model_path,gt_scaling_orig,category,just_classifier)

        # history = self.initialise_history(R,T,S)
        history = None

        return detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history

    def initialise_history(self,R,T,S):
        return None
 

    def get_infos_from_hard_example(self,index):
        index = random.choice(self.hard_example_indices)
        self.hard_example_indices.remove(index)
        if len (self.hard_example_indices) == 0:
            n_hard = np.round(self.config["hard_example_mining"]["percentage_hard_examples"] * self.__len__()).astype(int)
            self.hard_example_indices = np.arange(n_hard).tolist()

        detection_name,gt_name,model_3d_name = self.hard_examples['detection_name'][index],self.hard_examples['gt_name'][index],self.hard_examples['model_3d_name'][index]

        gt_infos = self.gt_infos_by_detection[detection_name]

        K = torch.Tensor(self.gt_infos[gt_name.replace('.png','.json')]['K'])
        bbox = self.gt_infos_by_detection[detection_name]['bbox']

        assert len(bbox) == 4,bbox
        augmented_bbox = self.augment_bbox(bbox,self.gt_infos_by_detection[detection_name])

        # R,T,correct,offset,r_correct,sym = self.get_R_and_T(detection_name,gt_name,augmented_bbox)
        R,T,S,correct,offset_t,offset_s,r_correct,sym,S_normalised = self.hard_examples['R'][index],self.hard_examples['T'][index],self.hard_examples['S'][index],self.hard_examples['correct'][index],self.hard_examples['offset_t'][index],self.hard_examples['offset_s'][index],self.hard_examples['r_correct'][index],self.hard_examples['sym'][index],self.hard_examples['S_normalised'][index]

        # return everything
        return detection_name,gt_name,model_3d_name,gt_infos,K,R,T,S,correct,offset_t,offset_s,r_correct,sym,S_normalised,bbox,augmented_bbox

    def get_infos_from_index_dict(self,index_dict):
        detection_name,gt_name,model_3d_name = index_dict['detection_name'],index_dict['gt_name'],index_dict['model_3d_name']

        gt_infos = self.gt_infos_by_detection[detection_name]
        use_roca_bbox = self.get_info_roca_bbox(gt_infos)

        K = torch.Tensor(self.gt_infos[gt_name.replace('.png','.json')]['K'])
        bbox = index_dict['bbox']

        augmented_bbox = index_dict['augmented_bbox']
        
        R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,img_size,S_gt,use_roca_bbox,bbox_3d,history = index_dict['R'],index_dict['T'],index_dict['S'],index_dict['correct'],index_dict['offset_t'],index_dict['offset_s'],index_dict['offset_r'],index_dict['r_correct'],index_dict['sym'],index_dict['S_normalised'],index_dict['T_gt'],index_dict['R_gt'],gt_infos['img_size'],index_dict['S_gt'],use_roca_bbox,index_dict['bbox_3d'],index_dict['history']

        return detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history

    def get_infos_from_index_dict_roca(self,index_dict):
        detection_name,gt_name,model_3d_name = index_dict['detection_name'],index_dict['gt_name'],index_dict['model_3d_name']
        detection_index = int(detection_name.split('_')[-1].split('.')[0])

        infos = self.roca_all_preds[gt_name][detection_index]
        use_roca_bbox = True

        K = torch.Tensor(infos["associated_gt_infos"]['K'])
        bbox = index_dict['bbox']

        if "scaling" not in infos["associated_gt_infos"]:
            S_gt = [0,0,0]
        else:
             S_gt = infos["associated_gt_infos"]["scaling"]
        img_size = infos["associated_gt_infos"]['img_size']

        augmented_bbox = index_dict['augmented_bbox']
        
        R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,img_size,S_gt,use_roca_bbox,bbox_3d,history = index_dict['R'],index_dict['T'],index_dict['S'],index_dict['correct'],index_dict['offset_t'],index_dict['offset_s'],index_dict['offset_r'],index_dict['r_correct'],index_dict['sym'],index_dict['S_normalised'],index_dict['T_gt'],index_dict['R_gt'],img_size,S_gt,use_roca_bbox,index_dict['bbox_3d'],index_dict['history']

        return detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history


    def get_infos_from_roca(self,index,just_classifier):


        detection_name,gt_name = self.index_to_names[index]
        detection_index = int(detection_name.split('_')[-1].split('.')[0])

        infos = self.roca_all_preds[gt_name][detection_index]

        K = torch.Tensor(infos["associated_gt_infos"]['K'])
        bbox = infos['bbox']

        use_roca_bbox = True
        R,model_path,S,T = self.get_rotation_modelpath_S_T(infos)
        model_3d_name = model_path.split('/')[1] + '_' + model_path.split('/')[2] + '.npz'
        category = infos['category'].replace('bookcase','bookshelf')
        f = infos["associated_gt_infos"]['focal_length']

        R_gt = infos["associated_gt_infos"]["rot_mat"]
        T_gt = infos["associated_gt_infos"]["trans_mat"]
        S_gt = infos["associated_gt_infos"]["scaling"]

        img_size = infos["associated_gt_infos"]['img_size']

        assert len(bbox) == 4,bbox
        augmented_bbox = self.augment_bbox(bbox,img_size)


        R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,bbox_3d = self.get_R_and_T_and_S(augmented_bbox,R_gt,T_gt,f,img_size,model_path,S_gt,category,just_classifier,S,T,R)

        history = self.initialise_history(R,T,S)

        return detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history


    def get_infos_from_roca_eval_without_gt(self,index,just_classifier):


        detection_name,gt_name = self.index_to_names[index]
        detection_index = int(detection_name.split('_')[-1].split('.')[0])

        infos = self.roca_all_preds[gt_name][detection_index]

        K = torch.Tensor(infos["associated_gt_infos"]['K'])
        bbox = infos['bbox']

        use_roca_bbox = True
        R,model_path,S,T = self.get_rotation_modelpath_S_T(infos,detection_name=detection_name)
        model_3d_name = model_path.split('/')[1] + '_' + model_path.split('/')[2] + '.npz'
        category = infos['category'].replace('bookcase','bookshelf')
        f = infos["associated_gt_infos"]['focal_length']

        T_gt = [0,0,0]
        S_gt = [0,0,0]
        R_gt = [[1,0,0],[0,1,0],[0,0,1]]
        img_size = infos["associated_gt_infos"]['img_size']

        assert len(bbox) == 4,bbox
        augmented_bbox = self.augment_bbox(bbox,img_size)

        R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,bbox_3d = self.get_R_and_T_and_S(augmented_bbox,R_gt,T_gt,f,img_size,model_path,S_gt,category,just_classifier,S,T,R)

        history = self.initialise_history(R,T,S)

        return detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history


    def get_rotation_modelpath_S_T(self,infos,detection_name=None):
        if self.eval_params["what_rotation"] == 'roca' or self.eval_params["what_rotation"] == 'roca_init':
            q = infos['q']
            q = [q[1],q[2],q[3],q[0]]
            R = scipy_rot.from_quat(q).as_matrix()
            invert = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
            R = np.matmul(invert,R)
            R = R.tolist()


        elif self.eval_params["what_rotation"] == 'lines_gt':
            R = infos["r_from_lines"]
            # R = infos['rotations_from_lines']['stage_2']["selected_R_gt"]

        elif self.eval_params["what_rotation"] == 'no_init':
            R = scipy_rot.from_euler('zyx',[0,0,-30], degrees=True).as_matrix().tolist()
            # R = infos['rotations_from_lines']['stage_2']["selected_R_gt"]

        elif self.eval_params["what_rotation"] == 'init_for_classification':
            R = scipy_rot.from_euler('zyx',[0,0,-30], degrees=True).as_matrix()
            R = np.matmul(R,self.transform_Rs[self.config["evaluate"]["rotation_index"]])
            R = R.tolist()

        elif self.eval_params["what_rotation"] == 'init_from_best_rotation_index':
            R = scipy_rot.from_euler('zyx',[0,0,-30], degrees=True).as_matrix()
            rotation_index = self.best_rotation_indices[detection_name]
            R = np.matmul(R,self.transform_Rs[rotation_index])
            R = R.tolist()

        elif self.eval_params["what_rotation"] == 'lines_roca':
            R = infos['rotations_from_lines']['stage_2']["selected_R_roca"]
        
        elif self.eval_params["what_rotation"] == 'gt':
            R = infos["associated_gt_infos"]["rot_mat"]


        if self.eval_params["what_retrieval"] == 'roca':
            model_path = "model/{}/{}/model_normalized.obj".format(infos['category'],infos['scene_cad_id'][1])

        elif self.eval_params["what_retrieval"] == 'gt':
            model_path = infos["associated_gt_infos"]["model"]



        if self.eval_params["what_scale"] == 'roca':
            S = infos['s']
        elif self.eval_params["what_scale"] == 'gt':
            S = infos["associated_gt_infos"]["scaling"]
        elif self.eval_params["what_scale"] == 'median':
            S = self.scaling_medians[model_path.split('/')[2]]
        elif self.eval_params["what_scale"] == 'pred':
            S = None

        
        if self.eval_params["what_translation"] == 'roca':
            t = infos['t']
            invert = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
            T = np.matmul(invert,t).tolist()

        elif self.eval_params["what_translation"] == 'gt':
            T = infos["associated_gt_infos"]["trans_mat"]
        
        elif self.eval_params["what_translation"] == 'pred':
            T = None

        return R,model_path,S,T

    def add_random_points(self,points_2d,img_size):
        n_samples = self.config["data"]["number_random_points"]
        x_coords = torch.rand((n_samples,),device=self.device) * img_size[0]
        y_coords = torch.rand((n_samples,),device=self.device) * img_size[1]
        points_sampled = torch.stack([x_coords,y_coords],dim=1)
        all_points = torch.cat([points_2d,points_sampled],dim=0)
        return all_points


    def __getitem__(self, tuple_index_dict_just_classifier):
        index_dict,just_classifier = tuple_index_dict_just_classifier
        t1 = time.time()
        if type(index_dict) == dict:
            if self.kind == 'train' or self.kind == 'val':
                detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history = self.get_infos_from_index_dict(index_dict)
            elif self.kind == 'val_roca':
                detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history = self.get_infos_from_index_dict_roca(index_dict)
        else:
            index = index_dict
            if self.kind == 'train' or self.kind == 'val':
                
                # if "data_04_small/train" in self.config["data"]["dir_path_2d_train"]:
                #     index = index  % 10
                random_number = random.random()
                if random_number < self.config["hard_example_mining"]["percentage_hard_examples"] and self.kind == 'train' and self.epoch > 0 and self.config["hard_example_mining"]["use"] == True:
                    detection_name,gt_name,model_3d_name,gt_infos,K,R,T,S,correct,offset_t,offset_s,r_correct,sym,S_normalised,bbox,augmented_bbox = self.get_infos_from_hard_example(index)
                else:
                    detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history = self.get_infos_from_index(index,just_classifier)

            elif self.kind == 'val_roca':
                if self.use_all_images == False:
                    detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history = self.get_infos_from_roca(index,just_classifier)
                elif self.use_all_images == True:
                    detection_name,gt_name,model_3d_name,K,R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,T_gt,R_gt,bbox,augmented_bbox,img_size,S_gt,use_roca_bbox,bbox_3d,history = self.get_infos_from_roca_eval_without_gt(index,just_classifier)

        t2 = time.time()


        if self.config['data']['rerender_points'] == True:
            info_3d,reprojected_points,name_orientation = self.create_embedding_3d(model_3d_name,R,T,S,img_size,K)
        elif self.config['data']['rerender_points'] == False:
            R_dummy = torch.eye(3)
            T_dummy = torch.Tensor([0,0,3])
            S_dummy = torch.ones(3)
            info_3d,reprojected_points,name_orientation = self.create_embedding_3d(model_3d_name,R_dummy,T_dummy,S_dummy,img_size,K)
        t3 = time.time()
        normals,depth = self.load_normals_and_depth(gt_name)
        img_rgb = self.load_rgb(gt_name)

        
        t4 = time.time()
        if self.config['data']['use_canny'] == False:
            lines_2d = self.process_lines(detection_name)
            info_2d = self.create_points(lines_2d[:self.config['data']['n_lines_2d']],augmented_bbox,bbox,img_size,self.config["data"]["img_size"],self.config['data']['n_points_per_line'],normals,depth,reprojected_points)
        elif self.config['data']['use_canny'] == True:
            if self.config['data']['use_all_points_2d'] == False:
                if self.config['data']["use_random_points"] == False:
                    points_2d = self.process_canny_points(gt_name.replace('.json','.npy'))
                elif self.config['data']["use_random_points"] == True:
                    points_2d = torch.zeros(size=(0,2)).to(self.device)
                    points_2d = self.add_random_points(points_2d,img_size)
            elif self.config['data']['use_all_points_2d'] == True:
                points_2d = self.sample_all_points()

            # info_2d = self.create_points_canny(points_2d[:self.config['data']['n_points_canny']],augmented_bbox,bbox,img_size,self.config["data"]["img_size"],self.config['data']['n_points_per_line'],normals,depth,K,reprojected_points)
            info_2d = self.create_points_canny(points_2d,augmented_bbox,bbox,img_size,self.config["data"]["img_size"],self.config['data']['n_points_per_line'],normals,depth,K,reprojected_points,img_rgb)

        t5 = time.time()

        list_info = []

        if self.config["data"]["input_RST"] == True:
            assert self.config["data"]["input_RST_and_CAD_ID"] == False
            info_RST = self.create_info_RST(R,S,T).to(self.device)
            list_info.append(info_RST)

        if self.config["data"]["input_RST_and_CAD_ID"] == True:
            assert self.config["data"]["input_RST"] == False
            info_RST = self.create_info_RST_plus_modelid(R,S,T,model_3d_name,sym,just_classifier).to(self.device)
            list_info.append(info_RST)


        if self.config["data"]["add_history"] == True:
            list_info.append(torch.Tensor(history).to(self.device))
        

        list_info.append(info_2d)

        if self.config["data"]["use_3d_points"] == True:
            list_info.append(info_3d)

        t6 = time.time()
        info_combined = torch.cat(list_info,dim=0)
        max_n_points = self.get_number_total_points()
        # print('max n oints',max_n_points)
        padded_info = torch.zeros((max_n_points,self.config["data"]["dims_per_pixel"]),dtype=torch.float32,device=self.device)

        assert padded_info.shape[0] >= info_combined.shape[0], (padded_info.shape,info_combined.shape) 

        padded_info[:min(info_combined.shape[0],max_n_points),:] = info_combined[:min(info_combined.shape[0],max_n_points),:]
        # print('padded info',padded_info.shape)

        extra_infos = {'detection_name': detection_name,'gt_name': gt_name,'model_3d_name': model_3d_name, 'category': model_3d_name.split('_')[0],"offset_t":offset_t,"offset_s":offset_s,"offset_r":offset_r,"r_correct":r_correct,"sym":sym,"name_orientation":name_orientation,"roca_bbox":use_roca_bbox,"R":R,"T":T,"S":S,"K":K,"correct":correct,"S_normalised":S_normalised,"img_size":img_size,'S_gt':S_gt,'T_gt':T_gt,'R_gt':R_gt,'bbox':bbox,'augmented_bbox':augmented_bbox,'bbox_3d':bbox_3d,"history":history}
        

        assert offset_r[3] > -0.0001, offset_r[3]
        
        target = self.create_targets(correct,offset_t,offset_s,offset_r)
        target = target.to(self.device)
        t7 = time.time()

       
        return padded_info,target,extra_infos

    def create_targets(self,correct,offset_t,offset_s,offset_r):
        if self.config['model']['regress_offsets'] == True:
            target = torch.cat([torch.Tensor(correct),torch.Tensor(offset_t),torch.Tensor(offset_s),torch.Tensor(offset_r)],dim=0)
        else:
            target = torch.Tensor(correct)
        return target

        # return target_t,target_s,target_r,target_sym,target_correct

    def get_info_roca_bbox(self,gt_infos):
        if self.kind == 'train':
            return False
        if self.kind == 'val':
            return gt_infos['use_roca_bbox']

    def get_number_total_points(self):

        points_bbox = 4 * self.config['data']['n_points_per_line']

        if self.config["data"]["input_RST"] == True or self.config["data"]["input_RST_and_CAD_ID"] == True:
            points_RST = 15
        elif self.config["data"]["input_RST"] == False:
            points_RST = 0

        if self.config["data"]["add_history"] == True:
            points_history = 15
        elif self.config["data"]["add_history"] == False:
            points_history = 0

        if self.config['data']['use_canny'] == False:
            points_2d = self.config['data']['n_lines_2d'] * self.config['data']['n_points_per_line'] * self.config['data']["number_locations_per_pixel_2d"]
        elif self.config['data']['use_canny'] == True:

            if self.config['data']['use_all_points_2d'] == False:
                if self.config['data']["use_random_points"] == False:
                    points_2d = self.config['data']['n_points_canny']
                elif self.config['data']['use_random_points'] == True:
                    points_2d = self.config['data']["number_random_points"]
            
            elif self.config['data']['use_all_points_2d'] == True:
                points_2d = self.config['data']["size_preloaded"][0] * self.config['data']["size_preloaded"][1]

        if self.config['data']['use_3d_points'] == True:
            points_reprojected = self.config['data']['n_points_3d'] * 3

            if self.config['data']["use_reprojected_points_as_query"] == True:
                points_2d += self.config['data']['n_points_3d'] * self.config['data']["number_locations_per_pixel_reprojected"]

        elif self.config['data']['use_3d_points'] == False:
            points_reprojected = 0

        if self.config["data"]["add_samples_from_within_bbox"] == True:
            points_2d += self.config['data']['number_of_samples_from_within_bbox'] * self.config['data']["number_locations_per_pixel_2d"]

        # print('points_bbox',points_bbox)
        # print(' points reproj', points_reprojected)
        # print('2d',points_2d)
        # print('points rst',points_RST)
        # print('points hist',points_history)
        return points_bbox + points_reprojected + points_2d + points_RST + points_history

    def reproject_3d_points(self,points_3d,R,T,img_size,config,K):

        img_size_reshaped = torch.Tensor([img_size[0],img_size[1]]).unsqueeze(0).repeat(points_3d.shape[0],1).to(get_device(points_3d))
        # pix = points_3d_to_pixel(points_3d,R,T,f,sw,img_size_reshaped,w)
        pix,mask = points_3d_to_pixel_from_calibration(points_3d,R,T,K)


        return pix,mask

    def get_random_mask(self, length, lower_bound):
        if self.kind == 'train':
            percentage = np.random.uniform(lower_bound,1)
        elif self.kind == 'val' or self.kind == 'val_roca':
            percentage = 1

        mask = np.random.uniform(0,1,size=(length,))
        mask = mask < percentage
        return torch.from_numpy(mask).to(self.device)


    def get_visibility_3d_points_ignore(self,model_3d_name,R,n_points):
        name = 'elev_015_azim_0.0'
        visibility = torch.ones(n_points,device=self.device).bool()
        return visibility,name

    def create_embedding_3d(self,model_3d_name,R,T,S,img_size,K):
        # points = self.lines_3d[model_3d_name]['points']
        # normals = self.lines_3d[model_3d_name]['normals']

        # print('create embedding 3d')
        R = torch.Tensor(R).to(self.device)
        T = torch.Tensor(T).to(self.device)
        t1 = time.time()

        points = self.lines_3d[model_3d_name.split('.')[0] + '_points'].to(self.device)
        normals = self.lines_3d[model_3d_name.split('.')[0] + '_normals'].to(self.device)
        assert len(points) == len(normals)
        assert normals.shape[1] == 3
        assert normals.shape[2] == 3

        t2 = time.time()

        visibility,name_orientation = self.get_visibility_3d_points_ignore(model_3d_name,R,points.shape[0])
        t3 = time.time()

        mask = self.get_random_mask(len(points),self.config["data_augmentation"]["percentage_points_3d"])
        points = points[mask]
        normals = normals[mask]
        visibility = visibility[mask]
        t4 = time.time()

        t5 = time.time()
        # print(t1_1 - t1,t1_2 - t1_1,t1_3 - t1_2,t2 - t1_3)

        # points = points * torch.Tensor(gt_infos['scaling'])
        points = points * torch.Tensor(S).to(self.device) 

        reprojected_points,mask_z_greater_0 = self.reproject_3d_points(points,R,T,img_size,self.config,K)
        reprojected_points = reprojected_points[mask_z_greater_0]
        normals = normals[mask_z_greater_0]
        points = points[mask_z_greater_0]
        visibility = visibility[mask_z_greater_0]


        reshaped_size = torch.Tensor([img_size[0],img_size[1]]).to(self.device).repeat((reprojected_points.shape[0],1))
        normalised_points = reprojected_points / reshaped_size
        t6 = time.time()
        # points_plus_channel = torch.cat([normalised_points,2*torch.ones_like(normalised_points[:,:1]).to(self.device)],dim=1)
        
        # 2 for invisible, 3 for visible
        channel = (1 * visibility + 2).unsqueeze(1)
        points_plus_channel = torch.cat([normalised_points,channel],dim=1)
        cc1 = torch.transpose(torch.matmul(R,torch.transpose(points,-1,-2)),-1,-2) + T

        depth = torch.linalg.norm(cc1,dim=-1).unsqueeze(1)

        # can also keep depth here
        # if self.config['data']['use_depth'] == False:
            # depth = depth * 0
        
        t7 = time.time()
        # because have 3 normals per point
        normals_rotated = torch.transpose(torch.matmul(R,torch.transpose(normals.view(-1,3),-1,-2)),-1,-2)
        r_around_z = torch.Tensor([[-1,0,0],[0,-1,0],[0,0,1]]).to(self.device)
        normals_rotated = torch.transpose(torch.matmul(r_around_z,torch.transpose(normals_rotated.view(-1,3),-1,-2)),-1,-2)

        # can keep normal values here ?
        if self.config['data']['use_normals'] == False:
            normals_rotated = normals_rotated * 0


        list_for_concat = [torch.repeat_interleave(points_plus_channel,repeats=3,dim=0),normals_rotated,torch.repeat_interleave(depth,repeats=3,dim=0)]

        if self.config["data"]["input_3d_coords"] == True:
            list_for_concat.append(torch.repeat_interleave(cc1,repeats=3,dim=0))

        if self.config["data"]["input_rgb"] == True:
            list_for_concat.append(torch.repeat_interleave(cc1,repeats=3,dim=0) * 0)

        combined = torch.cat(list_for_concat,dim=1)

        assert torch.all(combined[:,2] < 3.5),combined[:,2]

        reshaped_size_new_img = torch.Tensor(self.config['data']['img_size']).to(self.device).unsqueeze(0).repeat((reprojected_points.shape[0],1))
        reprojected_points = normalised_points * reshaped_size_new_img

        reprojected_points = reprojected_points

        if self.config['data']['use_reprojected_points_as_query'] == False:
            reprojected_points = None
        t8 = time.time()
        

        # self.add_times('t2-t1',t2-t1)
        # self.add_times('t3-t2',t3-t2)
        # self.add_times('t4-t3',t4-t3)
        # self.add_times('t5-t4',t5-t4)
        # self.add_times('t6-t5',t6-t5)
        # self.add_times('t7-t6',t7-t6)
        # self.add_times('t8-t7',t8-t7)

        # self.print_times()
        # print('create embedding 3d',t2-t1,t3-t2,t4-t3,t5-t4)
        return combined,reprojected_points,name_orientation

    def add_times(self,name,value):
        if name not in self.times:
            self.times[name] = [value]
        else:
            self.times[name].append(value)

    def print_times(self):
        for name,times in self.times.items():
            print('time:', name, 'mean: ', np.mean(times), 'median: ', np.median(times),np.std(times))
        # pass




       
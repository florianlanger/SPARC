from asyncio import run_coroutine_threadsafe
from operator import index
from tkinter import OFF
from xml.parsers.expat import model
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
import quaternion
from copy import deepcopy


from SPARC.utilities import load_json
from SPARC.utilities import get_model_to_infos_scannet

from SPARC.dataset.utilities_previous_probabilistic_formulation import lines_3d_to_pixel,get_device,sample_points_from_lines

def load_numpy_dir(dir,max_n=10000000):

    all_lines = torch.load(dir + '.pt')

    # all_lines = {}
    # n_lines = []
    # for file in tqdm(sorted(os.listdir(dir))[:max_n]):
    #     lines = torch.from_numpy(np.load(dir + '/' + file))
    #     # if lines_2d.shape[0] != 0:
    #     n_lines.append(lines.shape[0])
    #     all_lines[file] = lines
    return all_lines


def get_gt_infos(gt_dir,max_n=10000000):

    # all_infos = {}
    # for file in tqdm(sorted(os.listdir(gt_dir))[:max_n]):
    #     with open(gt_dir + '/' + file,'r') as f:
    #         all_infos[file] = json.load(f)

    with open(gt_dir + '.json','r') as f:
        all_infos = json.load(f)

    return all_infos

# def get_gt_infos_by_detection(gt_dir,max_n=10000000):

#     all_infos = {}
#     for file in tqdm(sorted(os.listdir(gt_dir))[:max_n]):
#         with open(gt_dir + '/' + file,'r') as f:
#             gt_infos = json.load(f)

#         infos_img = {"img":gt_infos["img"],"img_size":gt_infos["img_size"],"focal_length":gt_infos["focal_length"]}

#         for i in range(len(gt_infos['objects'])):
#             detection_name = file.split('.')[0] + '_' + str(gt_infos['objects'][i]['index']).zfill(2) + '.npy'
#             all_infos[detection_name] = {**infos_img,**gt_infos['objects'][i]}
#     return all_infos

def get_gt_infos_by_detection(gt_infos_all):

    all_infos = {}
    for file in sorted(gt_infos_all):
        gt_infos = gt_infos_all[file]

        infos_img = {"img":gt_infos["img"],"img_size":gt_infos["img_size"],"focal_length":gt_infos["focal_length"]}

        for i in range(len(gt_infos['objects'])):
            detection_name = file.split('.')[0] + '_' + str(gt_infos['objects'][i]['index']).zfill(2) + '.npy'
            all_infos[detection_name] = {**infos_img,**gt_infos['objects'][i]}
    return all_infos


def sample_relative_offset():
    limit = 0.2
    prob = 0.5

    if np.random.rand() < prob:
        relative_offset = np.random.uniform(-limit,limit,3)
    else:
        relative_offset = np.zeros(3)
    return relative_offset

def get_S_offset(gt_scaling_orig,gt_scaling_normalised,pred_S):
    # offset = gt_scaling_orig / pred_S - 1
    offset = gt_scaling_orig - pred_S
    return offset

def get_S(config,scaling_limits,model_to_infos,model_path,gt_scaling_orig,category,sampled_S,kind,decrease_range_factor):

    scaling_limits = scaling_limits[category]
    bbox_3d = np.array(model_to_infos[model_path.split('/')[1] + '_' + model_path.split('/')[2]]['bbox'])*2
    gt_scaling = bbox_3d * gt_scaling_orig
    

    if kind == 'train':

        scaling_limits_range = np.array(scaling_limits['max']) - np.array(scaling_limits['min'])

        scaling_limits_low = scaling_limits['min'] + scaling_limits_range * decrease_range_factor
        scaling_limits_high = scaling_limits['max'] - scaling_limits_range * decrease_range_factor
        S = sample_absolute(scaling_limits_low,scaling_limits_high)

    else:
        S = np.array(scaling_limits['mode'])
    
    S_normalised = np.zeros(3)
    relative_offset = get_S_offset(gt_scaling_orig,gt_scaling,S)
    s_correct = np.all(np.abs(relative_offset) < 0.2)


    return S,s_correct,relative_offset,S_normalised,bbox_3d

    

def sample_relative(S,lower,upper):
    return np.random.uniform(lower,upper,3) * S + S

def sample_absolute(lower,upper):
    return np.random.uniform(lower,upper)

def sample_S(gt_scaling,scaling_limits,sampling_info):

    S = sample_absolute(scaling_limits['min'],scaling_limits['max'])
    return S


def get_T(f,img_size,T_gt,config,bbox,threshold_correct,half_width_sampling_cube,kind,sampled_T=None):


    sensor_width = config['data']['sensor_width']


    offset = sample_offset_reprojected_in_bbox(bbox,img_size,sensor_width,f,T_gt,kind)


    T = np.array(T_gt) - offset

    correct = np.linalg.norm(T - T_gt) < threshold_correct
    return T,correct,offset


def get_T_for_regression(f,img_size,T_gt,config,bbox,threshold_correct,half_width_sampling_cube,kind,sampled_T=None):


    sensor_width = config['data']['sensor_width']

    offset = sample_offset_reprojected_in_bbox_for_regression(bbox,img_size,sensor_width,f,T_gt,kind)


    if sampled_T != None:
        offset = np.array(T_gt) - np.array(sampled_T)

    T = np.array(T_gt) - offset

    correct = np.linalg.norm(T - T_gt) < threshold_correct
    return T,correct,offset


def sample_offset_reprojected_in_bbox(bbox,img_size,sensor_width,f,T_gt,kind):

    img_size = np.array(img_size)
    T_gt = np.array(T_gt)

    z = 3

    if kind == 'train':
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        x_low = bbox[0] + bbox_width * 0.3
        x_high = bbox[0] + bbox_width * 0.7
        y_low = bbox[1] + bbox_height * 0.3
        y_high = bbox[1] + bbox_height * 0.7
        
        random_point_in_bbox = np.array([np.random.uniform(x_low,x_high),np.random.uniform(y_low,y_high)])
    else:
        random_point_in_bbox = np.array([(bbox[0]+ bbox[2]) / 2,(bbox[1]+ bbox[3]) / 2])
    # random_point_in_bbox = np.array([random.choice([bbox[0],bbox[2]]),random.choice([bbox[1],bbox[3]])])
    # random_point_in_bbox = np.array([bbox[0],bbox[1]])
    pb = - (random_point_in_bbox - img_size/2) * sensor_width / (img_size[0] * f)

    T_sampled = np.array([z * pb[0],z * pb[1],z])

    offset = T_gt - T_sampled
    return offset

def sample_offset_reprojected_in_bbox_for_regression(bbox,img_size,sensor_width,f,T_gt,kind):

    img_size = np.array(img_size)
    T_gt = np.array(T_gt)


    # z between 1 and 5
    if kind == 'train':
        z = np.random.uniform(1,5)
        random_point_in_bbox = np.array([np.random.uniform(bbox[0],bbox[2]),np.random.uniform(bbox[1],bbox[3])])
    else:
        # print('DEBUGGGGGGG')
        # z = 1
        z = 3
        random_point_in_bbox = np.array([(bbox[0]+ bbox[2]) / 2,(bbox[1]+ bbox[3]) / 2])

    pb = - (random_point_in_bbox - img_size/2) * sensor_width / (img_size[0] * f)

    T_sampled = np.array([z * pb[0],z * pb[1],z])

    offset = T_gt - T_sampled
    return offset



def get_T_limits(f,img_size,sensor_width,probabilistic_shape_config,bbox,gt_z):
    
    img_size = np.array(img_size)

    bbox = np.array(bbox)
    bbox_center = (bbox[0:2] + bbox[2:4]) / 2
    bbox_size = bbox[2:4] - bbox[0:2]


    T_lower_upper = np.zeros((2,2))
    for i,delta in enumerate(np.array([[probabilistic_shape_config['x']["range"],probabilistic_shape_config['y']["range"]],[-probabilistic_shape_config['x']["range"],-probabilistic_shape_config['y']["range"]]])):
        target_pixel = delta * bbox_size + bbox_center
        pb_xy = - (target_pixel - img_size/2) * sensor_width / img_size
        T_lower_upper[i] = pb_xy * gt_z / f
    
    xs = (T_lower_upper[0][0],T_lower_upper[1][0],probabilistic_shape_config['x']["steps"])
    ys = (T_lower_upper[0][1],T_lower_upper[1][1],probabilistic_shape_config['y']["steps"])


    # zs = (gt_z-probabilistic_shape_config['z']["range"],gt_z+probabilistic_shape_config['z']["range"],probabilistic_shape_config['z']["steps"])
    if probabilistic_shape_config["gt_z"] == "True":
        zs = (gt_z,gt_z,1)
    elif probabilistic_shape_config["gt_z"] == "False":
        zs = (1,probabilistic_shape_config['z']["range"],probabilistic_shape_config['z']["steps"])

    return (xs,ys,zs)

def sample_offset_random_from_limits(xs,ys,zs,gt_T):

    x = random.uniform(xs[0],xs[1])
    y = random.uniform(ys[0],ys[1])
    z = random.uniform(zs[0],zs[1])

    return np.array([x,y,z]) - gt_T

def sample_offset_around_threshold_cube(threshold_correct):

    delta = threshold_correct

    found_valid = False
    while not found_valid:
        x = random.uniform(-3*delta,3*delta)
        y = random.uniform(-3*delta,3*delta)
        z = random.uniform(-3*delta,3*delta)
        if not (x < delta and x > -delta and y < delta and y > -delta and z < delta and z > -delta):
            found_valid = True

    return np.array([x,y,z])

def sample_offset_large_cube(delta):


    x = random.uniform(-delta,delta)
    y = random.uniform(-delta,delta)
    z = random.uniform(-delta,delta)

    return np.array([x,y,z])



def sample_offset_threshold_cube(threshold_correct):
    delta = threshold_correct

    x = random.uniform(-delta,delta)
    y = random.uniform(-delta,delta)
    z = random.uniform(-delta,delta)

    return np.array([x,y,z])

def reproject_3d_lines(lines_3d,R,T,gt_infos,config):
    f = gt_infos["focal_length"]
    sw = config["data"]["sensor_width"]
    img_size = gt_infos["img_size"]
    w = img_size[0]
    # img_size_reshaped = torch.Tensor([img_size[1],img_size[0]]).unsqueeze(0).repeat(lines_3d.shape[0],1)
    img_size_reshaped = torch.Tensor([img_size[0],img_size[1]]).unsqueeze(0).repeat(lines_3d.shape[0],1)
    pix1_3d,pix2_3d = lines_3d_to_pixel(lines_3d,torch.Tensor(R),torch.Tensor(T),f,sw,img_size_reshaped,w)
    lines = torch.cat([pix1_3d,pix2_3d],dim=1).numpy()
    return lines

def reproject_3d_lines_v2(lines_3d,R,T,gt_infos,config):
    f = gt_infos["focal_length"]
    sw = config["data"]["sensor_width"]
    img_size = gt_infos["img_size"]
    w = img_size[0]
    # img_size_reshaped = torch.Tensor([img_size[1],img_size[0]]).unsqueeze(0).repeat(lines_3d.shape[0],1)
    img_size_reshaped = torch.Tensor([img_size[0],img_size[1]]).unsqueeze(0).repeat(lines_3d.shape[0],1).to(get_device(lines_3d))
    pix1_3d,pix2_3d = lines_3d_to_pixel(lines_3d,R,T,f,sw,img_size_reshaped,w)
    lines = torch.cat([pix1_3d,pix2_3d],dim=1)
    return lines

def reproject_3d_lines_v3(lines_3d,R,T,S,gt_infos,config):
    f = gt_infos["focal_length"]
    sw = config["data"]["sensor_width"]
    img_size = gt_infos["img_size"]
    w = img_size[0]
    # img_size_reshaped = torch.Tensor([img_size[1],img_size[0]]).unsqueeze(0).repeat(lines_3d.shape[0],1)
    img_size_reshaped = torch.Tensor([img_size[0],img_size[1]]).unsqueeze(0).repeat(lines_3d.shape[0],1).to(get_device(lines_3d))
    lines_3d = lines_3d * torch.cat([S,S],dim=1)
    pix1_3d,pix2_3d = lines_3d_to_pixel(lines_3d,R,T,f,sw,img_size_reshaped,w)
    lines = torch.cat([pix1_3d,pix2_3d],dim=1)
    return lines

def compute_lines_resized(list_lines,orig_img_size,target_size):
    ratio = np.array(target_size) / orig_img_size
    out_lines = []
    for lines in list_lines:
        out_lines.append(lines * np.concatenate([ratio,ratio],axis=0))
    return out_lines

def compute_lines_resized_torch(lines,orig_img_size,target_size):
    ratio = (torch.Tensor(target_size) / torch.Tensor(orig_img_size)).unsqueeze(0).repeat(lines.shape[0],2).to(get_device(lines))
    lines = lines * ratio
    return lines

def create_base_image(config,target_size,img_path=None,lines_2d=None,bbox=None):

    # if config["data"]["use_rgb"] == True:
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
    # else:
    img_shape = (target_size[1],target_size[0],3)
    img = torch.zeros(img_shape,dtype=torch.uint8)

    if config["data"]["use_lines_2d"] == True:
        # img = plot_matches_individual_color_no_endpoints(img,lines_2d[:,:2],lines_2d[:,2:4],line_colors=[0,255,0],thickness=1)
        img = plot_lines_3d_torch(lines_2d,img,lines_2d.shape[0],samples_per_pixel=1,channel=1).squeeze(0)
    if config["data"]["use_bbox"] == True:
        # img = draw_boxes(img,np.array([bbox]),thickness=1,color=(0,0,255))
        bbox_lines = bbox_to_lines(bbox)
        img = plot_lines_3d_torch(bbox_lines,img,bbox_lines.shape[0],samples_per_pixel=1,channel=2).squeeze(0)
    return img

def bbox_to_lines(bbox):
    # lines = []
    # lines.append([bbox[0],bbox[1],bbox[0],bbox[3]])
    # lines.append([bbox[0],bbox[1],bbox[2],bbox[1]])
    # lines.append([bbox[2],bbox[1],bbox[2],bbox[3]])
    # lines.append([bbox[0],bbox[3],bbox[2],bbox[3]])

    lines = torch.Tensor([[bbox[0],bbox[1],bbox[0],bbox[3]],
                          [bbox[0],bbox[1],bbox[2],bbox[1]],
                          [bbox[2],bbox[1],bbox[2],bbox[3]],
                          [bbox[0],bbox[3],bbox[2],bbox[3]]])
    return lines

def plot_lines_3d_torch(lines_2d_batched,base_img,n_lines_2d,samples_per_pixel=1,line_color=[255,0,0],channel=None):
        assert lines_2d_batched.shape[1] == 4 and len(lines_2d_batched.shape) == 2, lines_2d_batched.shape
        assert len(base_img.shape) == 3 and base_img.shape[2] == 3, base_img.shape
        assert lines_2d_batched.shape[0] % n_lines_2d == 0, (lines_2d_batched,n_lines_2d)
        t1 = time.time()
        n_images = lines_2d_batched.shape[0] // n_lines_2d

        device = get_device(lines_2d_batched)
        batched_images = base_img.unsqueeze(0).repeat(n_images,1,1,1).to(device)

        # max_length_lines = torch.max(torch.linalg.norm(lines_2d_batched[:,2:4] - lines_2d_batched[:,0:2],dim=1))
        max_length_lines = np.linalg.norm(base_img.shape[0:2])
        n_samples = int(np.round(max_length_lines.item() * samples_per_pixel))

        t2 = time.time()
        points = sample_points_from_lines(lines_2d_batched,n_samples)
        points = torch.round(points).long()
        assert points.shape == (lines_2d_batched.shape[0],n_samples,2), (points.shape,[lines_2d_batched.shape[0],n_samples,2])
        t3 = time.time()
        points = points.view(lines_2d_batched.shape[0]*n_samples,2)

        image_indices = torch.arange(0,n_images).long().repeat_interleave(n_lines_2d*n_samples).unsqueeze(1).to(device)

        full_indices = torch.cat([image_indices,points[:,1:2],points[:,0:1]],dim=1)

        assert full_indices.shape == (n_images*n_lines_2d*n_samples,3), (full_indices.shape,[n_images*n_lines_2d*n_samples,3])

        img_size_reshaped = torch.Tensor([base_img.shape[0],base_img.shape[1]]).unsqueeze(0).repeat(full_indices.shape[0],1).to(device)
        mask = torch.bitwise_and((full_indices[:,1:] >= 0).all(dim=1),(full_indices[:,1:] < img_size_reshaped).all(dim=1))
        masked_full_indices = full_indices[mask]

        if channel == None:
            batched_images[masked_full_indices[:,0],masked_full_indices[:,1],masked_full_indices[:,2],:] = torch.Tensor(line_color).to(torch.uint8).to(device).unsqueeze(0).repeat(masked_full_indices.shape[0],1)
        else:
            batched_images[masked_full_indices[:,0],masked_full_indices[:,1],masked_full_indices[:,2],channel] = 255*torch.ones(1,dtype=torch.uint8).to(device).repeat(masked_full_indices.shape[0])
        
        t4 = time.time()

        return batched_images

# def add_extra_channels(img,config,img_path):
#     img_for_concat = [img]

#     if config["data"]["use_rgb"] == True:
#         target_size = config["data"]["img_size"]
#         img_path_small = img_path.replace('/images/','/images_{}_{}/'.format(target_size[0],target_size[1]))
#         img_rgb = cv2.imread(img_path_small)
#         img_rgb = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)
#         img_for_concat.append(img_rgb)

#     if config["data"]["use_normals"] == True:
#         img = cv2.imread(img_path.replace('/images/','/norm/').replace('.jpg','.png'))
#         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         img_for_concat.append(img)

#     if config["data"]["use_alpha"] == True:
#         img = cv2.imread(img_path.replace('/images/','/alpha/').replace('.jpg','.png'))
#         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         img_for_concat.append(img)

#     return np.concatenate(img_for_concat,axis=2)

def get_extra_channels(config,img_path,target_size):
    img_for_concat = []

    if config["data"]["use_rgb"] == True:
        img_path_small = img_path.replace('/images/','/images_{}_{}/'.format(target_size[0],target_size[1]))
        img_rgb = cv2.imread(img_path_small)
        img_rgb = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)
        img_for_concat.append(img_rgb)

    if config["data"]["use_normals"] == True:
        img = cv2.imread(img_path.replace('/images/','/norm_{}_{}/'.format(target_size[0],target_size[1])).replace('.jpg','.png'))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_for_concat.append(img)

    if config["data"]["use_alpha"] == True:
        img = cv2.imread(img_path.replace('/images/','/alpha_{}_{}/'.format(target_size[0],target_size[1])).replace('.jpg','.png'))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_for_concat.append(img)

    return img_for_concat

def check_config(config):
    assert config["data"]["sample_T"]["percent_small"] + config["data"]["sample_T"]["percent_large"] <= 1.,(config["data"]["sample_T"])
    # assert config["data"]["sample_S"]["percent_small"] + config["data"]["sample_S"]["percent_large"] <= 1.,(config["data"]["sample_S"])
    # assert config["training"]["batch_size_val"] % config["training"]["val_grid_points_per_example"] == 0,(config["training"]["batch_size_val"],config["training"]["val_grid_points_per_example"])

    if config["data"]["use_crop"] == True:
        assert config["data"]["img_size"] == [128,96]

def get_datatype_to_channel_index(config):
    index_dict = {}
    channel_counter = 3
    if config["data"]["use_rgb"] == True:
        index_dict["rgb"] = channel_counter
        channel_counter += 3
    else:
        index_dict["rgb"] = None

    if config["data"]["use_normals"] == True:
        index_dict["normals"] = channel_counter
        channel_counter += 3
    else:
        index_dict["normals"] = None

    if config["data"]["use_alpha"] == True:
        index_dict["alpha"] = channel_counter
        channel_counter += 3
    else:
        index_dict["alpha"] = None
    return index_dict

class Dataset_lines(data.Dataset):
    def __init__(self,config,kind,use_all_images=False):
        'Initialization'
        # print('Len 50')
        dir_key = "dir_path_2d_" + kind
        self.dir_path_2d = config["data"][dir_key]
        self.use_all_images = use_all_images
        
        print('LOAD NORMALISED POINTS, RESCALED')
        self.lines_3d = torch.load(config["data"]["dir_path_3d_points_and_normals"]+ 'points_and_normals_combined_normalised.pt')
        self.model_name_to_index = self.get_model_name_to_index()

        if kind == 'train':
            self.gt_infos = get_gt_infos(self.dir_path_2d + '/gt_infos_{}'.format(config["data"]["objects_train"]),max_n=config["training"]['max_number'])
            # self.gt_infos_by_detection = get_gt_infos_by_detection(self.dir_path_2d + '/gt_infos_valid_objects',max_n=config["training"]['max_number'])
            self.gt_infos_by_detection = get_gt_infos_by_detection(self.gt_infos)
        
        elif kind == 'val':
            self.gt_infos = get_gt_infos(self.dir_path_2d + '/gt_infos_{}_roca_bbox'.format(config["data"]["objects_val"]),max_n=config["training"]['max_number'])
            
            self.gt_infos_by_detection = get_gt_infos_by_detection(self.gt_infos)

        elif kind == 'val_roca':
            if self.use_all_images == True:
                dir_key = "dir_path_2d_" + kind + '_all_images'
                self.dir_path_2d = config["data"][dir_key]
            self.roca_all_preds = load_json(config["data"]["path_roca_all_preds"])
            

        self.config = deepcopy(config)
        self.index_to_names = self.get_index_to_names(kind)
        self.threshold_correct = config["data"]["sample_T"]["threshold_correct_T"]
        self.image_size = config["data"]["img_size"]
        self.kind = kind
        self.half_width_sampling_cube = 1
        self.device = torch.device("cuda:{}".format(config["general"]["gpu"]))
        # self.device = torch.device("cuda")
        # assert False
        self.get_modelid_to_sym(config["data"]["dir_path_scan2cad_anno"])
        self.model_to_infos = get_model_to_infos_scannet(config["data"]["dir_path_scan2cad_anno"])
        self.scaling_limits = load_json(config["data"]["path_scaling_limits"])

        self.transform_Rs = [scipy_rot.from_euler('zyx',[0,y,0], degrees=True).as_matrix() for y in np.linspace(0,360,self.config["data_augmentation"]["N_rotations"],endpoint=False)]
                        # scipy_rot.from_euler('zyx',[0,90,0], degrees=True).as_matrix(),
                        # scipy_rot.from_euler('zyx',[0,180,0], degrees=True).as_matrix(),
                        # scipy_rot.from_euler('zyx',[0,270,0], degrees=True).as_matrix()]
        print('Len',self.__len__())

        check_config(config)

        # self.vis_distribution(self.lines_2d,kind + '_lines_2d.png')
        # self.vis_distribution(self.lines_3d,kind + '_lines_3d.png')

    def update_half_width_sampling_cube(self,epoch):
        self.half_width_sampling_cube = max(0.2,1-0.01*epoch)

    def get_modelid_to_sym(self,path_annos):
        annos = load_json(path_annos)
        id_to_sym = {}
        for i in range(len(annos)):
            for model in annos[i]["aligned_models"]:
                id_to_sym[model["id_cad"]] = model["sym"]
        self.id_to_sym = id_to_sym

    def __len__(self):
        if "data_04_small/train" in self.config["data"]["dir_path_2d_train"]:
            return len(self.index_to_names)
        else:
            # print('DEBUG only 100')
            # return 20
            return len(self.index_to_names)

    def vis_distribution(self,dict,name):
        n = [dict[key].shape[0] for key in dict]
        plt.hist(n,bins=100)
        plt.savefig(name)
        plt.close()

    def get_model_name_to_index(self):
        model_name_to_index = {}
        counter = 0
        for key in sorted(self.lines_3d):
            if '_normals' in key:
                continue
            model_name_to_index[key.replace('_points','')] = counter
            counter += 1
        return model_name_to_index

    def load_rendered_image(self,image_path):
        image = imageio.imread(image_path)[:,:,:3].astype(np.float32)/255.
        return image

    def load_compressed_file(self,path,type='numpy'):
        if type == 'numpy':
            data = np.load(path)
            torch_data = {}
            for key in tqdm(data.keys()):
                torch_data[key] = torch.from_numpy(data[key])
        elif type == 'torch':
            torch_data = torch.load(path)
        return torch_data


    def load_image_dir(self,dir,max_n=10000000,type='image'):

        all_lines = torch.load(dir + '.pt')

        return all_lines

    def load_single_image(self,path,type='image'):

        flag = cv2.IMREAD_COLOR
        if type == 'depth':
            flag = cv2.IMREAD_UNCHANGED

        loaded = cv2.imread(path,flag)
        if type == 'depth':
            loaded = loaded.astype(float) / 1000
        elif type == 'normal':
            loaded = cv2.cvtColor(loaded,cv2.COLOR_BGR2RGB)
            loaded = (- (np.array(loaded).astype(np.float32) / 255.0) + 0.5) * 2.0
            
        loaded = torch.from_numpy(loaded)

        return loaded


    def load_numpy_dir_compressed(self,dir,max_n=10000000):

        all_lines = {}
        for file in tqdm(sorted(os.listdir(dir))[:max_n]):
            
            # if 'table_843713faa2ee00cba5d9ad16964840ab.npz' not in file and 'cabinet_50c7e5dace1d5dffc6256085755e32ef' not in file:
            #     continue


            dict_of_arrays = np.load(dir + '/' + file,allow_pickle=True)
            all_lines[file] = {}
            for key in dict_of_arrays:
                all_lines[file][key] = torch.from_numpy(dict_of_arrays[key])

        return all_lines


    def convert_lines_3d(self,lines_3D):
        # lines_3D = lines_3D * torch.Tensor(scaling + scaling).unsqueeze(0).repeat(lines_3D.shape[0],1)
        lines_3D[:,3:6] = lines_3D[:,3:6] - lines_3D[:,:3]
        return lines_3D

    def augment_bbox(self,bbox,img_size):

        min_bbox_size = 10

        if self.kind == 'val' or self.kind == 'val_roca':
            return bbox

        else:
            max_change = self.config["data_augmentation"]["change_bbox_size_percent_img"] * max(img_size)

            valid_bbox = False
            while valid_bbox == False:
                offset = np.random.uniform(-max_change,max_change,size=4)
                new_bbox = offset + np.array(bbox)
                new_bbox = np.clip(new_bbox,np.zeros(4),np.array(img_size + img_size)-1)
                new_bbox = np.round(new_bbox)
                if (new_bbox[:2] + min_bbox_size < new_bbox[2:4]).all():
                    valid_bbox = True
            return new_bbox

    def augment_R(self,R):
        if self.kind == 'val' or self.kind == 'val_roca':
            # return np.array(R),np.array([1,0,0,0,1,0,0,0,1])
            return np.array(R),np.array([0,0,0,1])
        else:
            random_angles = np.random.uniform(-np.array(self.config["data_augmentation"]["change_R_angle_degree"]),self.config["data_augmentation"]["change_R_angle_degree"],size=3)
            
            # print('no random')
            # random_angles = np.array([0,0,40])
            
            inv_offset_R = scipy_rot.from_euler('zyx',random_angles, degrees=True)
            R = np.matmul(R,inv_offset_R.as_matrix())
            # offset_R = np.linalg.inv(inv_offset_R)
            # print('ORDER IMPORTANT')
            # offset_R = np.reshape(offset_R,-1)
            
            offset_R = inv_offset_R.inv().as_quat()

            return R,offset_R

    def get_R(self,R_gt,sampled_R,sym):
        if sampled_R is None:
            if self.kind == 'train' or self.kind == 'val':
                random_angles = np.random.uniform(-np.array(self.config["data_augmentation"]["change_R_angle_degree"]),self.config["data_augmentation"]["change_R_angle_degree"],size=3)
                inv_offset_R = scipy_rot.from_euler('zyx',random_angles, degrees=True)
                R = np.matmul(R_gt,inv_offset_R.as_matrix())
                R,r_correct,r_index = self.rotate_R_v2(R,sym)

        else:
            R = sampled_R
            r_correct = False
            r_index = 0

        return R,r_correct,r_index

    def get_R_for_regression(self,R_gt,sampled_R):
        if sampled_R == None:

            random_angles = np.random.uniform(-np.array(self.config["data_augmentation"]["change_R_angle_degree"]),self.config["data_augmentation"]["change_R_angle_degree"],size=3)
            inv_offset_R = scipy_rot.from_euler('zyx',random_angles, degrees=True)
            R = np.matmul(R_gt,inv_offset_R.as_matrix())
            offset_R = inv_offset_R.inv().as_quat()

        else:
            R = sampled_R
            offset_R = (scipy_rot.from_matrix(R).inv() * scipy_rot.from_matrix(R_gt)).as_quat()
        
        if offset_R[3] < 0:
            offset_R = offset_R * -1
        return R,offset_R


    def mask_random(self,array,lower_bound):

        percentage = np.random.uniform(lower_bound,1)

        if self.kind == 'val' or self.kind == 'val_roca':
            return array
        else:
            if array.shape[0] < 5:
                return array
            else:
                while True:
                    mask = np.random.uniform(0,1,size=array.shape[0])
                    mask = mask < percentage
                    new_array = array[mask]
                    if new_array.shape[0] > 3:
                        return array[mask]


    def create_image(self,lines_2d,lines_3d_reprojected,augmented_bbox,bbox,orig_img_size,img_path,target_size):
        
        lines_2d,lines_3d_reprojected,augmented_bbox,bbox = compute_lines_resized([lines_2d,lines_3d_reprojected,augmented_bbox,bbox],orig_img_size,target_size)
        lines_3d_reprojected = torch.from_numpy(lines_3d_reprojected)
        # img_path_small = img_path.replace('/images/','/images_{}_{}/'.format(target_size[0],target_size[1]))
        img = create_base_image(self.config,target_size,img_path=img_path,lines_2d=lines_2d,bbox=augmented_bbox)
        # img = plot_matches_individual_color_no_endpoints(img,lines_3d_reprojected[:,:2],lines_3d_reprojected[:,2:4],line_colors=[255,0,0],thickness=1)
        img = plot_lines_3d_torch(lines_3d_reprojected,img,lines_3d_reprojected.shape[0],samples_per_pixel=1,channel=0).squeeze(0)
        img = img.numpy()

        extra_channels = get_extra_channels(self.config,img_path,target_size)

        img = np.concatenate([img] + extra_channels,axis=2)
        
        return img,augmented_bbox

    def shuffle_tensor_first_dim(self,array):
        array = array[torch.randperm(array.shape[0])]
        return array

    def randomly_swap_endpoints(self,lines):
        assert len(lines.shape) == 2
        assert lines.shape[1] in [4,6]
        dims = int(lines.shape[1] / 2)


        mask = (np.random.rand(lines.shape[0]) > 0.5)
        new_lines = lines*0
        for i in range(lines.shape[0]):
            if mask[i]:
                new_lines[i,:dims] = lines[i,dims:]
                new_lines[i,dims:] = lines[i,:dims]
            else:
                new_lines[i,:] = lines[i,:]


        # lines = np.concatenate([lines[:,indices[:,0]:indices[:,0]+dims],lines[:,indices[:,1]:indices[:,1]+dims]],axis=1)
        return new_lines

    def process_lines(self,detection_name,model_3d_name):

        lines_2d = self.lines_2d[detection_name]
        lines_2d = lines_2d[:,[1,0,3,2]]
        lines_2d = self.shuffle_tensor_first_dim(lines_2d)
        lines_2d = self.randomly_swap_endpoints(lines_2d)

        lines_3d = self.lines_3d[model_3d_name]
        lines_3d = self.shuffle_tensor_first_dim(lines_3d)
        lines_3d = self.randomly_swap_endpoints(lines_3d)

        lines_3d = self.mask_random(lines_3d,self.config["data_augmentation"]["percentage_lines_3d"])
        lines_2d = self.mask_random(lines_2d,self.config["data_augmentation"]["percentage_lines_2d"])

        assert lines_2d.shape[0] != 0,lines_2d
        assert lines_3d.shape[0] != 0,lines_3d

        lines_3d = self.convert_lines_3d(lines_3d)

        return lines_2d,lines_3d

    def get_crop_indices(self,bbox,img_size):
        aspect_gt = img_size[0]/img_size[1]
        aspect_bbox = (bbox[2] - bbox[0])/(bbox[3] - bbox[1])

        target_percent = 0.7

        relative_margin = (1 - target_percent) / 2

        old_w = (bbox[2] - bbox[0])
        old_h = (bbox[3] - bbox[1])


        if aspect_gt > aspect_bbox:

            new_bbox_y_1 = bbox[1] - (relative_margin * old_h)  
            new_bbox_y_2 = bbox[3] + (relative_margin * old_h) 

            new_w = (new_bbox_y_2 - new_bbox_y_1) * aspect_gt
            
            center = (bbox[0] + bbox[2]) / 2
            new_bbox_x_1 = center - new_w / 2
            new_bbox_x_2 = center + new_w / 2

        else:
            new_bbox_x_1 = bbox[0] - (relative_margin * old_w)  
            new_bbox_x_2 = bbox[2] + (relative_margin * old_w)

            new_h = (new_bbox_x_2 - new_bbox_x_1) / aspect_gt
            
            center = (bbox[1] + bbox[3]) / 2
            new_bbox_y_1 = center - new_h / 2
            new_bbox_y_2 = center + new_h / 2
        
        add_offset = np.array([img_size[0],img_size[1],img_size[0],img_size[1]])
        return np.round([new_bbox_x_1,new_bbox_y_1,new_bbox_x_2,new_bbox_y_2]).astype(int) + 2*add_offset

    def resize_image(self,image,target_size):
        resized = []
        for i in range(image.shape[2]):
            resized.append(cv2.resize(image[:,:,i],target_size))
        return np.stack(resized,axis=2)

    def process_image(self,lines_3d,lines_2d,R,T,gt_infos,bbox,augmented_bbox):

        lines_3d_reprojected = reproject_3d_lines(lines_3d,R,T,gt_infos,self.config)
        img_path = self.dir_path_2d + '/images/' + gt_infos['img']

    
        if self.config["data"]["use_crop"] == False:
            img,_ = self.create_image(lines_2d,lines_3d_reprojected,augmented_bbox,bbox,gt_infos["img_size"],img_path,self.config["data"]["img_size"])

        elif self.config["data"]["use_crop"] == True:
            img,augmented_bbox = self.create_image(lines_2d,lines_3d_reprojected,augmented_bbox,bbox,gt_infos["img_size"],img_path,[256,192])
            crop_indices = self.get_crop_indices(augmented_bbox,[256,192])
            black_background = np.zeros((img.shape[0]*5,img.shape[1]*5,img.shape[2]),dtype=np.uint8)
            black_background[2*img.shape[0]:img.shape[0]*3,2*img.shape[1]:img.shape[1]*3,:] = img
            assert crop_indices[0] >= 0 and crop_indices[1] >= 0 and crop_indices[2] <= black_background.shape[1] and crop_indices[3] <= black_background.shape[0] and crop_indices[2] > crop_indices[0] and crop_indices[3] > crop_indices[1], (crop_indices,augmented_bbox)
            img = black_background[crop_indices[1]:crop_indices[3],crop_indices[0]:crop_indices[2],:]
            img = self.resize_image(img,[128,96])

        img = (img.astype(np.float32) / 255. - 0.5) * 2
        return img

    def rotate_R(self,R,sym):
        transform_Rs  = [scipy_rot.from_euler('zyx',[0,90,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,270,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,180,0], degrees=True).as_matrix()]

        if sym == '__SYM_NONE':
            index = np.random.randint(0,3)
            R = np.matmul(R,transform_Rs[index])
            r_correct = False
        elif sym == '__SYM_ROTATE_UP_2':
            index = np.random.randint(0,2)
            R = np.matmul(R,transform_Rs[index])
            r_correct = False
        else:
            r_correct = True
        
        return R,r_correct

    def rotate_R_v2(self,R,sym):
        n_rots = self.config["data_augmentation"]["N_rotations"]
        # index = np.random.randint(0,4)

        probs = [0.5] + [0.5 / (n_rots - 1 )] * (n_rots - 1)
        # probs = [0.4, 0.2, 0.2, 0.2]
        index = int(np.random.choice(n_rots, 1, p=probs))

        R = np.matmul(R,self.transform_Rs[index])

        r_correct = False

        if sym == '__SYM_NONE':
            if index in [0]:
                r_correct = True
        elif sym == '__SYM_ROTATE_UP_2':
            if index in [0,int(n_rots/2)]:
                r_correct = True
        elif sym == '__SYM_ROTATE_UP_4':
            if index in [0,int(n_rots/4),int(n_rots/2),int(3*n_rots/4)]:
                r_correct = True
        else:
            r_correct = True
        
        return R,r_correct,index
        

    def get_R_and_T(self,detection_name,gt_name,augmented_bbox):
        gt_infos = self.gt_infos_by_detection[detection_name]
        R = self.gt_infos_by_detection[detection_name]['rot_mat']

        rand_number = np.random.rand()

        model_id = gt_infos["model"].split('/')[2]
        sym = self.id_to_sym[model_id]

        if rand_number < self.config["data_augmentation"]["sample_wrong_R_percentage"] and self.kind == 'train':
            R = self.augment_R(R)
            R,r_correct = self.rotate_R(R,sym)
            T,t_correct,offset = get_T(self.gt_infos[gt_name],gt_infos,self.config,augmented_bbox,self.threshold_correct,self.half_width_sampling_cube)

        else:
            R = self.augment_R(R)
            r_correct = True
            T,t_correct,offset = get_T(self.gt_infos[gt_name],gt_infos,self.config,augmented_bbox,self.threshold_correct,self.half_width_sampling_cube)

        correct = r_correct and t_correct
        correct = correct * np.array([1]).astype(np.float32)

        return R,T,correct,offset,r_correct,sym


    def get_R_and_T_and_S(self,augmented_bbox,R_gt,T_gt,f,img_size,model_path,gt_scaling_orig,category,sample_for_classification=False,sampled_S=None,sampled_T=None,sampled_R=None):


        model_id = model_path.split('/')[2]
        sym = self.id_to_sym[model_id]

        
        if sample_for_classification == True:
            R,r_correct,r_index = self.get_R(R_gt,sampled_R,sym)
            T,t_correct,offset_t = get_T(f,img_size,T_gt,self.config,augmented_bbox,self.threshold_correct,self.half_width_sampling_cube,self.kind,sampled_T)
            
            decrease_range_factor = 0.0
            S,s_correct,offset_s,S_normalised,bbox_3d = get_S(self.config,self.scaling_limits,self.model_to_infos,model_path,gt_scaling_orig,category,sampled_S,self.kind,decrease_range_factor)

            correct = r_correct * np.array([1]).astype(np.float32)
            offset_s = offset_t * 0
            offset_r = np.ones(4) * 10
            offset_r[0] = r_index

        elif sample_for_classification == False:

            R,offset_r = self.get_R_for_regression(R_gt,sampled_R)
            T,t_correct,offset_t = get_T_for_regression(f,img_size,T_gt,self.config,augmented_bbox,self.threshold_correct,self.half_width_sampling_cube,self.kind,sampled_T)

            decrease_range_factor = 0.3
            S,s_correct,offset_s,S_normalised,bbox_3d = get_S(self.config,self.scaling_limits,self.model_to_infos,model_path,gt_scaling_orig,category,sampled_S,self.kind,decrease_range_factor)
            correct = True * np.array([1]).astype(np.float32)
            r_correct = True



        return R,T,S,correct,offset_t,offset_s,offset_r,r_correct,sym,S_normalised,bbox_3d


    def __getitem__(self, index):

        detection_name,gt_name,model_3d_name = self.index_to_names[index]
        gt_infos = self.gt_infos_by_detection[detection_name]
        
        lines_2d,lines_3d = self.process_lines(detection_name,model_3d_name)
        
        # print('self.gt_infos_by_detection',self.gt_infos_by_detection)
        bbox = self.gt_infos_by_detection[detection_name]['bbox']
        assert len(bbox) == 4,bbox
        augmented_bbox = self.augment_bbox(bbox,self.gt_infos_by_detection[detection_name]['img_size'])
        # augmented_bbox = bbox

        # R,T,correct,offset,r_correct,sym = self.get_R_and_T(detection_name,gt_name,augmented_bbox)
        R,T,S,correct,offset_t,offset_s,r_correct,sym,_ = self.get_R_and_T_and_S(detection_name,gt_name,augmented_bbox)
        lines_3d = lines_3d * torch.Tensor(S).repeat(2)
        
        img = self.process_image(lines_3d,lines_2d,R,T,gt_infos,bbox,augmented_bbox)

        extra_infos = {'detection_name': detection_name,'gt_name': gt_name,'model_3d_name': model_3d_name, 'category': model_3d_name.split('_')[0],"offset_t":offset_t,"offset_s":offset_s,"r_correct":r_correct,"sym":sym}
        
        return img,correct,extra_infos



       
from re import T
from tkinter import E
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import sys
# from SPARC.visualisation.visualise_3d.visualise_predictions_from_csvs_v3 import visualise, visualise_single_images import visualise
import torch
from scipy.spatial.transform import Rotation as scipy_rot
# np.set_printoptions(threshold=sys.maxsize)
from SPARC.visualisation.utils import load_rgb_image,draw_points_on_image,load_gt_reprojection,expand_indices
from SPARC.utilities import draw_text_block



def plot_points_preds_normals(points, labels,probabilities,correct,t_pred,s_pred,r_pred,t_correct,s_correct,r_correct,config,extra_infos,kind,no_render):
    # fig = plt.figure(figsize=(10, 10))

    if config['model']['type'] == 'pointnet':
        points = points.transpose(2, 1)

    points = points * np.array(config['data']['img_size'] + [1]*(config["data"]["dims_per_pixel"] - 2))


    n_examples = min(config["training"]["n_vis"],points.shape[0])

    n_rows = n_examples * 3
    n_col = 4

    all_images = []
    all_3d_vis = []
    for idx in np.arange(n_examples):
        

        text = ["gt:{},pred:{:.4f}".format(labels[idx].item(),probabilities[idx].item()),
                "r {} sym {}".format(extra_infos["r_correct"][idx].item(),extra_infos["sym"][idx]),
                "t off gt {}".format(np.round(extra_infos["offset_t"][idx],2).tolist()),
                "s off gt {}".format(np.round(extra_infos["offset_s"][idx],2).tolist()),
                "r off gt {}".format(np.round(extra_infos["offset_r"][idx],2).tolist())
                ]

        # single_example = points[idx,:,:]
        images = []
        # ax1 = fig.add_subplot(n_rows, n_col, 2*n_col*idx+1, xticks=[], yticks=[])
        # title = "gt:{},pred:{:.4f}".format(labels[idx].item(),probabilities[idx].item())
        # ax1.set_title(title,color=("green" if correct[idx] else "red"))
        original_image = load_rgb_image(config,kind,extra_infos["gt_name"][idx],normalised=False)

        img_drawn = draw_points_on_image(points[idx],original_image.copy())
        img_drawn = np.ascontiguousarray(img_drawn, dtype=np.uint8)

        draw_text_block(img_drawn,text,font_scale=1,font_thickness=1)
        images.append(img_drawn)
        # ax1.imshow(images[0])

        # ax2 = fig.add_subplot(n_rows, n_col, 2*n_col*idx+2, xticks=[], yticks=[])
        img,_ = load_image(config,kind,extra_infos["gt_name"][idx],type='normal')
        # ix_show = (19,20,50,63)
        ix_show = (30,31,27,40)
        # print('img  normals',img[ix_show[0]:ix_show[1],ix_show[2]:ix_show[3],:])
        # img[ix_show[0]:ix_show[1],ix_show[2]:ix_show[3]] = img[ix_show[0]:ix_show[1],ix_show[2]:ix_show[3]] * 0
        images.append(img)
        # ax2.imshow(images[1])

        # ax3 = fig.add_subplot(n_rows, n_col, 2*n_col*idx+3, xticks=[], yticks=[])
        img_normals = draw_normals_on_image(points[idx],config)
        images.append(img_normals)
        title = "t {} s {}".format(np.round(extra_infos["offset_t"][idx],2).tolist(),np.round(extra_infos["offset_s"][idx],2).tolist())
        # ax3.set_title(title,color='blue')
        # ax3.imshow(images[2])

        # ax = fig.add_subplot(n_rows, n_col, n_col*idx+3, xticks=[], yticks=[])
        # img,_ = load_image(config,kind,extra_infos["gt_name"][idx],type='rgb')
        # plt.imshow(img)


        # ax4 = fig.add_subplot(n_rows, n_col, 2*n_col*idx+4, xticks=[], yticks=[])
        img_depth_vis,img_depth_orig,max_depth = load_depth_image(config,kind,extra_infos["gt_name"][idx])

        # print('img depth orig',img_depth_orig[ix_show[0]:ix_show[1],ix_show[2]:ix_show[3]])
        # img_depth_orig[ix_show[0]:ix_show[1],ix_show[2]:ix_show[3]] = img_depth_orig[ix_show[0]:ix_show[1],ix_show[2]:ix_show[3]] * 0
        images.append(img_depth_vis)
        # plt.imshow(images[3])

        # ax5 = fig.add_subplot(n_rows, n_col, 2*n_col*idx+5, xticks=[], yticks=[])
        title = "r {} sym {}".format(extra_infos["r_correct"][idx].item(),extra_infos["sym"][idx])
        # ax5.set_title(title,color='blue')
        img_vis,_ = draw_depth_on_image(points[idx],config,max_depth,channel=0)
        images.append(img_vis)        
        # plt.imshow(images[4])

        # check_equal = img_depth[img_depth > 0] == img_depth_orig[img_depth > 0]
        # assert np.all(check_equal),(img_depth[img_depth > 0][~check_equal],img_depth_orig[img_depth > 0][~check_equal])

        # ax6 = fig.add_subplot(n_rows, n_col, 2*n_col*idx+6, xticks=[], yticks=[])
        img_depth_3d,img_orig_depth_reprojected = draw_depth_on_image(points[idx],config,max_depth,channel=2)
        # print('img depth 3d',img_depth_3d[ix_show[0]:ix_show[1],ix_show[2]:ix_show[3]])
        # img_depth_3d[ix_show[0]:ix_show[1],ix_show[2]:ix_show[3]] = img_depth[ix_show[0]:ix_show[1],ix_show[2]:ix_show[3]] * 0
        images.append(img_depth_3d)
        # plt.imshow(images[5])
        # ax6.set_title(extra_infos["detection_name"][idx])

        images_normals_3d = draw_normals_on_image_from_3d(points[idx],config)
        # for i in range(3):
        #     index_add = i + 7
        #     fig.add_subplot(n_rows, n_col, 2*n_col*idx+index_add, xticks=[], yticks=[])
        #     plt.imshow(images_normals_3d[i])

        images += images_normals_3d

        # ax10 = fig.add_subplot(n_rows, n_col, 2*n_col*idx+10, xticks=[], yticks=[])
        diff_depth = vis_diff_depth(img_orig_depth_reprojected,img_depth_orig,max_depth)
        # plt.imshow(diff_depth)
        images.append(diff_depth)

        # ax11 = fig.add_subplot(n_rows, n_col, 2*n_col*idx+10, xticks=[], yticks=[])
        diff_normals = vis_diff_normals(images[6:9],images[1])
        # plt.imshow(diff_depth)
        images.append(diff_normals)



        text += ["t pred {}".format(array_to_string(t_pred[idx])),"s pred " + array_to_string(s_pred[idx]),"r pred " + array_to_string(r_pred[idx]),"t correct " + str(t_correct[idx]),"s correct " + str(s_correct[idx]),"r correct " + str(r_correct[idx]),
        "s gt {}".format(array_to_string(extra_infos['S_gt'][idx])),"s start {}".format(array_to_string(extra_infos['S'][idx]))]

        # ax12 = fig.add_subplot(n_rows, n_col, 2*n_col*idx+11, xticks=[], yticks=[])
        img_rgb = load_gt_reprojection(config,extra_infos["gt_name"][idx])
        draw_text_block(img_rgb,text,font_scale=1,font_thickness=1)
        # plt.imshow(img_rgb)
        images.append(img_rgb)

        img_rgb = vis_render(extra_infos,idx,config,original_image.copy(),t_pred,s_pred,r_pred,no_render=no_render)
        draw_points_on_image(points[idx],img_rgb,size=1,which_channels=[3,2])
        images.append(img_rgb)
        img_rgb = vis_render(extra_infos,idx,config,original_image.copy(),t_pred,s_pred,r_pred,vis_offset=True,no_render=no_render)
        images.append(img_rgb)
        
        images += load_images_model_3d(extra_infos,idx,config)

        img_rgb = draw_rgb_on_image(points[idx],config,size_expand=1)
        images.append(img_rgb)

        all_images.append(images)
        if config["data"]["input_3d_coords"] == True:
            all_3d_vis.append(visualise_point_clouds(points[idx],config))


        # ax = fig.add_subplot(n_rows, n_col, n_col*idx+6, xticks=[], yticks=[])
        # img_rgb = load_rgb_image(config,kind,extra_infos["gt_name"][idx])
        # plt.imshow(img_rgb[...,::-1])
        # ax.set_title(extra_infos["detection_name"][idx])

    return all_images,all_3d_vis


def visualise_point_clouds(points,config):


    all_points = []
    vertex_colors = []

    # channels = [[0,4],[2,3]]
    channels = [[0],[2],[3],[4]]
    colors = [(0,0,255),(255,0,0),(0,255,0),(255,255,0)]
    # colors = [(0,0,255),(0,255,0)]


    for channel_indices,color in zip(channels,colors):

        # mask_channel = (torch.abs(points[:,2] - channel_indices[0]) < 0.1) | (torch.abs(points[:,2] - channel_indices[1]) < 0.1)
        mask_channel = (torch.abs(points[:,2] - channel_indices[0]) < 0.1)
        if torch.sum(mask_channel) == 0:
            continue
        points_channel = points[mask_channel,:].numpy()
        points_3d = points_channel[:,config["data"]["indices_3d"][0]:config["data"]["indices_3d"][1]]

        # vertex_color = np.expand_dims(np.array(color,dtype=int),0).repeat(points_3d.shape[0],0)
        # vertex_colors.append(vertex_color)
        vertex_colors.append(points_3d * 0 + color)
        all_points.append(points_3d)

    vertices = np.concatenate(all_points)
    vertex_colors = np.concatenate(vertex_colors)

    # print('vertices',vertices.shape)
    # print('vertex_colors',vertex_colors.shape)
    # print('vertex_colors',vertex_colors[:10],vertex_colors[-10:])

    return (vertices,vertex_colors)

def load_images_model_3d(extra_infos,idx,config):
    images = []

    names = [extra_infos["name_orientation"][idx],'elev_015_azim_67.5','elev_045_azim_157.5']

    kinds = ['mask_render_vis','mask_render_vis_plus_normals','mask_render_vis_plus_normals']

    model_name = extra_infos["model_3d_name"][idx].split('.')[0]

    for i in range(len(names)):
        name = names[i]
        path = config['data']["dir_path_3d_points_and_normals"] + kinds[i] + '/' + model_name + '_' + name + '.png'
        if os.path.exists(path):
            img = cv2.imread(path)
        else:
            img = np.zeros((256,256,3),dtype=np.uint8)
        draw_text_block(img,[name,model_name],top_left_corner=(20,190),font_scale=1,font_thickness=1)
        images.append(img)
    return images


def convert_K(K,width,height):

    # print('K',K)
    # print('width',width)
    # print('height',height)

    K[0,2] = -(K[0,2] - width/2)
    K[1,2] = -(K[1,2] - height/2)
    K = K/(width/2)
    K[2:4,2:4] = torch.Tensor([[0,1],[1,0]])
    # print('K',K)
    return K

def rescale_K(K,ratio):
    K[0,0] = K[0,0] * ratio
    K[1,1] = K[1,1] * ratio
    K[0,2] = K[0,2] * ratio
    K[1,2] = K[1,2] * ratio
    return K

def get_K_from_orig_image_to_render(K,orig_size,target_size):
    # assert orig_size[0] / orig_size[1] == target_size[0] / target_size[1], (orig_size,target_size)
    # most scannet images have 1296 x 968, which is not exactly 4:3
    assert np.abs(orig_size[1] / orig_size[0] - target_size[1] / target_size[0]) < 0.01, (orig_size,target_size)

    ratio = target_size[0] / orig_size[0]
    # print('K',K)
    # print('ratio',ratio)
    K = rescale_K(K,ratio)
    # print('K a',K)
    K = convert_K(K,target_size[0],target_size[1])
    K = torch.Tensor(K)
    # print('K here',K)
    return K

def load_4by4_from_txt(path):
    M = np.zeros((4,4))
    with open(path,'r') as f:
        content = f.readlines()
        for i in range(4):
            line = content[i].split()
            for j in range(4):
                M[i,j] = np.float32(line[j])
        return M

def array_to_string(array,sf=3):
    string = ''
    for i in range(array.shape[0]):
        string += '{:.3f} '.format(array[i])
    return string

def vis_render(extra_infos,idx,config,original_image,t_pred,s_pred,r_pred,vis_offset=False,no_render=False):
    if no_render == True:
        return original_image

    else:
        from SPARC.visualisation.vis_pose import load_mesh,render_mesh,overlay_rendered_image,render_mesh_from_calibration
        device = torch.device("cuda:{}".format(config["general"]["gpu"]))


        dir_shapes = config["data"]["dir_path_shapes_for_vis"]


        K = extra_infos["K"][idx].copy()

        orig_size = extra_infos["img_size"][idx]
        target_size = config["data"]["img_size"]
        K = get_K_from_orig_image_to_render(K,orig_size,target_size)
        K = K.to(device)


        if vis_offset == False:
            T = extra_infos['T'][idx]
            S = extra_infos['S'][idx]
            R = torch.Tensor(extra_infos['R'][idx])
        else:
            T = extra_infos['T'][idx] + t_pred[idx]
            # S = extra_infos['S'][idx] * (s_pred[idx] + 1)
            S = extra_infos['S'][idx] + s_pred[idx]

            # r_offset_pred = r_pred[idx].reshape(3,3)
            r_offset_pred = scipy_rot.from_quat(r_pred[idx]).as_matrix()
            R = torch.Tensor(np.matmul(extra_infos['R'][idx],r_offset_pred))
        
        # if config["data"]["sample_what"] ==  "T":
        #     S = np.array([extra_infos['S_gt'][idx][0],extra_infos['S_gt'][idx][1],extra_infos['S_gt'][idx][2]]).astype(np.float32)
        # else:


        full_path_model = dir_shapes + extra_infos["model_3d_name"][idx].split('_')[0] + '/' + extra_infos["model_3d_name"][idx].split('_')[1].split('.')[0] + '/model_normalized.obj'
    
        mesh = load_mesh(full_path_model,R,T,S,device)
        rendered_image = render_mesh_from_calibration(target_size[0],target_size[1],K,mesh,device)
        out_image = overlay_rendered_image(original_image,rendered_image)

        return out_image


def vis_diff_depth(reprojected_depth,full_depth,max_depth):

    show_max_depth = 1000

    reprojected_depth = reprojected_depth / 255. * max_depth
    full_depth = full_depth / 255. * max_depth

    diff = np.abs(reprojected_depth - full_depth)
    # now diff is in mm
    # clip to 1000 mm
    diff = np.clip(diff,0,show_max_depth)
    
    diff = (show_max_depth - diff) / show_max_depth
    # diff = diff / 1000
    diff = np.uint8(diff * 255)
    img = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    img[reprojected_depth == 0,:] = img[reprojected_depth == 0,:] * 0

    # print(diff[reprojected_depth > 0])
    # print(df)

    return img


def vis_diff_normals(reprojected_normals,full_normals):

    max_angle = 45 # degrees

    for i in range(3):
        reprojected_normals[i] = np.expand_dims(reprojected_normals[i],2)
    reprojected_normals = np.concatenate(reprojected_normals,axis=2)

    mask = np.all(np.abs(reprojected_normals[:,:,:,:] < 0.00001),(2,3))

    reprojected_normals = (reprojected_normals / (-255.) + 0.5) * 2

    full_normals = np.expand_dims(full_normals,2)
    full_normals = (full_normals / (-255.) + 0.5) * 2
    full_normals = np.tile(full_normals,(1,1,3,1))

    angles = np.arccos(np.clip(np.sum(reprojected_normals * full_normals,axis=3),-1,1))
    angles = np.abs(angles)
    angles = np.min(angles,axis=2)
    angles = angles * 180 / np.pi

    angles = np.clip(angles,0,max_angle)
    angles = angles / max_angle
    angles = 1 - angles

    assert np.all(angles <= 1) and np.all(angles >= 0) 
    angles = np.uint8(angles * 255)
    img = cv2.applyColorMap(angles, cv2.COLORMAP_JET)

    img[mask,:] = img[mask,:] * 0

    return img


def draw_normals_on_image(points,config,size_expand=1):

    # mask_channel = (torch.abs(points[:,2] - 2) < 0.1) | (torch.abs(points[:,2] - 4) < 0.1)
    mask_channel = (torch.abs(points[:,2] - 0) < 0.1) | (torch.abs(points[:,2] - 4) < 0.1)

    points_channel = points[mask_channel,:].numpy()
    img_size = config['data']['img_size']

    indices = np.round(points_channel[:,:2]).astype(np.int)

    indices = expand_indices(indices,size_expand)

    mask = np.all((indices >= np.zeros(2)) & (indices < np.array([img_size[0],img_size[1]]) -1 ),axis=1)
    indices = indices[mask,:]

    img_black = np.zeros([img_size[1],img_size[0],3],dtype=np.uint8)

    normals = np.round((points_channel[:,3:6] / 2. - 0.5) * (-255))

    assert np.min(normals) >= 0 and np.max(normals) < 256
    normals = normals.astype(np.uint8)

    normals = np.repeat(normals,(2*size_expand+1)**2,axis=0)
    normals = normals[mask,:]

    img_black[indices[:,1],indices[:,0],:] = normals

    return img_black

def draw_normals_on_image_from_3d(points,config,size_expand=1):

    img_size = config['data']['img_size']
    # all_imgs = []
    # for i in range(3):
    #     img_black = np.zeros([img_size[1],img_size[0],3],dtype=np.uint8)
    #     all_imgs.append(img_black)
    # return all_imgs

    mask_channel = (torch.abs(points[:,2] - 2) < 0.1) | (torch.abs(points[:,2] - 3) < 0.1)
    points_channel = points[mask_channel,:].numpy()
    img_size = config['data']['img_size']

    points_channel = points_channel.reshape(-1,3,config['data']['dims_per_pixel'])


    assert np.all(points_channel[:,0,:2] == points_channel[:,1,:2])
    assert np.all(points_channel[:,0,:2] == points_channel[:,2,:2])

    indices = np.round(points_channel[:,0,:2]).astype(np.int)

    indices = expand_indices(indices,size_expand)

    mask = np.all((indices >= np.zeros(2)) & (indices < np.array([img_size[0],img_size[1]]) -1 ),axis=1)
    indices = indices[mask,:]



    # mask = ((indices[:,0] >= 0) & (indices[:,0] < img_size[0])) & ((indices[:,1] >= 0) & (indices[:,1] < img_size[1]))
    # indices = indices[mask,:]
    normals = points_channel[:,:,3:6]
    normals = np.repeat(normals,(2*size_expand+1)**2,axis=0)
    normals = normals[mask,:,:]


    sort_indices = np.argsort(normals[:,:,0],axis=1)


    first = normals[np.arange(len(indices)),sort_indices[:,0],:]
    second = normals[np.arange(len(indices)),sort_indices[:,1],:]
    third = normals[np.arange(len(indices)),sort_indices[:,2],:]
    # print('first',first.shape)
    sorted_normals = [first,second,third]

    all_imgs = []
    for i in range(3):
        img_black = np.zeros([img_size[1],img_size[0],3],dtype=np.uint8)

        img_black[indices[:,1],indices[:,0],:] = np.round((sorted_normals[i] / 2. - 0.5) * (-255)).astype(np.uint8)
        all_imgs.append(img_black)

    return all_imgs

def draw_depth_on_image(points,config,max_depth,channel,size_expand=1):
    # points_channel = points[points[:,2] == channel,:].numpy()
    if channel == 0:
        mask_channel = (torch.abs(points[:,2] - 0) < 0.1)| (torch.abs(points[:,2] - 4) < 0.1)
    if channel == 2:
        mask_channel = (torch.abs(points[:,2] - 2) < 0.1) | (torch.abs(points[:,2] - 3) < 0.1)


    points_channel = points[mask_channel,:].numpy()
    img_size = config['data']['img_size']


    indices = np.round(points_channel[:,:2]).astype(np.int)

    indices = expand_indices(indices,size_expand)
    mask = np.all((indices >= np.zeros(2)) & (indices < np.array([img_size[0],img_size[1]]) ),axis=1)
    indices = indices[mask,:]


    depth = (points_channel[:,6] * 1000) / max_depth
    depth = np.repeat(depth,(2*size_expand+1)**2,axis=0)
    depth = depth[mask]
    depth = np.clip(depth,0,1)

    img_vis = np.zeros([img_size[1],img_size[0],3],dtype=np.uint8)
    img_orig_depth = np.zeros([img_size[1],img_size[0]])

    if depth.shape[0] != 0:
        normalised_depth = (1 - depth)
        values = np.uint8(normalised_depth * 255)
        colors = cv2.applyColorMap(values, cv2.COLORMAP_JET)
        colors = colors.squeeze(1)

        img_vis[indices[:,1],indices[:,0],:] = colors
        img_orig_depth[indices[:,1],indices[:,0]] = depth
    
    img_orig_depth = np.round((img_orig_depth * 255)).astype(np.uint8)

    return img_vis,img_orig_depth

def draw_rgb_on_image(points,config,size_expand=1):

    img_size = config['data']['img_size']
    img_vis = np.zeros([img_size[1],img_size[0],3],dtype=np.uint8)
    if config['data']["input_rgb"] == False:
        return img_vis


    mask_channel = (torch.abs(points[:,2] - 0) < 0.1)| (torch.abs(points[:,2] - 4) < 0.1)

    points_channel = points[mask_channel,:].numpy()


    indices = np.round(points_channel[:,:2]).astype(np.int)

    indices = expand_indices(indices,size_expand)
    mask = np.all((indices >= np.zeros(2)) & (indices < np.array([img_size[0],img_size[1]]) ),axis=1)
    indices = indices[mask,:]

    rgb = points_channel[:,config["data"]["indices_rgb"][0]:config["data"]["indices_rgb"][1]] * 255
    rgb = np.repeat(rgb,(2*size_expand+1)**2,axis=0)
    rgb = rgb[mask]
    rgb = np.clip(rgb,0,255)


    img_vis = np.zeros([img_size[1],img_size[0],3],dtype=np.uint8)

    if rgb.shape[0] != 0:
        img_vis[indices[:,1],indices[:,0],:] = rgb


    return img_vis


def load_image(config,kind,gt_name,type='rgb'):
    dir_key = "dir_path_2d_" + kind
    dir_path_2d = config["data"][dir_key]

    if type == 'rgb' or type == 'normal':
        if type == 'rgb':
            path = dir_path_2d + '/images_{}_{}/'.format(config['data']['img_size'][0],config['data']['img_size'][1]) + gt_name.replace('.json','.jpg')
        elif type == 'normal':
            path = dir_path_2d + '/{}_{}_{}/'.format(config["data"]["name_norm_folder"],config['data']['img_size'][0],config['data']['img_size'][1]) + gt_name.replace('.json','.png')
        assert os.path.exists(path), "image not found: {}".format(path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        max_depth = None

    # elif type == 'depth':
    #     path = dir_path_2d + '/depth_{}_{}/'.format(config['data']['img_size'][0],config['data']['img_size'][1]) + gt_name.replace('.json','.png')
    #     assert os.path.exists(path), "image not found: {}".format(path)
    #     depth = cv2.imread(path,cv2.IMREAD_UNCHANGED)

    #     max_depth = 7000

    #     depth = depth.astype(float)
    #     depth = np.clip(depth,0,max_depth)

    #     normalised = (max_depth - depth) / max_depth

    #     values = np.uint8(normalised * 255)
    #     image = cv2.applyColorMap(values, cv2.COLORMAP_JET)

    return image,max_depth


def load_depth_image(config,kind,gt_name):
    dir_key = "dir_path_2d_" + kind
    dir_path_2d = config["data"][dir_key]

    path = dir_path_2d + '/{}_{}_{}/'.format(config["data"]["name_depth_folder"],config['data']['img_size'][0],config['data']['img_size'][1]) + gt_name.replace('.json','.png')
    assert os.path.exists(path), "image not found: {}".format(path)
    depth = cv2.imread(path,cv2.IMREAD_UNCHANGED)

    max_depth = 5000

    depth = depth.astype(float)
    depth = np.clip(depth,0,max_depth)

    normalised = (max_depth - depth) / max_depth

    values = np.uint8(normalised * 255)
    img_depth_vis = cv2.applyColorMap(values, cv2.COLORMAP_JET)
    img_depth_orig = np.round((depth/max_depth * 255)).astype(np.uint8)

    return img_depth_vis,img_depth_orig,max_depth
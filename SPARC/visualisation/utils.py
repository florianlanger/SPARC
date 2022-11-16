from tkinter import Y
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def load_rgb_image(config,kind,gt_name,normalised=True):
    dir_key = "dir_path_2d_" + kind
    dir_path_2d = config["data"][dir_key]
    path_img = dir_path_2d + '/images_{}_{}/'.format(config['data']['img_size'][0],config['data']['img_size'][1]) + gt_name.replace('.json',config["data"]["image_file_ending"])
    assert os.path.exists(path_img),path_img
    bgr = cv2.imread(path_img)
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    img_size = tuple(config["data"]["img_size"])
    rgb = cv2.resize(rgb,img_size)
    if normalised == True:
        rgb = rgb.astype(float) / 255.
    else:
        rgb = rgb
    return rgb

def load_gt_reprojection(config,gt_name):

    rgb = np.zeros((96,128,3),dtype=np.uint8)
    img_size = tuple(config["data"]["img_size"])
    rgb = cv2.resize(rgb,img_size)
    # rgb = rgb.astype(float) / 255.
    return rgb

def show_rgb_plus_lines(img_rgb, img_lines):

    img_lines = (img_lines / 2.) + 0.5     # unnormalize
    img_lines = img_lines.numpy()
    img_lines = np.transpose(img_lines, (1, 2, 0))

    combined = (img_lines + img_rgb ) /2
    plt.imshow(combined)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = (img / 2.) + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def draw_points_on_image(points,img_rgb,size=1,which_channels=None):

    channel_indices = [0,1,4,3,2]
    colors = [(0,255,0),(0,0,255),(255,255,0),(255,0,0),(0,0,120)]
    if which_channels is not None:
        indices_sequence = [channel_indices.index(which_channel) for which_channel in which_channels]
        colors = [colors[index] for index in indices_sequence]
        channel_indices = which_channels

    for channel,color in zip(channel_indices,colors):
        points_channel = points[points[:,2] == channel,:][:,:2]

        
        indices = np.round(points_channel.numpy()).astype(np.int)
        indices = expand_indices(indices,size)
        mask = np.all((indices >= np.zeros(2)) & (indices < np.array([img_rgb.shape[1],img_rgb.shape[0]])),axis=1)
        indices = indices[mask,:]
    
        y = indices[:,1]
        x = indices[:,0]

        img_rgb[y,x,:] = img_rgb[y,x,:]* 0 + np.array(color)

    return img_rgb

def expand_indices(indices,size):
    # print('indices',indices[:30])
    size_x = np.tile(np.arange(-size,size+1),size*2+1)
    size_y = np.repeat(np.arange(-size,size+1),size*2+1)
    size_array = np.stack([size_x,size_y],axis=1)

    n_points = indices.shape[0]
    indices = np.repeat(indices,size_array.shape[0],axis=0)
    size_array = np.tile(size_array,(n_points,1))
    indices = indices + size_array
    # print('indices',indices[:30])
    return indices

def plot_points_channels(ax,channel_0,channel_1,channel_2,point_size):
    ax.scatter(channel_0[:,0],channel_0[:,1],channel_0[:,2], c='green',s=point_size,depthshade=False)
    ax.scatter(channel_1[:,0],channel_1[:,1],channel_1[:,2], c='blue',s=point_size,depthshade=False)
    ax.scatter(channel_2[:,0],channel_2[:,1],channel_2[:,2], c='red',s=point_size,depthshade=False)
    ax.set_xlim((0,128))
    ax.set_ylim((0,96))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax
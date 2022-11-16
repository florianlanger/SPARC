import torch
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
from numpy.lib.utils import info
import torch
import os
import sys
import json
# import quaternion
import shutil

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform,PerspectiveCameras,FoVPerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
from scipy.spatial.transform import Rotation as scipy_rot

def load_mesh(full_path,R,T,scaling,device,color=(1,1,1),remesh=False):
    assert os.path.exists(full_path),full_path

    if remesh == False:
        vertices_origin,faces,_ = load_obj(full_path, device=device,create_texture_atlas=False, load_textures=False)
        faces = faces[0]
    elif remesh == True:
        vertices_origin,faces = remesh(full_path,max_edge_length=0.1,device=device)
    R = torch.Tensor(R).to(device) 
    T = torch.Tensor(T).to(device)
    scaling = torch.Tensor(scaling).to(device)
    scaled_vertices = vertices_origin * scaling.unsqueeze(0).repeat(vertices_origin.shape[0],1)
    vertices = torch.transpose(torch.matmul(R,torch.transpose(scaled_vertices,0,1)),0,1) + T
    # if color == None:
    #     textures = Textures(verts_rgb=torch.ones((1,vertices.shape[0],3),device=device))
    # else:
    textures = Textures(verts_rgb=torch.ones((1,vertices.shape[0],3),device=device)*torch.Tensor(color).to(device))
    mesh = Meshes(verts=[vertices], faces=[faces],textures=textures)
    return mesh



def render_mesh(w,h,f,mesh,device,sw,flip=False):
    if w >= h:
        fov = 2 * np.arctan((sw/2)/f)
    elif w < h:
        fov = 2 * np.arctan(((sw/2) * h/w)/f)

    # fov = 2 * np.arctan(w/(2*1169))


    r_cam = torch.eye(3).unsqueeze(0).repeat(1,1,1)
    t_cam = torch.zeros((1,3))
    # print('CHANGE BACK render_mesh in vis_pose')
    cameras_pix = FoVPerspectiveCameras(device=device,fov = fov,degrees=False,R = r_cam, T = t_cam)
    # print('f',f,'r_cam',r_cam,'t_cam',t_cam,'ppoint',ppoint)
    # print('w',w,'h',h)
    # cameras_pix = PerspectiveCameras(device=device,focal_length = f,R = r_cam, T = t_cam,principal_point=ppoint,image_size=torch.Tensor([[h,w]]))
    # print('cameras_pix.get_full_projection_transform()',cameras_pix.get_full_projection_transform().get_matrix())
    #principal_point=ppoint

    raster_settings_soft = RasterizationSettings(image_size = max(w,h),blur_radius=0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras_pix,raster_settings=raster_settings_soft)
    renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=SoftPhongShader(device=device))

    image = renderer_textured(mesh,cameras=cameras_pix).cpu().numpy()[0,:,:,:]

    # # flip
    # image = image[::-1,::-1,:]
    # # crop
    # diff = w-h
    # image_cropped = image[int(diff/2):-int(diff/2),:,:]
    # # print(np.max(image_cropped))
    # # print(np.min(image_cropped))
    # image_clipped = np.clip(0,255,image_cropped*255).astype(int)
    # image = image_clipped
    # print(image[:3,:3])

    # crop
    if w >= h:
        image = image[int((w-h)/2):int((w+h)/2),:,:]
    elif w < h:
        image = image[:,int((h-w)/2):int((h+w)/2),:]

    if flip:
        image = image[::-1,::-1,:]

    return image

def render_mesh_from_calibration(w,h,K,mesh,device,r_cam=torch.eye(3),t_cam=torch.zeros(3),bin_size=0):

    r_cam = r_cam.unsqueeze(0)
    t_cam = t_cam.unsqueeze(0)
    cameras_pix = PerspectiveCameras(device=device,K = K.unsqueeze(0),R = r_cam, T = t_cam)

    raster_settings_soft = RasterizationSettings(image_size = max(w,h),blur_radius=0, faces_per_pixel=1,bin_size=bin_size)
    rasterizer = MeshRasterizer(cameras=cameras_pix,raster_settings=raster_settings_soft)
    renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=SoftPhongShader(device=device))

    image = renderer_textured(mesh,cameras=cameras_pix).cpu().numpy()[0,:,:,:]

    # crop
    if w >= h:
        image = image[int((w-h)/2):int((w+h)/2),:,:]
    elif w < h:
        image = image[:,int((h-w)/2):int((h+w)/2),:]

    return image


def overlay_rendered_image(original_image,rendered_image):
    h,w,_ = original_image.shape
    alpha = 255*np.ones((h,w,4),dtype=np.uint8)
    alpha[:,:,:3] = original_image
    alpha = np.clip(alpha,a_min=0,a_max=255)

    image = np.round((255*rendered_image)).astype(np.uint8)
    image[np.where((image == [255,255,255,0]).all(axis = 2))] = [0,0,0,0]
    overlayed = cv2.addWeighted(alpha[:,:,:3],0.5,image[:,:,:3],0.5,0)
    return overlayed

def overlay_rendered_image_v2(original_image,rendered_image):

    image = np.round((255*rendered_image)).astype(np.uint8)
    mask = np.where((image != [255,255,255,0]).all(axis = 2))
    original_image[mask] = 0.1 * original_image[mask[0],mask[1],:] +  0.9 * image[mask[0],mask[1],:3]
    original_image = np.clip(original_image,a_min=0,a_max=255).astype(np.uint8)
    return original_image

def just_rendered_image(rendered_image):

    image = np.round((255*rendered_image)).astype(np.uint8)
    # image[np.where((image == [255,255,255,0]).all(axis = 2))] = [255,255,255,0]
    return image[:,:,:3]

def plot_points(img,pixels_real,color=(0,255,0)):
    assert type(pixels_real).__module__ == 'numpy'

    size = int(max(img.shape) / 200)
    size = max(1,size)

    for pixels in pixels_real:
        if (pixels >= 0).all():
            try:
                img[pixels[0]-size:pixels[0]+size,pixels[1]-size:pixels[1]+size,:] = np.tile(np.array([[color]]),(2*size,2*size,1))
            except ValueError:
                pass

    return img

def plot_matches(img,pixels_real,pixels_rendered,line_color=(0,255,0)):
    assert pixels_real.shape == pixels_rendered.shape
    assert type(pixels_real).__module__ == 'numpy'
    assert type(pixels_rendered).__module__ == 'numpy'


    size = int(max(img.shape) / 200)
    size = max(1,size)


    kinds = [pixels_real,pixels_rendered]
    colors = [[255,0,0],[0,0,255],[0,255,0]]

    for color,kind in zip(colors,kinds):
        if len(kind.shape) != 1:
            for pixels in kind:
                if (pixels >= 0).all():
                    try:
                        img[pixels[0]-size:pixels[0]+size,pixels[1]-size:pixels[1]+size,:] = np.tile(np.array([[color]]),(2*size,2*size,1))
                    except ValueError:
                        pass

    
    scale = max(img.shape) / 500.
    thickness = max(int(np.round(scale)),1)
    for i in range(pixels_rendered.shape[0]):
        # if (pixels_rendered[i] >= 0).all() and (pixels_real[i] >= 0).all():
        cv2.line(img, tuple(pixels_rendered[i,::-1]), tuple(pixels_real[i,::-1]), line_color, thickness)

    return img


def plot_matches_individual_color(img,pixels_real,pixels_rendered,line_colors):
    assert pixels_real.shape == pixels_rendered.shape
    assert type(pixels_real).__module__ == 'numpy'
    assert type(pixels_rendered).__module__ == 'numpy'


    size = int(max(img.shape) / 200)
    size = max(1,size)


    kinds = [pixels_real,pixels_rendered]
    colors = [[255,0,0],[0,0,255],[0,255,0]]

    for color,kind in zip(colors,kinds):
        if len(kind.shape) != 1:
            for pixels in kind:
                if (pixels >= 0).all():
                    try:
                        img[pixels[0]-size:pixels[0]+size,pixels[1]-size:pixels[1]+size,:] = np.tile(np.array([[color]]),(2*size,2*size,1))
                    except ValueError:
                        pass

    
    scale = max(img.shape) / 500.
    thickness = max(int(np.round(scale)),1)


    for i in range(pixels_rendered.shape[0]):
        # if (pixels_rendered[i] >= 0).all() and (pixels_real[i] >= 0).all():

        cv2.line(img, tuple(pixels_rendered[i,::-1]), tuple(pixels_real[i,::-1]), line_colors[i], thickness)

    return img

def plot_matches_individual_color_no_endpoints(img,pixels1,pixels2,line_colors=[255,0,0],thickness=None):

    assert pixels1.shape == pixels2.shape
    assert pixels1.shape[1] == 2
    if type(line_colors[0]) != int:
        assert len(line_colors) == pixels1.shape[0]

    pixels1 = np.round(np.array(pixels1)).astype(int)
    pixels2 = np.round(np.array(pixels2)).astype(int)

    scale = max(img.shape) / 500.
    if thickness == None:
        thickness = max(int(np.round(scale)),1)

    if type(line_colors[0]) == int:
        line_colors = [line_colors] * pixels1.shape[0]

    # print('pixels1',pixels1)
    # print('pixels2',pixels2)
    for i in range(pixels1.shape[0]):
        # print(tuple(pixels1[i]),tuple(pixels2[i]))
        # had crash here if numbers overflow or wrong type which think because was say int32 instead of int16
        if (pixels1[i] > -10000).all() and (pixels1 < 10000).all() and (pixels2[i] > -10000).all() and (pixels2 < 10000).all():
            cv2.line(img, tuple(pixels1[i]), tuple(pixels2[i]), line_colors[i], thickness)

    return img
    

def plot_polygons(img,pix1_3d,pix2_3d,pix1_2d,pix2_2d,indices_lines_2d_to_3d,indices_which_way_round,mask_plot_polygons):


    assert indices_which_way_round.shape[0] == pix1_2d.shape[0]
    for i in range(pix1_2d.shape[0]):
        if mask_plot_polygons[i] == False:
            continue

        p1 = pix1_3d[indices_lines_2d_to_3d[i]]
        p2 = pix2_3d[indices_lines_2d_to_3d[i]]

        if indices_which_way_round[i] == 0:
            p3 = pix1_2d[i]
            p4 = pix2_2d[i]
        elif indices_which_way_round[i] == 1:
            p3 = pix2_2d[i]
            p4 = pix1_2d[i]

        pts = torch.stack([p1,p2,p3,p4])
        pts = np.round(pts.numpy()).astype(np.int32)
        # if (pts > 0).all():
        cv2.fillPoly(img,[pts],(0,255,255,0.5))

    return img

def plot_polygons_v2(img,pix1_3d,pix2_3d,pix1_2d,pix2_2d,indices_lines_2d_to_3d,indices_which_way_round,mask_plot_polygons):

    all_indices = [[0,1,2,3],[0,1,3,2],[0,2,1,3],[0,2,3,1],[0,3,1,2],[0,3,2,1]]

    assert indices_which_way_round.shape[0] == pix1_2d.shape[0]
    for i in range(pix1_2d.shape[0]):
        if mask_plot_polygons[i] == False:
            continue

        
        p1 = pix1_3d[indices_lines_2d_to_3d[i]]
        p2 = pix2_3d[indices_lines_2d_to_3d[i]]
        p3 = pix1_2d[i]
        p4 = pix2_2d[i]
        points = [p1,p2,p3,p4]

        selected_indices = all_indices[indices_which_way_round[i,indices_lines_2d_to_3d[i]]]


        pts = np.stack([points[selected_indices[0]],points[selected_indices[1]],points[selected_indices[2]],points[selected_indices[3]]])
        # print(np.stack([points[selected_indices[0]],points[selected_indices[1]],points[selected_indices[2]],points[selected_indices[3]]]))
        pts = np.round(pts).astype(np.int32)
        # if (pts > 0).all():
        cv2.fillPoly(img,[pts],(0,255,255,0.5))

    return img


def put_text(img,text,relative_position):
    h,w,_ = img.shape
    font_size = min(h,w)/25 * 0.05
    thickness = int(font_size * 4)

    y = int(relative_position[0] * h)
    x = int(relative_position[1] * w)
    # x = int(min(h,w) /8)
    # y = 2*x
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_size,(255, 0, 0), thickness, cv2.LINE_AA)
    return img



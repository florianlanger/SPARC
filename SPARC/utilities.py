import os
import shutil
import json
import cv2
import numpy as np
import torch

def create_directories(exp_path):

    os.mkdir(exp_path)
    os.mkdir(exp_path + '/code')
    os.mkdir(exp_path + '/log_files')
    os.mkdir(exp_path + '/saved_models')
    os.mkdir(exp_path + '/vis')
    os.mkdir(exp_path + '/vis_roca_eval')
    os.mkdir(exp_path + '/vis_3d')
    os.mkdir(exp_path + '/vis_3d_roca_eval')
    os.mkdir(exp_path + '/predictions')

    

def dict_replace_value(d, old, new):
    x = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_replace_value(v, old, new)
        elif isinstance(v, str):
            # print(v)
            v = v.replace(old, new)
            # print(v)
        x[k] = v
    return x

def load_json(path):
    with open(path,'r') as f:
        loaded = json.load(f)
    return loaded

def draw_text_block(img,text,top_left_corner=(20,20),font_scale=3,font_thickness=2):

    line_height = 20 * font_scale

    for i,line in enumerate(text):
        pos = (top_left_corner[0],top_left_corner[1] + (i + 1) *line_height)
        draw_text(img,line,pos=pos,font_scale=font_scale,font_thickness=font_thickness)


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def writePlyFile(file_name, vertices, colors):

    ply_header = '''ply
                format ascii 1.0
                element vertex %(vert_num)d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
               '''
    vertices = vertices.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices, colors])
    with open(file_name, 'w') as f:
      f.write(ply_header % dict(vert_num=len(vertices)))
      np.savetxt(f, vertices, '%f %f %f %d %d %d')

def convert_K(K,width,height):
    K[0,2] = K[0,2] - width/2
    K[1,2] = K[1,2] - height/2
    K = K/(width/2)
    K[2:4,2:4] = torch.Tensor([[0,1],[1,0]])
    return K

def make_dir_save(out_dir,assert_not_exist=True):
    if os.path.exists(out_dir):
        if assert_not_exist == True:
            assert os.listdir(out_dir) == []
    else:
        os.mkdir(out_dir)


def get_model_to_infos_scannet(path_full_annos):
    with open(path_full_annos,'r') as f:
        annos = json.load(f)

    top = {}
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"

    infos = {}

    for scene in annos:
        for model in scene['aligned_models']:
            if model['catid_cad'] in top:
                name = top[model['catid_cad']] + '_' + model['id_cad']
                infos_model = {}
                infos_model["bbox"] = model["bbox"]
                infos_model["center"] = model["center"]
                if name in model:
                    assert infos_model == infos[name], (infos_model,infos[name])
                else:
                    infos[name] = infos_model
    return infos

def get_model_to_infos_scannet_just_id(path):
    with open(path,'r') as f:
        annos = json.load(f)

    top = {}
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    top["02818832"] = "bed"

    infos = {}

    for scene in annos:
        for model in scene['aligned_models']:
            if model['catid_cad'] in top:
                name = model['id_cad']
                infos_model = {}
                infos_model["bbox"] = model["bbox"]
                infos_model["center"] = model["center"]
                infos_model["category"] = top[model['catid_cad']]
                if name in model:
                    assert infos_model == infos[name], (infos_model,infos[name])
                else:
                    infos[name] = infos_model
    return infos

from cmath import exp
from SPARC.visualisation.visualisation_points_and_normals import plot_points_preds_normals
from SPARC.utilities import writePlyFile
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np

def visualise_preds(writer,latest_metrics,inputs,net,config,kind,epoch,extra_infos,exp_path,num_batch,iter_refinement,roca_eval_combo=None):
    probabilities = latest_metrics["probabilities"].cpu().detach().numpy()
    labels = latest_metrics["labels"].cpu().detach().numpy()
    correct = latest_metrics["correct"].cpu().detach().numpy()
    t_pred = latest_metrics["t_pred"].cpu().detach().numpy()
    s_pred = latest_metrics["s_pred"].cpu().detach().numpy()
    r_pred = latest_metrics["r_pred"].cpu().detach().numpy()

    t_correct = latest_metrics["t_correct"].cpu().detach().numpy()
    s_correct = latest_metrics["s_correct"].cpu().detach().numpy()
    r_correct = latest_metrics["s_correct"].cpu().detach().numpy()

    
    no_render = not config["general"]["run_on_octopus"]
    images,info_vis_3d = plot_points_preds_normals(inputs.cpu(), labels,probabilities,correct,t_pred,s_pred,r_pred,t_correct,s_correct,r_correct,config,extra_infos,kind,no_render=no_render)
    save_images(images,epoch,exp_path,kind,extra_infos,probabilities,correct,labels,num_batch,iter_refinement,roca_eval_combo=roca_eval_combo)
    save_vis_3d(info_vis_3d,epoch,exp_path,kind,extra_infos,num_batch,iter_refinement,roca_eval_combo)



def save_images(images,epoch,exp_path,kind,extra_infos,probabilities,correct,labels,num_batch,iter_refinement,roca_eval_combo=None):

    names = ['reprojected_points','normal_full','normal_selected','depth_full','depth_selected','depth_reprojected','normal_reprojected_1','normal_reprojected_2','normal_reprojected_3','diff_depth','diff_normals','gt_reprojection','render','render_prediction','visibility_1','visibility_2','visibility_3','rgb']

    for i in range(len(images)):
        if roca_eval_combo == None:
            base_path = exp_path + '/vis/{}_epoch_{}_num_batch_{}_example_{}_refinement_{}_{}'.format(kind,str(epoch).zfill(6),str(num_batch).zfill(6),str(i).zfill(3),str(iter_refinement).zfill(2),extra_infos["detection_name"][i].split('.')[0])
        else:
            base_path = exp_path + '/vis_roca_eval/{}_epoch_{}_num_batch_{}_example_{}_refinement_{}_{}'.format(roca_eval_combo,str(epoch).zfill(6),str(num_batch).zfill(6),str(i).zfill(3),str(iter_refinement).zfill(2),extra_infos["detection_name"][i].split('.')[0])
        for j in range(len(images[i])):
            plt.imsave(base_path + '_{}_pred_{}_label_{}_{}.png'.format(names[j],np.round(probabilities[i,0].item(),3),np.round(labels[i,0].item(),3),str(correct[i].item())),images[i][j])
        combined = combine_images(images[i])
        combined = cv2.cvtColor(combined,cv2.COLOR_BGR2RGB)

        cv2.imwrite(base_path + '_combined_pred_{}_label_{}_{}.png'.format(np.round(probabilities[i,0].item(),3),np.round(labels[i,0].item(),3),str(correct[i].item())),combined)

def save_vis_3d(info_vis_3d,epoch,exp_path,kind,extra_infos,num_batch,iter_refinement,roca_eval_combo=None):

    for i in range(len(info_vis_3d)):
        if roca_eval_combo == None:
            base_path = exp_path + '/vis_3d/{}_epoch_{}_num_batch_{}_example_{}_refinement_{}_{}'.format(kind,str(epoch).zfill(6),str(num_batch).zfill(6),str(i).zfill(3),str(iter_refinement).zfill(2),extra_infos["detection_name"][i].split('.')[0])
        else:
            base_path = exp_path + '/vis_3d_roca_eval/{}_epoch_{}_num_batch_{}_example_{}_refinement_{}_{}'.format(roca_eval_combo,str(epoch).zfill(6),str(num_batch).zfill(6),str(i).zfill(3),str(iter_refinement).zfill(2),extra_infos["detection_name"][i].split('.')[0])
    
        out_path = base_path + '_vis_3d.ply'
        writePlyFile(out_path,info_vis_3d[i][0],info_vis_3d[i][1])

def combine_images(images):
    # for i in range(3,6):
    #     images[i] = cv2.cvtColor(images[i],cv2.COLOR_GRAY2RGB)

    combined_top = cv2.hconcat([images[0],images[1],images[2],images[3]])
    combined_middle = cv2.hconcat([images[4],images[5],images[6],images[7]])
    combined_bottom = cv2.hconcat([images[8],images[9],images[10],images[11]])

    combined = cv2.vconcat([combined_top,combined_middle,combined_bottom])

    combined_models = cv2.vconcat([images[14],images[15],images[16]])
    combined_models = cv2.resize(combined_models,(360,1080))

    combined_renders = cv2.vconcat([images[12],images[13],images[17]])

    combined = cv2.hconcat([combined,combined_renders,combined_models])
    return combined


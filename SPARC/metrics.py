import torch
import numpy as np


def track_cat_accuracies(all_metrics,writer,epoch,kind,N_refinements,config,name='correct'):


    all_predictions,all_labels,all_categories,all_roca_bbox = all_metrics['all_predictions'],all_metrics['all_labels'],all_metrics['all_categories'],all_metrics['all_roca_bbox']

    assert len(all_predictions) == len(all_labels)
    assert len(all_predictions) == len(all_categories)

    unique_cats = set(all_categories)
    all_predictions = (np.array(all_predictions) > 0.5).tolist()
    all_roca_bbox = np.array(all_roca_bbox)

    preds_by_cat = {}
    labels_by_cat = {}
    for cat in unique_cats:
        preds_by_cat[cat] = []
        labels_by_cat[cat] = []
    
    for i in range(len(all_predictions)):
        preds_by_cat[all_categories[i]].append(all_predictions[i])
        labels_by_cat[all_categories[i]].append(all_labels[i])


    for cat in unique_cats:
        assert len(preds_by_cat[cat]) == len(labels_by_cat[cat])
        accuracy = np.sum(np.array(preds_by_cat[cat]) == np.array(labels_by_cat[cat])) / len(preds_by_cat[cat])
        writer.add_scalar(name + ' ' + cat + ' ' + kind,accuracy,epoch)


    all_correct = np.array(all_predictions) == np.array(all_labels)
    accuracy = np.sum(all_correct) / len(all_labels)
    writer.add_scalar(name + ' ' + 'all' + ' ' + kind,accuracy,epoch)

    if not np.all(all_roca_bbox == False):
        accuracy = np.mean(all_correct[all_roca_bbox]*1)
        writer.add_scalar(name + ' ' + 'all' + ' ' + kind + ' roca bbox',accuracy,epoch)
    if not np.all(all_roca_bbox == True):
        accuracy = np.mean(all_correct[~all_roca_bbox]*1)
        writer.add_scalar(name + ' ' + 'all' + ' ' + kind + ' no roca bbox',accuracy,epoch)

    writer.add_scalar('Number total ' + kind + 'roca bbox',np.sum(all_roca_bbox),epoch)
    writer.add_scalar('Number total ' + kind + ' no roca bbox',np.sum(~all_roca_bbox),epoch)

    for what_metric in ['weighted_classification_loss','weighted_t_loss','weighted_s_loss','weighted_r_loss']:
        writer.add_scalar(kind + ' ' + 'all' + ' ' + what_metric,np.mean(all_metrics[what_metric]),epoch)
     

    for iter_refinement in range(N_refinements):
        mask = iter_refinement == np.array(all_metrics['iter_refinement'])
        iter_refinement = str(iter_refinement)
        writer.add_scalar('Average t error ' + kind + 'refine step ' + iter_refinement,np.mean(np.array(all_metrics['t_distance'])[mask]),epoch)
        writer.add_scalar('Average s error ' + kind + 'refine step ' + iter_refinement,np.mean(np.array(all_metrics['s_distance'])[mask]),epoch)
        writer.add_scalar('t correct ' + kind + 'refine step ' + iter_refinement,np.mean(np.array(all_metrics['t_correct'] * 1)[mask]),epoch)
        writer.add_scalar('s correct ' + kind + 'refine step ' + iter_refinement,np.mean(np.array(all_metrics['s_correct']* 1)[mask]),epoch)
        writer.add_scalar('Average len t target ' + kind + 'refine step ' + iter_refinement,np.mean(np.linalg.norm(np.array(all_metrics['t_offset'])[mask,:],axis=1)),epoch)
        writer.add_scalar('Average len s target ' + kind + 'refine step ' + iter_refinement,np.mean(np.linalg.norm(np.array(all_metrics['s_offset'])[mask,:],axis=1)),epoch)
        
        both_correct = np.array(all_metrics['s_correct'])[mask] & np.array(all_metrics['t_correct'])[mask]
        writer.add_scalar('t and s correct ' + kind + 'refine step ' + iter_refinement,np.mean(both_correct * 1),epoch)
   

def get_distances_per_point(outputs,offsets,extra_infos,config):
    img_size_reshape = torch.Tensor(config["data"]['img_size']).unsqueeze(0).unsqueeze(0).repeat(outputs.shape[0],outputs.shape[1],1)
    # print('img size reshap',img_size_reshape[0,0])
    # print('(outputs - offsets).cpu() * img_size_reshape)**2',((outputs - offsets).cpu() * img_size_reshape**2)[0,0,:])
    dists = (torch.sum(((outputs - offsets).cpu() * img_size_reshape)**2,dim=2)**0.5).detach().numpy()
    # print('dists',dists)
    average_dists = []
    for i in range(dists.shape[0]):
        average_dists.append(np.mean(dists[i,:extra_infos['n_reprojected'][i]]))
    # print('extra_infos["n_reprojected"]',extra_infos['n_reprojected'])
    # print('average_dist',average_dist)
    # print('outputs',outputs[0,:extra_infos['n_reprojected']])
    # print('offsets',offsets[0,:extra_infos['n_reprojected']])
    return average_dists
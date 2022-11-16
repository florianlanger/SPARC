
import torch
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
import time
import itertools

# add to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from SPARC.utilities import load_json
from SPARC.utilities import create_directories
from SPARC.dataset.batch_sampler import get_batch_from_dataset,get_index_infos
from SPARC.evaluate.evaluate_roca import eval_predictions


from SPARC.visualisation.visualisation_main import visualise_preds
from SPARC.visualisation.confusion_matrix import visualise_confusion_matrices
from SPARC.metrics import track_cat_accuracies
from SPARC.losses.loss import get_combined_loss
from SPARC.main_utilities import save_predictions,set_network_state,create_shuffled_indices,get_N_refinements_and_sample_for_classification_v3,update_running_metrics,process_config,set_device,load_network,get_optimiser,set_network_state,save_checkpoint,create_data_loaders



def one_epoch(dataset,net,optimizer,criterion,epoch,writer,kind,device,config,exp_path):

    all_metrics = {'all_predictions':[],'all_labels':[],'all_categories':[],'all_roca_bbox':[],'all_losses':[],'all_distances':[],'all_extra_infos':[],'t_distance': [], 's_distance': [],'r_distance':[],'t_correct': [],
    's_correct': [],'running_loss':0.0,'n_correct_total':0,'counter_examples':0,'iter_refinement':[],'t_offset':[],'s_offset':[],'r_offset':[],'detection_names':[],'s_pred':[],'t_pred':[],'r_pred':[],'weighted_classification_loss':[],'weighted_t_loss':[],'weighted_s_loss':[],'weighted_r_loss':[]}
    
    set_network_state(net,kind)

    
    roca_eval_combo = None
    if kind == 'val_roca':
        roca_eval_combo = 'translation_{}_scale_{}_rotation_{}_retrieval_{}'.format(dataset.eval_params['what_translation'],dataset.eval_params['what_scale'],dataset.eval_params['what_rotation'],dataset.eval_params['what_retrieval'])

    all_indices = create_shuffled_indices(len(dataset),config,kind)

    N_total_batches = int(np.ceil(len(all_indices) / config['training']["batch_size"]))
    # N_refinements = 1 if epoch < config["training"]["use_refinement_after_which_epoch"] else config["training"]["refinement_per_object"]

    for num_batch in tqdm(range(N_total_batches)):
    # for num_batch in range(N_total_batches):
        index_infos = all_indices[num_batch*config['training']["batch_size"]:(num_batch+1)*config['training']["batch_size"]]

        N_refinements,just_classifier = get_N_refinements_and_sample_for_classification_v3(kind,config)

        for iter_refinement in range(N_refinements):
            data = get_batch_from_dataset(dataset,index_infos,just_classifier)
            inputs, targets, extra_infos = data


            optimizer.zero_grad()


            outputs = net(inputs)


            loss,latest_metrics = get_combined_loss(outputs,targets,config,just_classifier)

            mean_loss = torch.mean(loss)
            if kind == 'train':
                mean_loss.backward()
                optimizer.step()
            t4 = time.time()
            all_metrics = update_running_metrics(all_metrics,latest_metrics,extra_infos,loss,iter_refinement)

            t5 = time.time()
            if num_batch == 0 and epoch % config['training']["vis_interval"] == 0:
                visualise_preds(writer,latest_metrics,inputs,net,config,kind,epoch,extra_infos,exp_path,num_batch,iter_refinement,roca_eval_combo)


            index_infos = get_index_infos(outputs,extra_infos,config,iter_refinement)
            t6 = time.time()
            # print('t2 - t1: {}'.format(t2 - t1))
            # print('t3 - t2: {}'.format(t3 - t2))
            # print('t4 - t3: {}'.format(t4 - t3))
            # print('t5 - t4: {}'.format(t5 - t4))
            # print('t6 - t5: {}'.format(t6 - t5))

    if epoch % config['training']["vis_interval"] == 0:
        visualise_confusion_matrices(all_metrics['all_predictions'],all_metrics['all_labels'],all_metrics['all_categories'],writer,epoch,kind)

    if kind == 'val_roca':
        eval_path = save_predictions(all_metrics,N_refinements,epoch,exp_path,dataset.eval_params,dataset.use_all_images)
        if config["general"]["run_on_octopus" ] == True:
            n_scenes_vis = config["training"]["n_vis_scenes"]
        else:
            n_scenes_vis = 0

        eval_predictions(eval_path,config,n_scenes_vis=n_scenes_vis,eval_all_images=dataset.use_all_images)
    track_cat_accuracies(all_metrics,writer,epoch,kind,N_refinements,config)
    writer.add_scalar(kind + ' loss',all_metrics['running_loss'] / all_metrics['counter_examples'],epoch)

    return all_metrics['running_loss'] / all_metrics['counter_examples'], all_metrics['n_correct_total']/ all_metrics['counter_examples'], all_metrics['all_extra_infos'],all_metrics['all_losses']




def start_new():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = load_json('{}/config.json'.format(dir_path))
    config = process_config(config)
    exp_path = '{}/{}_{}'.format(config["general"]["output_dir"],datetime.now().strftime("date_%Y_%m_%d_time_%H_%M_%S"),config["general"]["name"])
    print(exp_path)
    create_directories(exp_path)
    device = set_device(config)

    network = load_network(config,device)
    optimizer = get_optimiser(config,network)
    
    start_epoch = 0

    return network,optimizer,exp_path,config,device,start_epoch



def resume_checkpoint(checkpoint_path):
    exp_path = checkpoint_path.rsplit('/',2)[0]
    config = load_json('{}/config.json'.format(exp_path))
    config = process_config(config)
    device = set_device(config)

    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    optimiser_saved = torch.load(checkpoint_path.replace('network_epoch','optimizer_epoch'),map_location=torch.device('cpu'))
    start_epoch = int(checkpoint_path.split('/')[-1].split('.')[0].split('_')[-1])

    network = load_network(config,device)
    optimizer = get_optimiser(config,network)


    network.load_state_dict(checkpoint)

    optimizer.load_state_dict(optimiser_saved)
    # start_epoch = checkpoint['epoch']
    return network,optimizer,exp_path,config,device,start_epoch




def validate_roca(val_roca_dataset,network,optimizer,criterion,epoch,writer,device,config,exp_path):

    assert val_roca_dataset.kind == 'val_roca', val_roca_dataset.kind
    # print('in eval have kind ',val_roca_dataset.kind)

    if epoch % config["training"]["roca_eval_interval"] == 0: # and epoch != 0:
        if val_roca_dataset.use_all_images == True:
            a = [['pred'],['pred'],['roca','roca_init'],['roca']]
        else:
            a = [['pred'],['pred'],['no_init'],['roca']]
       
        all_combinations = list(itertools.product(*a))

        for combo in all_combinations:

            if combo[0] == 'pred':
                val_roca_dataset.config["data"]["sample_T"]["ignore_prediction"] = False
            else:
                val_roca_dataset.config["data"]["sample_T"]["ignore_prediction"] = True

            if combo[1] == 'pred':
                val_roca_dataset.config["data"]["sample_S"]["ignore_prediction"] = False
            else:
                val_roca_dataset.config["data"]["sample_S"]["ignore_prediction"] = True

            if combo[2] == 'roca_init' or combo[2] == 'no_init' or combo[2] == 'init_from_best_rotation_index':
                val_roca_dataset.config["data"]["sample_R"]["ignore_prediction"] = False
            else:
                val_roca_dataset.config["data"]["sample_R"]["ignore_prediction"] = True


            eval_roca_dict = {'what_translation':combo[0],'what_scale':combo[1],'what_rotation':combo[2],'what_retrieval':combo[3]}
            print('eval dict',eval_roca_dict)
            val_roca_dataset.eval_params = eval_roca_dict
            val_loss,val_accuracy,_,_ = one_epoch(val_roca_dataset,network,optimizer,criterion,epoch,writer,"val_roca",device,config,exp_path)


def main():
    torch.manual_seed(1)
    np.random.seed(0)

    if len(sys.argv) == 1:
        network,optimizer,exp_path,config,device,start_epoch = start_new()
    elif len(sys.argv) == 2:
        network,optimizer,exp_path,config,device,start_epoch = resume_checkpoint(sys.argv[1])


    writer = SummaryWriter(exp_path + '/log_files',max_queue=10000, flush_secs=600)
    train_dataset,val_dataset,val_roca_dataset,val_roca_dataset_all_images = create_data_loaders(config)
    criterion = None

    # print('debug')
    # with torch.no_grad():
    #     validate_roca(val_roca_dataset_all_images,network,optimizer,criterion,0,writer,device,config,exp_path)

    for epoch in tqdm(range(start_epoch,config["training"]["n_epochs"])):

        train_loss,train_accuracy,all_extra_infos,all_losses = one_epoch(train_dataset,network,optimizer,criterion,epoch,writer,'train',device,config,exp_path)
        # train_loss,train_accuracy,all_extra_infos,all_losses = one_epoch(val_dataset,network,optimizer,criterion,epoch,writer,'train',device,config,exp_path)
        metric_dict = {"train_loss_last_epoch": train_loss,"train_accuracy_last_epoch":train_accuracy}
        if config["training"]["validate"]  == True:
            with torch.no_grad():
                # val_loss,val_accuracy,_,_ = one_epoch(val_dataset,network,optimizer,criterion,epoch,writer,"val",device,config,exp_path)
                
                if config["training"]["validate_roca"] == True:
                    validate_roca(val_roca_dataset_all_images,network,optimizer,criterion,epoch,writer,device,config,exp_path)

        if not 'small' in config['data']["dir_path_2d_train"]:
            save_checkpoint(exp_path + '/saved_models',epoch,network,optimizer,config)

    # log_hparams(writer,config,metric_dict)

    hparam_dict = {'bs': config["training"]["batch_size"],'lr': config["training"]["learning_rate"],'optimiser':config["training"]['optimiser'],'n_epochs':epoch}
    writer.add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)
    # writer.close()

if __name__ == "__main__":
    main()
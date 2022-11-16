import torch
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
import itertools
import os

# add to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from SPARC.utilities import load_json
from SPARC.utilities import create_directories


from SPARC.utilities import dict_replace_value
from SPARC.main_utilities import process_config,set_device,load_network,get_optimiser,create_data_loaders
from SPARC.main import one_epoch



def evaluate_checkpoint(checkpoint_path,n_refinements,index_rotation,output_dir,run_on_octopus=True,gpu_number=0):
    
    name_exp_eval = datetime.now().strftime("date_%Y_%m_%d_time_%H_%M_%S") 
    config_path = checkpoint_path.rsplit('/',1)[0] + '/config.json'
    config = load_json(config_path)
    config = process_config(config)


    config["general"]["name"] = name_exp_eval
    config["general"]["run_on_octopus"] = run_on_octopus
    config["general"]["gpu"] = str(gpu_number)
    config["training"]["only_eval_roca"] = True
    config["training"]["vis_interval"] = config["training"]["save_interval"]


    config["evaluate"]["rotation_index"] = index_rotation


    config["training"]["refinement_per_object"] = int(n_refinements)
    config["training"]["n_epochs"] = 0
    config["training"]["n_vis"] = 3
    config["training"]["n_scenes_vis"] = 3

    exp_path = '{}/{}_EVAL_REFINE_{}_{}_rotation_index_{}'.format(output_dir,datetime.now().strftime("date_%Y_%m_%d_time_%H_%M_%S"),n_refinements,name_exp_eval,index_rotation)
    print(exp_path)
    # print('config:',config["data"])
    create_directories(exp_path)
    device = set_device(config)

    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    network = load_network(config,device)
    network.load_state_dict(checkpoint)
    optimizer = get_optimiser(config,network)
    
    start_epoch = 0

    return network,optimizer,exp_path,config,device,start_epoch


def validate_roca(val_roca_dataset,network,optimizer,criterion,epoch,writer,device,config,exp_path,eval_method):

    assert val_roca_dataset.kind == 'val_roca', val_roca_dataset.kind
    # print('in eval have kind ',val_roca_dataset.kind)

    if epoch % config["training"]["roca_eval_interval"] == 0 and epoch != 0:
        # translation, scale , rotation, retrieval
        # a = [['gt','roca','median','pred'],['lines','roca','gt'],['roca','gt']]
        
        if val_roca_dataset.use_all_images == True:
            a = [['pred'],['pred'],[eval_method],['roca']]

        else:
            a = [['gt','roca','pred'],['gt','roca','median'],['gt','roca'],['roca']]
            a = [['gt','roca','pred'],['pred','gt','roca','median'],['roca_init','gt','roca'],['roca']]
            a = [['pred'],['pred','roca','median'],['no_init','roca_init','roca'],['roca']]
            # a = [['pred'],['roca'],['roca'],['roca']]
       
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


    checkpoint_path,n_refinements,index_rotation,eval_method,output_dir = sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4],sys.argv[5]


    network,optimizer,exp_path,config,device,start_epoch = evaluate_checkpoint(checkpoint_path,n_refinements,index_rotation,output_dir,run_on_octopus=True,gpu_number=0)

    writer = SummaryWriter(exp_path + '/log_files',max_queue=10000, flush_secs=600)
    train_dataset,val_dataset,val_roca_dataset,val_roca_dataset_all_images = create_data_loaders(config)
    criterion = None

    if eval_method == 'init_from_best_rotation_index':
        val_roca_dataset_all_images.best_rotation_indices = load_json(output_dir + '/best_rotation_index.json')


    if len(sys.argv) > 2:
        with torch.no_grad():
            validate_roca(val_roca_dataset_all_images,network,optimizer,criterion,config["training"]["roca_eval_interval"],writer,device,config,exp_path,eval_method)


if __name__ == "__main__":
    main()
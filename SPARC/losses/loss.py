
from asyncio import constants
import torch

def get_combined_loss(outputs,targets,config,just_classifier=False):

    constants_multiplier = config['loss']['constants_multiplier']

    probabilities = torch.sigmoid(outputs[:,0:1])
    labels = targets[:,0:1]
    classification_loss = torch.nn.BCELoss(reduction='none')(probabilities, labels)

    binary_prediction = (probabilities > 0.5)
    binary_labels = (labels > 0.5)
    correct = binary_prediction == binary_labels

    metrics = {'correct': correct, 'probabilities': probabilities, 'labels': labels}

    if outputs.shape[1] == 1:
        return classification_loss, metrics
    
    else:
        t_offset = torch.sum((targets[:,1:4] - outputs[:,1:4])**2,dim=1).unsqueeze(1)
        s_offset = torch.sum((targets[:,4:7] - outputs[:,4:7])**2,dim=1).unsqueeze(1)
        # r_offset = torch.sum((targets[:,7:16] - outputs[:,7:16])**2,dim=1).unsqueeze(1)
        r_offset = torch.sum((targets[:,7:11] - outputs[:,7:11])**2,dim=1).unsqueeze(1)

        weighted_classification_loss = classification_loss * constants_multiplier["classification"]
        weighted_t_loss = t_offset * constants_multiplier['t']
        weighted_s_loss = s_offset * constants_multiplier['s']
        weighted_r_loss = r_offset * constants_multiplier['r']
        if just_classifier == True:
            loss = weighted_classification_loss

            # metrics['t_distance'] = t_offset[:0,:] ** 0.5
            # metrics['s_distance'] = s_offset[:0,:] ** 0.5
            # metrics['r_distance'] = r_offset[:0,:] ** 0.5
            # metrics['t_correct'] = (t_offset[:0,:] ** 0.5 < 0.2).squeeze(1)
            # metrics['s_correct'] = torch.all(torch.abs((outputs[:0,4:7] + 1) / (targets[:0,4:7] + 1) - 1) < 0.2,dim=1)
            # metrics['t_pred'] = outputs[:0,1:4]
            # metrics['s_pred'] = outputs[:0,4:7]
            # metrics['r_pred'] = outputs[:0,7:11]
            # metrics['weighted_classification_loss'] = weighted_classification_loss
            # metrics['weighted_t_loss'] = weighted_t_loss[:0,:]
            # metrics['weighted_s_loss'] = weighted_s_loss[:0,:]
            # metrics['weighted_r_loss'] = weighted_r_loss[:0,:]

        else:
            # loss = weighted_classification_loss + weighted_t_loss + weighted_s_loss + weighted_r_loss
            loss = weighted_t_loss + weighted_s_loss + weighted_r_loss

        metrics['t_distance'] = t_offset ** 0.5
        metrics['s_distance'] = s_offset ** 0.5
        metrics['r_distance'] = r_offset ** 0.5
        metrics['t_correct'] = (t_offset ** 0.5 < 0.2).squeeze(1)
        metrics['s_correct'] = torch.all(torch.abs((outputs[:,4:7] + 1) / (targets[:,4:7] + 1) - 1) < 0.2,dim=1)
        metrics['t_pred'] = outputs[:,1:4]
        metrics['s_pred'] = outputs[:,4:7]
        metrics['r_pred'] = outputs[:,7:11]
        metrics['weighted_classification_loss'] = weighted_classification_loss
        metrics['weighted_t_loss'] = weighted_t_loss
        metrics['weighted_s_loss'] = weighted_s_loss
        metrics['weighted_r_loss'] = weighted_r_loss


        return loss,metrics

def get_criterion(config):
    if config['data']['targets'] == 'labels':
        criterion = torch.nn.BCELoss(reduction='none')
    elif config['data']['targets'] == 'offsets':
        criterion = torch.nn.MSELoss(reduction='none')
    return criterion
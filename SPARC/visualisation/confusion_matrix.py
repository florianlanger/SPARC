from matplotlib import pyplot as plt
import numpy as np
import itertools
from sklearn import metrics



def plot_confusion_matrix(cm, class_names,category,n_items): 
    figure = plt.figure(figsize=(8, 8)) 
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.inferno) 
    plt.title(category + ' ' + str(n_items),font={'size':32}) 
    plt.colorbar() 
    tick_marks = np.arange(len(class_names)) 
    plt.xticks(tick_marks, class_names) 
    plt.yticks(tick_marks, class_names)
  
    threshold = cm.max() / 2. 

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):   
        color = "black" if cm[i, j] > threshold else "white"   
        plt.text(j, i, cm[i, j], horizontalalignment="center",font={'size':32},color=color)  
    
    plt.tight_layout() 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label') 

    return figure



def visualise_confusion_matrices(all_predictions,all_labels,all_categories,writer,global_step,kind):
    assert len(all_predictions) == len(all_labels)
    assert len(all_predictions) == len(all_categories)

    unique_cats = set(all_categories)
    all_predictions = (np.array(all_predictions) > 0.5).tolist()

    preds_by_cat = {}
    labels_by_cat = {}
    for cat in unique_cats:
        preds_by_cat[cat] = []
        labels_by_cat[cat] = []
    
    for i in range(len(all_predictions)):
        preds_by_cat[all_categories[i]].append(all_predictions[i])
        labels_by_cat[all_categories[i]].append(all_labels[i])


    for cat in unique_cats:
        cm = metrics.confusion_matrix(labels_by_cat[cat], preds_by_cat[cat])
        fig = plot_confusion_matrix(cm, ['0','1'],cat,len(labels_by_cat[cat]))
        writer.add_figure('confusion matrix ' + cat + ' ' + kind,fig,global_step)

    cm = metrics.confusion_matrix(all_labels, all_predictions)
    fig = plot_confusion_matrix(cm, ['0','1'],'all',len(all_predictions))
    writer.add_figure('confusion matrix ' + 'all' + ' ' + kind,fig,global_step)
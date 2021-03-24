import os

import sktime
#Watch out, this is from old Version. New is sktime.utils.data_io
#from sktime.utils.load_data import load_from_tsfile_to_dataframe 
#from sktime.utils.data_container import from_nested_to_2d_array

#If sktime >= 0.5.3.
from sktime.utils.data_io import load_from_tsfile_to_dataframe, load_from_arff_to_dataframe
from sktime.utils.data_processing import from_nested_to_2d_array 

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt

from .constants import UCR_SELECTION
from .constants import CLASSIFIERS
from .constants import ITERATIONS

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            #if another machine created path meanwhile
            return None
        return directory_path


def test():
    return 0

def read_dataset(root_dir, dataset_name):
    datasets_dict = {}
    curr_root_dir = root_dir.replace('-temp', '')

    #For UCR
    root_dir_dataset = curr_root_dir + '/' + 'UCRArchive_2018'

    x_train, y_train = load_from_tsfile_to_dataframe(root_dir_dataset + '/'+ dataset_name + '/' + dataset_name + '_TRAIN.ts')
    x_test, y_test = load_from_tsfile_to_dataframe(root_dir_dataset + '/'+ dataset_name + '/' + dataset_name + '_TEST.ts')

    #x_train, y_train = load_from_arff_to_dataframe(root_dir_dataset + '/'+ dataset_name + '/' + dataset_name + '_TRAIN.arff')
    #x_test, y_test = load_from_arff_to_dataframe(root_dir_dataset + '/'+ dataset_name + '/' + dataset_name + '_TEST.arff')

    #print(x_train)

    x_train = from_nested_to_2d_array(x_train, return_numpy=True)
    x_test = from_nested_to_2d_array(x_test, return_numpy=True)

    # znorm
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    return datasets_dict

def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)

def save_logs(output_dir, hist, y_pred, y_true, duration, verbose, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_dir + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_dir + 'df_metrics.csv', index=False)

    if(verbose):
        print(df_metrics)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_dir + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_dir + 'epochs_loss.png', verbose)

    return df_metrics


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res

def plot_epochs_metric(hist, file_name, verbose, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')

    #if(verbose):
        #plt.show()

    plt.close()

def generate_results_csv(output_file_name, root_dir):
    root_dir += '/'
    res = pd.DataFrame(data=np.zeros((0, 7), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name',
                                'precision', 'accuracy', 'recall', 'duration'])
    for classifier_name in CLASSIFIERS: 
        archive_name = 'UCRArchive_2018'

        for it in range(ITERATIONS):
            if it != 0:
                archive_name_dir = archive_name + '/_itr_' + str(it)
            else:
                archive_name_dir = archive_name
                
            for dataset_name in UCR_SELECTION:
                if classifier_name == 'emn_cv':
                    output_dir = root_dir + '/results/' + classifier_name + '/' + archive_name + '/' + dataset_name \
                                    + '/' + 'df_metrics' + str(it) + '.csv' 
                else:
                    output_dir = root_dir + '/results/' + classifier_name + '/' \
                                    + archive_name_dir + '/' + dataset_name + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue
                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['archive_name'] = archive_name
                df_metrics['dataset_name'] = dataset_name
                res = pd.concat((res, df_metrics), axis=0, sort=False)
                
    
    res.to_csv(root_dir + output_file_name, index=False)
    
    #Get the mean accuracies
    res = pd.DataFrame({
        'accuracy': res.groupby(
        ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
    }).reset_index()
    
    return res

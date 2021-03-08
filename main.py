import os

import numpy as np
import sklearn

from utils.utils import read_dataset
from utils.utils import create_directory
from utils.constants import UCR_NO_VARY
from utils.constants import CLASSIFIERS
from utils.constants import ITERATIONS

def fit_classifier():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    #print('Number of Classes: %s' % nb_classes)
    
    #one-hot-encoding
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    
    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1) #See if this is really needed later
    
    if len(x_train.shape) == 2: #if univariate, check to see if this may make things harder later on
        #adds dimension making it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
    input_shape = x_train.shape[1:]
    #print(x_train.shape)
    tune = True
    verbose = True
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_dir, tune, verbose)
    
    classifier.fit(x_train, y_train, x_test, y_test, y_true)
    
def create_classifier(classifier_name, input_shape, nb_classes, output_dir, tune, verbose=False):
    if classifier_name == 'mlp':
        from models import mlp
        return mlp.Classifier_MLP(output_dir, input_shape, nb_classes, verbose)
    if classifier_name == 'lstmfcn':
        from models import lstmfcn
        return lstmfcn.Classifier_LSTMFCN(output_dir, nb_classes, tune, verbose)
    if classifier_name == 'emn':
        from models import emn
        return emn.Classifier_EMN(output_dir, nb_classes, verbose)

root_dir = os.getcwd()
print(root_dir)




for classifier_name in CLASSIFIERS:
    print('classifier_name', classifier_name)

    for iter in range(ITERATIONS):
        print('\t\titer', iter)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)

        tmp_output_dir = root_dir + '/results/' + classifier_name + '/UCRArchive_2018/' + trr + '/'

        for dataset_name in UCR_NO_VARY:
            print('\t\t\tdataset_name: ', dataset_name)

            output_dir = tmp_output_dir + dataset_name + '/'

            create_directory(output_dir)

            datasets_dict = read_dataset(root_dir, dataset_name)

            fit_classifier()

            print('\t\t\t\tDONE')

            # the creation of this directory means
            create_directory(output_dir + '/DONE')

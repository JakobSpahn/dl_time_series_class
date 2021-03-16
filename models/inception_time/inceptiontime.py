import numpy as np
import time
import gc

from .inception import Classifier_INCEPTION

from utils.utils import create_directory, calculate_metrics


class Classifier_InceptionTime:

    def __init__(self, output_dir, input_shape, nb_classes, verbose=True, num_ensemble_it=5):
        self.output_dir = output_dir
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.verbose = verbose
        self.num_ensemble_it = 5

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        ##################################
        ##Train n individual classifiers##
        ##################################
        for it in range(self.num_ensemble_it):
            if it == 0:
                itr_str = 'network'
                verbosity = self.verbose
            else:
                itr_str = 'network'+str(it)
                verbosity = False

            tmp_output_dir = self.output_dir + itr_str + '/'
            create_directory(tmp_output_dir)

            inception = Classifier_INCEPTION(
                tmp_output_dir, self.input_shape, self.nb_classes, verbose = verbosity)
            
            print('Fitting network {0} out of {1}'.format(
                it+1, self.num_ensemble_it))
            
            inception.fit(x_train, y_train, x_val, y_val, y_true)

        #######################################
        ##Ensemble the individual classifiers##
        #######################################
        start_time = time.time()

        y_pred = np.zeros(shape=y_val.shape)

        ll = 0

        for it in range(self.num_ensemble_it):
            if it == 0:
                itr_str = 'network'
            else:
                itr_str = 'network'+str(it)

            classifier_dir = self.output_dir + itr_str + '/'

            predictions_file_name = classifier_dir + 'y_pred.npy'

            curr_y_pred = np.load(predictions_file_name)

            y_pred = y_pred + curr_y_pred

            ll += 1

        # average predictions
        y_pred = y_pred / ll

        # save predictions
        np.save(self.output_dir + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        duration = time.time() - start_time

        df_metrics = calculate_metrics(y_true, y_pred, duration)
        print(df_metrics)

        df_metrics.to_csv(self.output_dir + 'df_metrics.csv', index=False)

        gc.collect()

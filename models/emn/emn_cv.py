import time
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from .base_emn import Base_Classifier_EMN
from utils.utils import save_logs, calculate_metrics, create_directory
import csv
import pandas as pd
import os


class Classifier_EMN_CV:
    def __init__(self, output_dir, nb_classes, verbose):
        self.output_dir = output_dir
        self.nb_classes = nb_classes
        self.verbose = verbose

        # Hyperparameters for first grid search
        self.input_scaling = [0.1, 1]
        self.connectivity = [0.3, 0.7]
        self.num_filter = [90, 120, 150]

        # Hyperparameters for secont grid search
        self.ratio = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

    def fit(self, x_train, y_train, x_test, y_test, y_true):
        start_time_tuning = time.time()


        #######################
        ##Grid Search Stage 1##
        #######################
        param_grid_1 = dict(input_scaling=self.input_scaling,
                            connectivity=self.connectivity,
                            num_filter=self.num_filter)
        
        emn_stage_1 = Base_Classifier_EMN(nb_classes=self.nb_classes, verbose=False)

        grid_1 = GridSearchCV(estimator=emn_stage_1,
                              param_grid=param_grid_1, cv=3, verbose=3)
        grid_1_result = grid_1.fit(x_train, y_train)

        #######################
        ##Grid Search Stage 2##
        #######################
        param_grid_2 = dict(ratio=self.ratio)

        emn_stage_2 = grid_1_result.best_estimator_

        grid_2 = GridSearchCV(estimator=emn_stage_2,
                              param_grid=param_grid_2, cv=3, verbose=3)
        grid_2_result = grid_2.fit(x_train, y_train)

        duration_tuning = time.time() - start_time_tuning

        # Print Tune Duration
        print('Tune duration: {0}s'.format(duration_tuning))
        f = open(self.output_dir + '/grid_search_duration.txt', 'w')
        f.write(str(duration_tuning))
        f.close()

        #####################################
        ##Final Training on whole train set##
        #####################################
        emn_final = grid_2_result.best_estimator_
        if self.verbose:
            emn_final.verbose = True
        
        # Save Params
        df_final_tuned = pd.DataFrame([[emn_final.input_scaling, emn_final.connectivity, emn_final.num_filter, emn_final.ratio]], columns=[
            'input_scaling', 'connectivity', 'num_filter', 'ratio'])
        print(df_final_tuned)
        df_final_tuned.to_csv(self.output_dir +
                              'df_final_params.csv', index=False)

        #Three iterations for the best parameters
        for i in range(3):

            start_time = time.time()

            emn_final.fit(x_train, y_train)
            y_pred = emn_final.predict(x_test)

            duration = time.time() - start_time

            # Save Metrics
            df_metrics = calculate_metrics(y_true, y_pred, duration)
            if self.verbose:
                print(df_metrics)
            df_metrics.to_csv(self.output_dir +
                              'df_metrics' + str(i) + '.csv', index=False)

     
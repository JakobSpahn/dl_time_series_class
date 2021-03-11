import time
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from .emn_base import Classifier_EMN
from utils.utils import save_logs


class Classifier_EMN_CV:
    def __init__(self, output_dir, verbose):
        self.output_dir = output_dir
        self.verbose = verbose

        #Hyperparameters for first grid search
        self.input_scaling = [0.1, 1]
        self.connectivity = [0.3,0.7]
        self.num_filter = [60, 90, 120]

        #Hyperparameters for secont grid search
        self.ratio = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

    def fit(self, x_train, y_train, x_test, y_test, y_true):
        #######################
        ##Grid Search Stage 1##
        #######################
        param_grid_1 = dict(input_scaling=self.input_scaling,
                            connectivity=self.connectivity,
                            num_filter=self.num_filter)

        emn_stage_1 = Classifier_EMN(verbose=False)

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


        #####################################
        ##Final Training on whole train set##
        #####################################
        emn_final = grid_2_result.best_estimator_
        if self.verbose :
            emn_final.verbose = True

        start_time = time.time()

        emn_final.fit(x_train, y_train)
        y_pred = emn_final.predict(x_test)

        duration = time.time() - start_time

        save_logs(self.output_dir, emn_final.hist_, y_pred, y_true,
                  duration, self.verbose, lr=False)



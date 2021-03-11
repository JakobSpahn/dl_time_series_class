import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import pandas as pd

from layers.reservoir import Reservoir

from utils.utils import save_logs, calculate_metrics, create_directory

import matplotlib

from tqdm.keras import TqdmCallback

from sklearn.model_selection import train_test_split


class Classifier_EMN:
    def __init__(self, output_dir, nb_classes, verbose):
        self.output_dir = output_dir
        self.nb_classes = nb_classes
        self.verbose = verbose

        # Hyperparameters ESN
        #self.esn_config = {'units':32, 'connect':0.7,'IS':0.1,"spectral":0.9,'leaky':1}

        self.units = 32
        self.spectral = 0.9
        self.input_scaling = [0.1, 1]
        self.connectivity = [0.3, 0.7]
        self.leaky = 1

        # Hyperparameters Convolutions
        #self.conv_config = {'epoch':500,'batch':25,'ratio':[0.6,0.7]}
        self.epoch = 500
        self.batch = 25
        self.ratio = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        self.num_filter = [60, 90, 120]

        self.final_params_selected = []

    def build_model(self, input_shape, nb_classes, len_series, ratio, num_filter):
        #ratio = self.conv_config['ratio']
        nb_rows = [np.int(ratio[0]*len_series), np.int(ratio[1]*len_series)]
        nb_cols = input_shape[2]

        input_layer = keras.layers.Input(input_shape)

        x_layer_1 = keras.layers.Conv2D(num_filter, (nb_rows[0], nb_cols), kernel_initializer='lecun_uniform', activation='relu',
                                        padding='valid', strides=(1, 1), data_format='channels_last')(input_layer)
        x_layer_1 = keras.layers.GlobalMaxPooling2D(
            data_format='channels_first')(x_layer_1)

        y_layer_1 = keras.layers.Conv2D(num_filter, (nb_rows[1], nb_cols), kernel_initializer='lecun_uniform', activation='relu',
                                        padding='valid', strides=(1, 1), data_format='channels_last')(input_layer)
        y_layer_1 = keras.layers.GlobalMaxPooling2D(
            data_format='channels_last')(y_layer_1)

        concat_layer = keras.layers.concatenate([x_layer_1, y_layer_1])

        layer_2 = keras.layers.Dense(
            64, kernel_initializer='lecun_uniform', activation='relu')(concat_layer)

        layer_3 = keras.layers.Dense(
            128, kernel_initializer='lecun_uniform', activation='relu')(layer_2)
        layer_3 = keras.layers.Dropout(0.25)(layer_3)

        output_layer = keras.layers.Dense(
            nb_classes, kernel_initializer='lecun_uniform', activation='softmax')(layer_3)

        model = keras.models.Model(input_layer, output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001), metrics=['accuracy'])

        #factor = 1. / np.cbrt(2)

        #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=factor, patience=100, min_lr=1e-4, cooldown=0, mode='auto')

        self.callbacks = [TqdmCallback(verbose=0)]

        return model

    def reshape_shuffle(self, x_train, y_train, nb_samples, res_units, len_series):

        # Generate template for train data
        train_data = np.zeros((nb_samples, 1, len_series, res_units))
        train_labels = np.zeros((nb_samples, self.nb_classes))

        # Generate Shuffle template
        # Array with size==samples, every value==index
        L_train = [x_train for x_train in range(nb_samples)]
        np.random.shuffle(L_train)

        # For every series -> shuffle train and labels
        for m in range(nb_samples):
            train_data[m, 0, :, :] = x_train[L_train[m], :, :]
            train_labels[m, :] = y_train[L_train[m], :]

        return train_data, train_labels

    def ff_esn(self, input_scaling, connect):
        units = self.units
        spectral_radius = self.spectral
        leaky = self.leaky
        n_in = 1

        escnn = Reservoir(units, n_in, input_scaling,
                          spectral_radius, connect, leaky)
        x_train = escnn.set_weights(self.x_train)
        x_val = escnn.set_weights(self.x_val)
        x_test = escnn.set_weights(self.x_test)

        nb_samples_train = np.shape(x_train)[0]
        nb_samples_val = np.shape(x_val)[0]
        nb_samples_test = np.shape(x_test)[0]
        self.len_series = x_val.shape[1]

        # Reshape test and train data. Train data is also shuffled
        x_val = np.reshape(x_val, (nb_samples_val,
                                   self.len_series, self.units, 1))
        x_test = np.reshape(x_test, (nb_samples_test,
                                     self.len_series, self.units, 1))

        x_train, y_train = self.reshape_shuffle(
            x_train, self.y_train, nb_samples_train, self.units, self.len_series)

        # From NCHW to NHWC
        x_train = tf.transpose(x_train, [0, 2, 3, 1])
        print('NHWC: {0}'.format(x_train.shape))
        # print(x_train.shape)

        return x_train, y_train, x_val, x_test

    def tune_esn(self):
        input_scaling_final = None
        connect_final = None
        num_filter_final = None
        x_train_final = None
        y_train_final = None
        x_val_final = None
        x_test_final = None
        duration_final = None
        model_final = None
        hist_final = None

        current_acc = 0

        self.it = 0

        for input_scaling in self.input_scaling:
            for connect in self.connectivity:

                x_train, y_train, x_val, x_test = self.ff_esn(input_scaling, connect)

                for num_filter in self.num_filter:

                    ratio = [0.1, 0.2]

                    # 2. Build Model
                    input_shape = (self.len_series, self.units, 1)
                    model = self.build_model(
                        input_shape, self.nb_classes, self.len_series, ratio, num_filter)

                    #if(self.verbose == True):
                        #model.summary()

                    # 3. Train Model
                    batch = self.batch
                    epoch = self.epoch

                    start_time = time.time()

                    hist = model.fit(x_train, y_train, batch_size=batch, epochs=epoch,
                                     verbose=False, validation_data=(x_val, self.y_val), callbacks=self.callbacks)

                    duration = time.time() - start_time

                    model_loss, model_acc = model.evaluate(
                        x_val, self.y_val, verbose=False)
                    print('val_loss: {0}, val_acc: {1}'.format(
                        model_loss, model_acc))

                    y_pred = model.predict(x_test)
                    # convert the predicted from binary to integer
                    y_pred = np.argmax(y_pred, axis=1)
                    df_metrics = calculate_metrics(
                        self.y_true, y_pred, duration)

                    temp_output_dir = self.output_dir + str(self.it)+'/'
                    create_directory(temp_output_dir)

                    df_metrics.to_csv(temp_output_dir +
                                      'df_metrics.csv', index=False)
                    model.save(temp_output_dir + 'model.hdf5')

                    params = [input_scaling, connect, num_filter, ratio]
                    param_print = pd.DataFrame(np.array([params], dtype=object), columns=[
                        'input_scaling', 'connectivity', 'num_filter', 'ratio'])
                    param_print.to_csv(temp_output_dir +
                                       'df_params.csv', index=False)

                    if (model_acc > current_acc):
                        print('New winner')
                        input_scaling_final = input_scaling
                        connect_final = connect
                        num_filter_final = num_filter
                        x_train_final = x_train
                        y_train_final = y_train
                        x_val_final = x_val
                        x_test_final = x_test
                        duration_final = duration
                        model_final = model
                        hist_final = hist
                        current_acc = model_acc

                    self.it += 1
                    keras.backend.clear_session()

        print('Final input_scaling: {0}; Final connectivity: {1}; Final filter: {2}'.format(
            input_scaling_final, connect_final, num_filter_final))
        self.final_params_selected.append(input_scaling_final)
        self.final_params_selected.append(connect_final)
        self.final_params_selected.append(num_filter_final)

        return x_train_final, y_train_final, x_val_final, x_test_final, model_final, hist_final, duration_final, current_acc, num_filter_final

    def fit(self, x_train, y_train, x_val, y_val, y_true):

        self.y_true = y_true

        self.x_test = x_val
        self.y_test = y_val

        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(x_train, y_train, test_size=0.2)

        # 1. Tune ESN and num_filter
        self.x_train, self.y_train, self.x_val, self.x_test, model_init, hist_init, duration_init, acc_init, num_filter = self.tune_esn()

        current_acc = acc_init
        hist_final = hist_init
        model_final = model_init
        duration_final = duration_init
        ratio_final = [0.1, 0.2]

        for ratio in self.ratio[1:]:

            # 1. Build Model
            input_shape = (self.len_series, self.units, 1)
            model = self.build_model(
                input_shape, self.nb_classes, self.len_series, ratio, num_filter)
            #if(self.verbose == True):
                #model.summary()

            # 3. Train Model
            batch = self.batch
            epoch = self.epoch

            start_time = time.time()

            hist = model.fit(self.x_train, self.y_train, batch_size=batch, epochs=epoch,
                             verbose=False, validation_data=(self.x_val, self.y_val), callbacks=self.callbacks)

            duration = time.time() - start_time

            model_loss, model_acc = model.evaluate(
                self.x_val, self.y_val, verbose=False)

            print('val_loss: {0}, val_acc: {1}'.format(
                model_loss, model_acc))

            y_pred = model.predict(self.x_test)
            # convert the predicted from binary to integer
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(self.y_true, y_pred, duration)

            temp_output_dir = self.output_dir + str(self.it)+'/'
            create_directory(temp_output_dir)

            df_metrics.to_csv(temp_output_dir +
                              'df_metrics.csv', index=False)
            model.save(temp_output_dir + 'model.hdf5')

            params = [self.final_params_selected[0],
                      self.final_params_selected[1], self.final_params_selected[2], ratio]
            param_print = pd.DataFrame(np.array([params], dtype=object), columns=[
                'input_scaling', 'connectivity', 'num_filter', 'ratio'])
            param_print.to_csv(temp_output_dir + 'df_params.csv', index=False)

            if (model_acc > current_acc):
                print('New winner')
                hist_final = hist
                model_final = model
                duration_final = duration
                ratio_final = ratio
                current_acc = model_acc

            keras.backend.clear_session()
            self.it += 1

        print('Final ratio: {0}'.format(ratio_final))
        self.final_params_selected.append(ratio_final)
        self.model = model_final
        self.hist = hist_final

        y_pred = self.model.predict(self.x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        param_print = pd.DataFrame(np.array([self.final_params_selected], dtype=object), columns=[
                                   'input_scaling', 'connectivity', 'num_filter', 'ratio'])

        param_print.to_csv(self.output_dir +
                           'df_final_params.csv', index=False)

        save_logs(self.output_dir, self.hist, y_pred, self.y_true,
                  duration_final, self.verbose, lr=False)

        keras.backend.clear_session()

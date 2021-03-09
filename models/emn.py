import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import pandas as pd

from layers.reservoir import Reservoir

from utils.utils import save_logs

import matplotlib

from tqdm.keras import TqdmCallback

class Classifier_EMN:
    def __init__(self, output_dir, nb_classes, verbose):
        self.output_dir = output_dir
        self.nb_classes = nb_classes
        self.verbose = verbose
        
        #Hyperparameters ESN
        #self.esn_config = {'units':32, 'connect':0.7,'IS':0.1,"spectral":0.9,'leaky':1}

        self.units = 32
        self.spectral = 0.9
        self.input_scaling = [0.1,1]
        self.connectivity = [0.3,0.7]
        self.leaky = 1

        #Hyperparameters Convolutions
        #self.conv_config = {'epoch':500,'batch':25,'ratio':[0.6,0.7]}
        self.epoch = 500
        self.batch = 25
        self.ratio = [[0.1,0.2],[0.2,0.3],[0.3,0.4],[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8]]

        self.final_params_selected = []
        
        
    def build_model(self, input_shape, nb_classes, len_series, ratio):
        #ratio = self.conv_config['ratio']
        nb_rows = [np.int(ratio[0]*len_series),np.int(ratio[1]*len_series)]
        nb_cols = input_shape[2]
        
        input_layer = keras.layers.Input(input_shape)
        
        x_layer_1 = keras.layers.Conv2D(120, (nb_rows[0], nb_cols), kernel_initializer='lecun_uniform', activation='relu',
                                       padding='valid', strides=(1,1), data_format = 'channels_last')(input_layer)
        x_layer_1 = keras.layers.GlobalMaxPooling2D(data_format = 'channels_first')(x_layer_1)
        
        
        
        y_layer_1 = keras.layers.Conv2D(120, (nb_rows[1], nb_cols), kernel_initializer='lecun_uniform', activation='relu',
                                       padding='valid', strides=(1,1), data_format = 'channels_last')(input_layer)
        y_layer_1 = keras.layers.GlobalMaxPooling2D(data_format = 'channels_last')(y_layer_1)
        
        
        
        concat_layer = keras.layers.concatenate([x_layer_1, y_layer_1])

        layer_2 = keras.layers.Dense(64, kernel_initializer='lecun_uniform', activation = 'relu')(concat_layer)

        layer_3 = keras.layers.Dense(128, kernel_initializer='lecun_uniform', activation = 'relu')(layer_2)
        layer_3 = keras.layers.Dropout(0.25)(layer_3)
        
        output_layer = keras.layers.Dense(nb_classes, kernel_initializer='lecun_uniform', activation='softmax')(layer_3)
        
        model = keras.models.Model(input_layer, output_layer)
        
        model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        
        #factor = 1. / np.cbrt(2)
		
        #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=factor, patience=100, min_lr=1e-4, cooldown=0, mode='auto')

        self.callbacks = [TqdmCallback(verbose=0)]
        
        
        return model
        
    def reshape_shuffle(self, x_train, y_train, nb_samples, res_units, len_series):

        #Generate template for train data
        train_data = np.zeros((nb_samples, 1, len_series, res_units))
        train_labels = np.zeros((nb_samples, self.nb_classes))

        #Generate Shuffle template
        L_train = [x_train for x_train in range(nb_samples)] #Array with size==samples, every value==index 
        np.random.shuffle(L_train)
        
        #For every series -> shuffle train and labels
        for m in range(nb_samples):
            train_data[m,0,:,:] = x_train[L_train[m],:,:] 
            train_labels[m,:] = y_train[L_train[m],:]
        
        return train_data, train_labels      
    
    def ff_esn(self, x_train, x_val, y_train, input_scaling, connect):
        units = self.units
        spectral_radius = self.spectral
        leaky = self.leaky
        n_in = 1
        
        escnn = Reservoir(units, n_in, input_scaling, spectral_radius, connect, leaky)
        x_train = escnn.set_weights(x_train)
        x_val = escnn.set_weights(x_val)

        self.nb_samples_train = np.shape(x_train)[0]
        self.nb_samples_val = np.shape(x_val)[0]
        self.len_series = x_val.shape[1]

        #Reshape test and train data. Train data is also shuffled
        x_val = np.reshape(x_val,(self.nb_samples_val, self.len_series, self.units, 1))
        x_train, y_train = self.reshape_shuffle(x_train, y_train, self.nb_samples_train, self.units, self.len_series) 

        #From NCHW to NHWC
        x_train = tf.transpose(x_train, [0, 2, 3, 1])
        print('NHWC: {0}'.format(x_train.shape))
        #print(x_train.shape)
        
        return x_train, x_val, y_train

    def tune_esn(self, x_train_init, x_val_init, y_train_init, y_val):
        input_scaling_final = None
        connect_final = None
        x_train_final = None
        x_val_final = None
        y_train_final = None 
        duration_final = None   
        model_final = None
        hist_final = None   

        current_acc = 0.0

        for input_scaling in self.input_scaling:
            for connect in self.connectivity:
                ratio = [0.1,0.2]

                x_train, x_val, y_train = self.ff_esn(x_train_init, x_val_init, y_train_init, input_scaling, connect)

                #2. Build Model
                input_shape = (self.len_series, self.units, 1)
                model = self.build_model(input_shape, self.nb_classes, self.len_series, ratio)
                if(self.verbose==True):
                    model.summary()
        
        
                #3. Train Model
                batch = self.batch
                epoch = self.epoch
        
                start_time = time.time()
        
                hist = model.fit(x_train, y_train, batch_size=batch, epochs=epoch, 
                    verbose=False, validation_data=(x_val,y_val), callbacks=self.callbacks)
        
                duration = time.time() - start_time

                model_acc = model.evaluate(x_train, y_train, verbose=False)[1]

                if (model_acc > current_acc):
                    input_scaling_final = input_scaling
                    connect_final = connect
                    x_train_final = x_train
                    x_val_final = x_val
                    y_train_final = y_train
                    duration_final = duration
                    model_final = model
                    hist_final = hist
                    #np.savetxt(self.output_dir+'IS.txt',IS_final)
                    #np.savetxt(self.output_dir+'connectivity.txt', connect_final)
                    current_acc = model_acc
                
                keras.backend.clear_session()
        print('Final input_scaling: {0}; Final connectivity: {1}'.format(input_scaling_final, connect_final))
        self.final_params_selected.append(input_scaling_final)
        self.final_params_selected.append(connect_final)

        return x_train_final, x_val_final, y_train_final, model_final, hist_final, duration_final, current_acc 
        
    def fit(self, x_train, y_train, x_val, y_val, y_true):
        #1. Tune ESN
        x_train, x_val, y_train, model_init, hist_init, duration_init, acc_init = self.tune_esn(x_train, x_val, y_train, y_val)

        current_acc = acc_init
        hist_final = hist_init
        model_final = model_init
        duration_final = duration_init
        ratio_final = [0.1,0.2]

        
        for ratio in self.ratio[1:]:

            #1. Build Model
            input_shape = (self.len_series, self.units, 1)
            model = self.build_model(input_shape, self.nb_classes, self.len_series, ratio)
            if(self.verbose==True):
                model.summary()
        
        
            #3. Train Model
            batch = self.batch
            epoch = self.epoch
            
            start_time = time.time()
            
            hist = model.fit(x_train, y_train, batch_size=batch, epochs=epoch,
                verbose=False, validation_data=(x_train,y_train), callbacks=self.callbacks)
            
            duration = time.time() - start_time

            model_acc = model.evaluate(x_val, y_val, verbose=False)[1]

            if (model_acc > current_acc):
                hist_final = hist
                model_final = model
                duration_final = duration
                ratio_final = ratio
                current_acc = model_acc
            
            keras.backend.clear_session()
        print('Final ratio: {0}'.format(ratio_final))
        self.final_params_selected.append(ratio_final)
        self.model = model_final
        self.hist = hist_final 

        y_pred = self.model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred , axis=1)

        param_print = pd.DataFrame(np.array([self.final_params_selected], dtype=object), columns=['input_scaling','connectivity','ratio'])
        param_print.to_csv(self.output_dir + 'final_params.csv',index=False)

        save_logs(self.output_dir, self.hist, y_pred, y_true, duration_final, self.verbose, lr=False)

        keras.backend.clear_session()
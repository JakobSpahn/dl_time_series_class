import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import pandas

from sklearn.model_selection import train_test_split

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import create_directory

import matplotlib

from tqdm.keras import TqdmCallback


class Classifier_LSTMFCN:
    
    def __init__(self, output_dir, nb_classes, tune=True, verbose=False):
        self.output_dir = output_dir
        self.nb_classes = nb_classes
        #if build == True:
            #self.model = self.build_model(input_shape, nb_classes)
            #if(verbose == True):
                #self.model.summary()
            #self.verbose = verbose
            #self.model.save_weights(self.output_dir + 'model_init.hdf5')
        self.verbose = verbose

        #Hyperparameter Tuning:
        if tune:
            self.lstm_cells = [8,64,128]
        else:
            self.lstm_cells = [8]
        return
    
    def build_model(self, input_shape, nb_classes, num_cells=64):
        input_layer = keras.layers.Input(input_shape)
        
        #Dimension shuffle. Receives input as multivariate time series with one single time step 
        #x_layer_1 = keras.layers.Permute((2,1))(input_layer)
        
        #LSTM Block
        
        x_layer_2 = keras.layers.LSTM(num_cells)(input_layer)
        #x_layer_2 = keras.layers.Attention()(x_layer_2)
        x_layer_2 = keras.layers.Dropout(0.8)(x_layer_2)
        
        
        # Conv Block 1
        
        y_layer_1 = keras.layers.Permute((2,1))(input_layer)
        y_layer_1 = keras.layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y_layer_1)
        y_layer_1 = keras.layers.BatchNormalization()(y_layer_1)
        y_layer_1 = keras.layers.Activation('relu')(y_layer_1)
        
        #Conv Block 2
        
        y_layer_2 = keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y_layer_1)
        y_layer_2 = keras.layers.BatchNormalization()(y_layer_2)
        y_layer_2 = keras.layers.Activation('relu')(y_layer_2)
        
        #Conv Block 3
        
        y_layer_3 = keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y_layer_2)
        y_layer_3  = keras.layers.BatchNormalization()(y_layer_3 )
        y_layer_3  = keras.layers.Activation('relu')(y_layer_3 )
        
        #Gap Layer
        
        y_layer_gap = keras.layers.GlobalAveragePooling1D()(y_layer_3)
        
        
        #FINAL
        
        concat_layer = keras.layers.concatenate([x_layer_2,y_layer_gap])
        
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(concat_layer)
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3),
            metrics=['accuracy'])
        
        factor = 1. / np.cbrt(2)
		
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=factor, patience=100, min_lr=1e-4, cooldown=0, mode='auto')

        file_path = self.output_dir + 'best_curr_weights.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
            save_best_only=True, save_weights_only=True, verbose=0)

        self.callbacks = [reduce_lr, model_checkpoint,TqdmCallback(verbose=0)] #reduce_lr,
        
        return model
    
    def train(self):
        batch_size = self.batch_size
        nb_epochs = self.nb_epochs

        curr_loss = 1e10
        final_model = None
        final_hist = None
        final_cell = None
        final_dur = None

        input_shape = self.x_train.shape[1:]


        for cell in self.lstm_cells:
            model = self.build_model(input_shape, self.nb_classes, cell)

            if self.verbose:
                model.summary()

            start_time = time.time()

            hist = model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=nb_epochs,
            verbose=False, validation_data=(self.x_test,self.y_test), callbacks=self.callbacks)
            
            model.load_weights(self.output_dir + 'best_curr_weights.hdf5')
            print("Weights loaded from {0}best_curr_weights.hdf5".format(self.output_dir))

            #Best model loaded, now evaluate on train set (No overfitting for test set)
            model_loss, model_acc = model.evaluate(self.x_train, self.y_train, batch_size=batch_size, verbose=False)
            print('Best weights --> val loss: {0}, val acc: {1}'.format(model_loss, model_acc))

            duration = time.time() - start_time
            
            
            y_pred = model.predict(self.x_test)
            # convert the predicted from binary to integer
            y_pred = np.argmax(y_pred , axis=1)
            df_metrics = calculate_metrics(self.y_true,y_pred,duration)

            temp_output_dir = self.output_dir+'num_cells_'+str(cell)+'/'
            create_directory(temp_output_dir)
            
            df_metrics.to_csv(temp_output_dir + 'df_metrics.csv', index=False)
            model.save(temp_output_dir+ 'model.hdf5')

            if(model_loss < curr_loss):
                curr_loss = model_loss
                final_cell = cell
                #curr_loss = model_loss
                final_model = model
                final_hist = hist
                final_dur = duration
                final_model.save(self.output_dir + 'best_model.hdf5')

            keras.backend.clear_session()

        

        print('Final Cell Selected:',final_cell)
        file_cells = open(self.output_dir + 'best_num_cells.txt','w')
        file_cells.write(str(final_cell))
        file_cells.close()

        return final_model, final_hist, final_dur

    def fit(self, x_train, y_train, x_test, y_test,y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        
        self.x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
        self.x_test = np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1]))
        self.y_train = y_train
        self.y_test = y_test
        self.y_true = y_true

        print('x_train.shape: {0}'.format(self.x_train.shape))
        

        self.batch_size = 128
        self.nb_epochs = 2000

        self.model, hist, duration = self.train()

        #Finally predict on test
        y_pred = self.model.predict(self.x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred , axis=1)

        save_logs(self.output_dir, hist, y_pred, self.y_true, duration, self.verbose)
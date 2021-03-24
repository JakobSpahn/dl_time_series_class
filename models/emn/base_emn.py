from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm.keras import TqdmCallback
from layers.reservoir import Reservoir
import numpy as np
import logging
import tensorflow.keras as keras
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)


class Base_Classifier_EMN(BaseEstimator, ClassifierMixin):
    def __init__(self, res_units=32,
                 spectral_radius=0.9, input_scaling=0.1,
                 connectivity=0.3, leaky=1, n_in=1,
                 epochs=500, batch_size=25,
                 ratio=[0.1, 0.2], num_filter=120,
                 nb_classes = None, verbose=True):
        self.res_units = res_units
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.connectivity = connectivity
        self.leaky = leaky
        self.n_in = n_in
        self.epochs = epochs
        self.batch_size = batch_size
        self.ratio = ratio
        self.num_filter = num_filter

        self.nb_classes = nb_classes
        self.verbose = verbose

    def build_model(self, input_shape, nb_classes, len_series):
        nb_rows = [np.int(self.ratio[0]*len_series),
                   np.int(self.ratio[1]*len_series)]
        nb_cols = input_shape[2]

        input_layer = keras.layers.Input(input_shape)

        x_layer_1 = keras.layers.Conv2D(self.num_filter, (nb_rows[0], nb_cols), kernel_initializer='lecun_uniform', activation='relu',
                                        padding='valid', strides=(1, 1), data_format='channels_first')(input_layer)
        x_layer_1 = keras.layers.GlobalMaxPooling2D(
            data_format='channels_first')(x_layer_1)

        y_layer_1 = keras.layers.Conv2D(self.num_filter, (nb_rows[1], nb_cols), kernel_initializer='lecun_uniform', activation='relu',
                                        padding='valid', strides=(1, 1), data_format='channels_first')(input_layer)
        y_layer_1 = keras.layers.GlobalMaxPooling2D(
            data_format='channels_first')(y_layer_1)

        concat_layer = keras.layers.concatenate([x_layer_1, y_layer_1])
        #concat_layer = keras.layers.Dense(128, kernel_initializer = 'lecun_uniform', activation = 'relu')(concat_layer)
        #concat_layer = keras.layers.Dense(128, kernel_initializer = 'lecun_uniform', activation = 'relu')(concat_layer)
        concat_layer = keras.layers.Dropout(0.25)(concat_layer)

        output_layer = keras.layers.Dense(
            nb_classes, kernel_initializer='lecun_uniform', activation='softmax')(concat_layer)

        model = keras.models.Model(input_layer, output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001), metrics=['accuracy'])

        self.callbacks_ = []

        if self.verbose:
            self.callbacks_.append(TqdmCallback(verbose=0))

        return model

    def _reshape_shuffle(self, x_train, y_train, nb_samples, nb_classes, len_series):
        #train_data = x_train.reshape(nb_samples, 1, len_series, self.res_units)
        #train_labels = y_train.reshape(nb_samples, nb_classes)
        # Generate template for train data
        train_data = np.zeros((nb_samples, 1, len_series, self.res_units))
        train_labels = np.zeros((nb_samples, nb_classes))

        # Generate Shuffle template
        # Array with size==samples, every value==index
        L_train = [x_train for x_train in range(nb_samples)]
        np.random.shuffle(L_train)

        # For every series -> shuffle train and labels
        for m in range(nb_samples):
            train_data[m, 0, :, :] = x_train[L_train[m], :, :]
            train_labels[m, :] = y_train[L_train[m], :]

        return train_data, train_labels

    def fit(self, x, y):
        if not self.nb_classes:
            raise ValueError('nb_classes is an essential parameter')

        self.escnn_ = Reservoir(self.res_units, self.n_in,
                                self.input_scaling, self.spectral_radius,
                                self.connectivity, self.leaky, verbose=self.verbose)
        x = self.escnn_.set_weights(x)

        nb_samples_x = np.shape(x)[0]
        len_series = x.shape[1]
        input_shape = (1, len_series, self.res_units)

        x, y = self._reshape_shuffle(x, y, nb_samples_x, self.nb_classes, len_series)

        # From NCHW to NHWC
        #x = tf.transpose(x, [0, 2, 3, 1])

        self.model_ = self.build_model(input_shape, self.nb_classes, len_series)

        if self.verbose:
            self.model_.summary()

        self.hist_ = self.model_.fit(x, y, batch_size=self.batch_size, epochs=self.epochs,
                                    verbose=False, callbacks=self.callbacks_)

        keras.backend.clear_session()

        return self

    def predict(self, x):

        check_is_fitted(self)

        x = self.escnn_.set_weights(x)
        nb_samples_test = np.shape(x)[0]
        len_series = x.shape[1]
        x = np.reshape(x, (nb_samples_test, 1, len_series, self.res_units))

        y_pred = self.model_.predict(x)
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred

    def score(self, x, y):
        x = self.escnn_.set_weights(x)
        nb_samples_x = np.shape(x)[0]
        len_series = x.shape[1]
        x = np.reshape(x, (nb_samples_x, 1, len_series, self.res_units))

        outputs = self.model_.evaluate(x, y, verbose=False)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model_.metrics_names, outputs):
            if name in ['accuracy', 'acc']:
                return output

import time

import tensorflow as tf
import numpy as np
from tqdm.auto import trange

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV


class Base_Classifier_ROCKET_TF(BaseEstimator, ClassifierMixin):
    '''
    X/Input needs to be in shape=(num_samples, c_in, inp_len)
    '''
    def __init__(self, n_kernels=10_000, kss=[7, 9, 11], verbose=True):
        self.n_kernels = n_kernels
        self.kss = kss
        self.verbose = verbose

    def fit(self, x, y):
        self.c_in_ = x.shape[1]
        self.inp_len_ = x.shape[-1]

        self.kernels_ = ROCKET_TF(
            self.c_in_, self.inp_len_, self.n_kernels, self.kss, self.verbose)

        x = tf.convert_to_tensor(x, dtype=tf.float32)

        time_a = time.perf_counter()
        self.x_train_ = self.kernels_.call(x)
        time_b = time.perf_counter()
        self.train_timings_ = [time_b - time_a]

        if self.verbose:
            print('fit classifier')
        time_a = time.perf_counter()
        self.ridge_cv_ = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10), normalize=True)
        self.ridge_cv_.fit(self.x_train_, y)
        time_b = time.perf_counter()
        self.train_timings_.append(time_b - time_a)

        return self

    def predict(self, x):
        check_is_fitted(self)

        x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Check if x_test has already been transformed and get the timing for that
        try:
            if tf.size(self.x_test_, out_type=tf.bool):
                self.test_timings_ = [self.test_timings_[0]]
        except AttributeError:
            time_a = time.perf_counter()
            self.x_test_ = self.kernels_.call(x)
            time_b = time.perf_counter()
            self.test_timings_ = [time_b - time_a]

        if self.verbose:
            print('predict')
        time_a = time.perf_counter()
        y_pred = self.ridge_cv_.predict(self.x_test_)
        time_b = time.perf_counter()
        self.test_timings_.append(time_b - time_a)

        return y_pred

    def score(self, x, y):
        check_is_fitted(self)

        x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Check if x_test has already been transformed and get the timing for that
        try:
            if tf.size(self.x_test_, out_type=tf.bool):
                self.test_timings_ = [self.test_timings_[0]]
        except AttributeError:
            time_a = time.perf_counter()
            self.x_test_ = self.kernels_.call(x)
            time_b = time.perf_counter()
            self.test_timings_ = [time_b - time_a]

        if self.verbose:
            print('test')
        time_a = time.perf_counter()
        acc = self.ridge_cv_.score(self.x_test_, y)
        time_b = time.perf_counter()
        self.test_timings_.append(time_b - time_a)

        return acc


class ROCKET_TF(tf.Module):
    '''
    ROCKET_TF is a powerful re-implementation of of the ROCKET-method based on TensorFlow.
    Due to being reimplemented in TensorFlow, it allows the use of GPU. 
    This makes it possible to transform very large time series data 
    in a shorter time than the original CPU-only implementation.

    The __init__ method is called with the following arguments:
    c_in: number of channels. For univariate time series, this corresponds to 1.
    inp_len: length of time series
    n_kernels: number of kernels
    kss: array which holds the kernel sizes (batch_shape) to chose from during each kernel initialization

    To transform a time series the call() method is called:
    x: 3d tensor of type tf.tensor with dtype=tf.float32. 
    The data format for the convolutions is NWC.
    An exemplary shape of input tensor: shape=(num_samples, c_in, inp_len)
    '''

    def __init__(self, c_in, inp_len, n_kernels=10_000, kss=[7, 9, 11], verbose=True):
        kss = [ks for ks in kss if ks < inp_len]
        _weights = []
        _biases = []
        _dilations = []
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0,
                                            np.log2((inp_len - 1) // (ks - 1)))

            weight = tf.random.normal([ks, c_in, 1], dtype=tf.float32)
            weight -= tf.math.reduce_mean(weight)

            bias = 2 * (tf.random.normal([1], dtype=tf.float32) - .5)

            _weights.append(weight)
            _biases.append(bias)
            _dilations.append(dilation)

        self.weights = _weights
        self.biases = _biases
        self.dilations = _dilations
        self.n_kernels = n_kernels
        self.verbose = verbose

    def call(self, x):
        _output = []
        if self.verbose:
            n_kernels = trange(self.n_kernels, desc='apply kernels')
        else:
            n_kernels = range(self.n_kernels)

        for i in n_kernels:
            bias = self.biases[i]
            dilation = self.dilations[i]
            weight = self.weights[i]

            tensor = tf.nn.conv1d(
                x, filters=weight, stride=1, padding='VALID', data_format='NCW', dilations=dilation)
            tensor = tf.nn.bias_add(tensor, bias, data_format='NCW')

            _max = tf.math.reduce_max(tensor, axis=-1)

            temp = tf.cast(tf.math.greater(tensor, 0), tf.float32)
            _ppv = tf.math.reduce_sum(temp, axis=-1) / tensor.shape[-1]

            _output.append(_max)
            _output.append(_ppv)

        return tf.concat(_output, axis=1)

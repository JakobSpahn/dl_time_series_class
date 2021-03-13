import time
import numpy as np
from numba import njit, prange
from tqdm.auto import trange

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV


class Base_Classifier_ROCKET(BaseEstimator, ClassifierMixin):
    '''
    X/Input needs to be in shape=(num_samples, inp_len)
    '''
    def __init__(self, n_kernels=10_000, kss=(7, 9, 11), verbose=True):
        self.n_kernels = n_kernels
        self.kss = kss
        self.verbose = verbose

    def fit(self, x, y):
        self.inp_len_ = x.shape[-1]

        if self.verbose:
            print('generate kernels')
        self.kernels_ = generate_kernels(self.inp_len_, self.n_kernels, self.kss)

        if self.verbose:
            print('apply kernels')
        time_a = time.perf_counter()
        self.x_train_ = apply_kernels(x, self.kernels_)
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

        # Check if x_test has already been transformed and get the timing for that
        try:
            if self.x_test_.size:
                self.test_timings_ = [self.test_timings_[0]]
        except AttributeError:
            if self.verbose:
                print('apply kernels')
            time_a = time.perf_counter()
            self.x_test_ = apply_kernels(x, self.kernels_)
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

        # Check if x_test has already been transformed and get the timing for that
        try:
            if self.x_test_.size:
                self.test_timings_ = [self.test_timings_[0]]
        except AttributeError:
            if self.verbose:
                print('apply kernels')
            time_a = time.perf_counter()
            self.x_test_ = apply_kernels(x, self.kernels_)
            time_b = time.perf_counter()
            self.test_timings_ = [time_b - time_a]

        if self.verbose:
            print('test')
        time_a = time.perf_counter()
        acc = self.ridge_cv_.score(self.x_test_, y)
        time_b = time.perf_counter()
        self.test_timings_.append(time_b - time_a)

        return acc


# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)
@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64,Tuple((int32,int32,int32)))")
def generate_kernels(input_length, num_kernels, kss=(7,9,11)):

    candidate_lengths = np.array(kss, dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    a1 = 0

    for i in range(num_kernels):

        _length = lengths[i]

        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings

@njit(fastmath = True)
def apply_kernel(X, weights, length, bias, dilation, padding):

    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < input_length:

                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max

@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel = True, fastmath = True)
def apply_kernels(X, kernels):

    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype = np.float64) # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0 # for weights
        a2 = 0 # for features

        for j in range(num_kernels):

            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = \
            apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X
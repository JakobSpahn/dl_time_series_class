import scipy as sp
import numpy as np
import tensorflow as tf
from tqdm.auto import trange

class Reservoir:
    def __init__(self, units, n_in, IS, spectral_radius, connectivity, leaky, bias=False, activation='tanh',seed=1882517, verbose = True):
        self.verbose = verbose
        
        self.units = units
        self.n_in = n_in
        self.IS = IS
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.leaky = leaky
        self.bias = bias
        self.activation=activation

        if(self.verbose):
            print('Initializing reservoir space...')
        
        #Set Seed for Numpy:
        rng = np.random.RandomState(seed)
        
        #Random Initialization of input weight values
        self.W_in = 2*np.array(np.random.random(size=(units, n_in)))-1
        
        #Generate sparsity matrix
        W_res_temp = sp.sparse.rand(self.units, self.units, self.connectivity) #(-> units x units), density of matrix = sparsity
        
        #Find one (k=1) eigenvalue and the corresponding eigenvector
        vals, vecs = sp.sparse.linalg.eigsh(W_res_temp, k=1)
        
        #Init reservoir weight values
        self.W_res = self.spectral_radius * W_res_temp / vals[0]
        
        #Init bias
        if bias:
            b_bound=0.1
            self.b = 2 * b_bound * np.random.random(size=(self.units, 1))-b_bound
        else:
            self.b = 0
    
    def set_weights(self, data):
        #print('Setting weights')

        n_forget_steps = 0
        num_samples = np.shape(data)[0]
        num_frames = np.shape(data)[1]
        
        #Init weight template: num_samples * timestamps * units
        weights = np.empty((num_samples,(num_frames-n_forget_steps), self.units), np.float32)
        
        if self.verbose:
            range_num_samples = trange(num_samples, desc='setting reservoir weights')
        else: 
            range_num_samples = range(num_samples)

        #For every time series in the dataset:
        for i_sample in range_num_samples:
            series = data[i_sample]
            #print('Create weights of %4d-th sample'%(i_sample)
            collect_weights = np.zeros((num_frames-n_forget_steps, self.units))
            x = np.zeros((self.units, 1))
            
            #Now apply weights for every timestamp
            for t in range(num_frames):
                u_t = np.asarray([series[t,:]]).T #Get every timestamp. Requires 3D shape to begin with
                
                #ToDo: Extend Dict
                xUpd = {
                    'tanh': lambda bias: np.tanh(np.dot(self.W_in, self.IS*u_t) + \
                                                 np.dot(self.W_res.toarray(), x) + bias)
                }[self.activation](self.b)
                
                x = (1-self.leaky)*x + self.leaky*xUpd
                if t >= n_forget_steps:
                    collect_weights[t-n_forget_steps,:] = x.T
                
                
            
            collect_weights = np.asarray(collect_weights)
            weights[i_sample] = collect_weights

        

        return weights
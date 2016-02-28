__author__ = '1001925'


import os
import sys
import timeit
import utils
import numpy as np
import pickle



import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from normLib import normActiv,denormActiv,sigmoid,tanh
from utils_copy import ravel_param, unravel_param,save,load,RMSE,error_ratio,load_data

def load_numerical_params_ae(ae):
    ae = load(ae)
    W1=ae.W1.get_value(borrow=True)
    W2=ae.W2.get_value(borrow=True)
    b1=ae.b1.get_value(borrow=True)
    b2=ae.b2.get_value(borrow=True)
    G =ae.G.get_value(borrow= True)


    return W1,W2,b1,b2,G,


def load_tensor_params_ae(ae):
    ae = load(ae)
    W1=ae.W1
    W2=ae.W2
    b1=ae.b1
    b2=ae.b2
    G =ae.G
    G_decay = ae.G_decay
    multi_sparse_weight = ae.multi_sparse_weight
    multi_sparsity = ae.multi_sparsity
    return W1,W2,b1,b2,G,G_decay,multi_sparsity,multi_sparse_weight


def load_numerical_params_sae(sae):
    sae = load(sae)
    W1=sae.W1.get_value(borrow=True)
    W2=sae.W2.get_value(borrow=True)
    b1=sae.b1.get_value(borrow=True)
    b2=sae.b2.get_value(borrow=True)
    G =sae.G.get_value(borrow= True)

    W3=sae.W3.get_value(borrow=True)
    W4=sae.W4.get_value(borrow=True)
    b3=sae.b3.get_value(borrow=True)
    b4=sae.b4.get_value(borrow=True)
    G_share =sae.G_share.get_value(borrow= True)

    return W1,W2,b1,b2,G,W3,W4,b3,b4,G_share


def load_data_sae(param_list,data,indi_matrix,batch_size,train = True):

    print('Loading training data for sae ................')
    numMod = len(param_list)
    raw = np.load(data)
    raw = raw[0:2400,:]
    visible_size = np.shape(raw)[1]
    visible_size_Mod = int(visible_size/numMod)
    train_list = [0]*numMod
    trainstats_list = [0]*numMod

    raw1 = raw * indi_matrix

    for i in range(numMod):
        train_list[i], trainstats_list[i] = normActiv(raw1[:,i*visible_size_Mod:(i+1)*visible_size_Mod])
    datasets = np.concatenate(train_list, axis=1)

    data_test = datasets

    datasets = theano.shared(np.asarray(datasets,dtype=theano.config.floatX),name='datasets', borrow=True)
    indi_matrix1 = theano.shared(np.asarray(indi_matrix,dtype = theano.config.floatX ),name='indi_matrix', borrow=True)
    n_train_batches = datasets.get_value(borrow=True).shape[0] / batch_size

    if train:

        return datasets,indi_matrix1,data_test,n_train_batches,numMod,raw,trainstats_list,visible_size_Mod

    else:
        return datasets,indi_matrix1, n_train_batches



def get_h1(missing,x,W1,b1,G):

    if missing:
            # bias_matrix = T.dot(self.indi_matrix,self.bias_matrix)
            # hidden_values = T.tanh(T.dot(self.x, self.W1*self.G) + bias_matrix + self.b1)
        hidden_values = np.tanh(np.dot(x, W1*G) + b1)

    else:
        hidden_values = np.tanh(np.dot(x, W1*G) + b1)
    return hidden_values

def get_h2(h1,W2,b2,G_share):
    hidden_values = np.tanh(np.dot(h1,W2*G_share) +b2)
    return hidden_values

def get_h3(h2,W3,b3,G_share):
    hidden_values = np.tanh(np.dot(h2, W3*G_share.T) + b3)
    return hidden_values

def get_reconstruct(missing,h3,W4,b4,G):

    if missing:
        return np.dot(h3, W4*G.T) + b4
        #return np.tanh(np.dot(h3, W4*G.T) + b4)
    else:
        return np.tanh(np.dot(h3, W4*G.T) + b4)





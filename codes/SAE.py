__author__ = '1001925'


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from normLib import normActiv,denormActiv,sigmoid,tanh
from utils_copy import ravel_param, unravel_param,save,load,RMSE,error_ratio,load_data
from load_models import load_numerical_params_ae,load_tensor_params_ae,load_data_sae,load_numerical_params_sae

from load_models import get_h1,get_h2,get_h3,get_reconstruct

import os
import sys
import timeit
import utils
import numpy as np
import pickle
from AE_RMSE_Missing import test_AE as build_block


class SAE(object):

    def __init__(self,
                 ae1,
                 ae2,
        input = None,
        indi_matrix = None,
        missing = False,
        param_list = None,
                 ):

        self.x = input
        self.indi_matrix = indi_matrix
        self.param_list = param_list
        self.missing = missing


        self.W1,self.W4,self.b1,self.b4,self.G = load_tensor_params_ae(ae1)
        self.W2,self.W3,self.b2,self.b3,self.G_share = load_tensor_params_ae(ae2)

        self.G_decay1 = ae1.G_decay
        self.G_decay2 = ae2.G_decay

        self.multi_sparsity1 = ae1.multi_sparsity
        self.multi_sparsity2 = ae1.multi_sparsity

        self.multi_sparse_weight1 = ae1.multi_sparse_weight
        self.multi_sparse_weight2 = ae2.multi_sparse_weight

        self.norm_indi = ae1.norm_indi

        self.params = [self.W1, self.W2, self.W3, self.W4, self.b1, self.b2,self.b3, self.b4]

    def get_h1(self):

        if self.missing:
            # bias_matrix = T.dot(self.indi_matrix,self.bias_matrix)
            # hidden_values = T.tanh(T.dot(self.x, self.W1*self.G) + bias_matrix + self.b1)
            hidden_values = T.tanh(T.dot(self.x, self.W1*self.G) + self.b1)

        else:
            hidden_values = T.tanh(T.dot(self.x, self.W1*self.G) + self.b1)
        return hidden_values

    def get_h2(self,h1):
        hidden_values = T.tanh(T.dot(h1, self.W2*self.G_share) + self.b2)
        return hidden_values

    def get_h3(self,h2):
        hidden_values = T.tanh(T.dot(h2, self.W3*self.G_share.T) + self.b3)
        return hidden_values

    def get_reconstructed(self, h3):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        if self.missing:
            return T.dot(h3, self.W4*self.G.T) + self.b4
        else:
            return T.tanh(T.dot(h3, self.W4*self.G.T) + self.b4)

    def finetuning(self,learning_rate):

        h1 = self.get_h1()
        h2 = self.get_h2(h1)
        h3 = self.get_h2(h2)
        z = self.get_reconstructed(h3)

        residue = z - self.x


        if self.missing:

            cost_part1 = 0.5*T.sum(((self.indi_matrix*residue)**2)/self.norm_indi)\
                         +0.5*(T.sum((self.W1*self.G_decay1)**2)+T.sum((self.W2*self.G_decay1.T)**2))

            mean_activation1 = T.dot(self.indi_matrix.T, h1)/self.norm_indi

            KL1 = (1+mean_activation1)*(T.log(1+mean_activation1) -T.log(1+self.multi_sparsity1))\
                 + (1-mean_activation1)*(T.log(1-mean_activation1) - T.log(1-self.multi_sparsity1))

            mean_activation2 = T.mean(h3,axis=0)

            cost_part2 = 1/self.x.shape[1]*T.sum(self.multi_sparse_weight1 * KL1)

            KL2 = (1+mean_activation2)*(T.log(1+mean_activation2) -T.log(1+self.multi_sparsity2))\
                 + (1-mean_activation2)*(T.log(1-mean_activation2) - T.log(1-self.multi_sparsity2))

            cost_part3 = T.sum(self.multi_sparse_weight2 * KL2)

            cost = cost_part1 + cost_part2 + cost_part3

            gparams = T.grad(cost, self.params)
            # generate the list of updates
            updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ]


        else:

            cost_part1 = 0.5*T.sum((residue**2)/self.x.shape[0])\
                         +0.5*(T.sum((self.W1*self.G_decay1)**2)+T.sum((self.W2*self.G_decay1.T)**2))

            mean_activation1 = T.mean(h1,axis=0)

            KL1 = (1+mean_activation1)*(T.log(1+mean_activation1) -T.log(1+self.multi_sparsity1))\
                 + (1-mean_activation1)*(T.log(1-mean_activation1) - T.log(1-self.multi_sparsity1))

            mean_activation2 = T.mean(h3,axis=0)

            cost_part2 = T.sum(self.multi_sparse_weight1 * KL1)

            KL2 = (1+mean_activation2)*(T.log(1+mean_activation2) -T.log(1+self.multi_sparsity2))\
                 + (1-mean_activation2)*(T.log(1-mean_activation2) - T.log(1-self.multi_sparsity2))

            cost_part3 = T.sum(self.multi_sparse_weight2 * KL2)

            cost = cost_part1 + cost_part2 + cost_part3

            gparams = T.grad(cost, self.params)
            # generate the list of updates
            updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ]

        return (cost, updates)

    def get_cost(self):

        h1 = self.get_h1()
        h2 = self.get_h2(h1)
        h3 = self.get_h2(h2)
        z = self.get_reconstructed(h3)

        residue = z*self.indi_matrix - self.x

        if self.missing:

            cost_part1 = 0.5*T.sum(((self.indi_matrix*residue)**2)/self.norm_indi)

        else:

            cost_part1 = 0.5*T.sum((residue**2)/self.x.shape[0])

        return cost_part1



def test_SAE(data = '',validationdata='',param_list= [], missing1=True, missing2 = False,share1 = False,share2 = True,
             missingrate1 = 0, missingrate2 = 0,learningrate = 0.08, training_epochs = 10,batch_size = 3000,
             output_folder = ''):


    newpath = '../Result/'+ output_folder
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    ae1, h1, h_valid, indi_matrix_test,indi_matrix_valid_test = \
        build_block(data,validationdata, param_list,share1,missing1, missingrate1,
                      learningrate, training_epochs,
                      batch_size, output_folder='m_HI_1',order = 1)

    print('Fininshed training the first auto encoder')


    ae2,h2,h2_valid,indi_matrix_test2,indi_matrix_valid_test2 = \
        build_block(h1, h_valid, param_list,share2,missing = missing2, missing_rate = missingrate2,
                      learning_rate=learningrate, training_epochs= training_epochs,
                      batch_size = 30, output_folder='m_HI_2',order = 2)

    print('Fininshed training the second auto encoder')


    datasets,indi_matrix,data_test,n_train_batches,numMod,raw,trainstats_list,visible_size_Mod =\
        load_data_sae(param_list,data,indi_matrix_test,batch_size,train = True)


    valid_batch_size = 306
    validset,valid_indi_matrix, n_valid_batches =\
        load_data_sae(param_list,validationdata,indi_matrix_valid_test,valid_batch_size,train = False)


    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')
    y = T.matrix('y')# the data is presented as rasterized images
    # end-snippet-2

    sae = SAE(
        ae1,
        ae2,
        input = x,
        indi_matrix = y,
        missing = missing1,
        param_list = None,
    )

    cost,updates = sae.finetuning(learningrate)

    train_sae = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: datasets[index * batch_size: (index + 1) * batch_size],
            y: indi_matrix[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='warn',
    )


    validate_cost = sae.get_cost()

    validate_sae = theano.function(
        [index],
        validate_cost,
        givens={
            x: validset[index * valid_batch_size: (index + 1) * valid_batch_size],
            y: valid_indi_matrix[index * valid_batch_size: (index + 1) * valid_batch_size]
        },
        on_unused_input='warn',
    )


    ###############
    # TRAIN MODEL #
    ###############
    print('... training the sae model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    best_epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        c= []
        for minibatch_index in range(int(n_train_batches)):

            a = train_sae(minibatch_index)
            c.append(a)
            print(a)
            # iteration number indicate how many batches we have already runned on
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_sae(i)
                                     for i in range(int(n_valid_batches))]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation cost %f ' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    # save the best model
                    print('saving the model for epoch %i' % epoch)
                    best_epoch = epoch
                    #f = open('../Result_test/best_model_epoch_' +str() + '.txt', 'w')
                    save(newpath + '/best_model_epoch_' +str(epoch) +'.pkl', sae)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f'

        )
        % (best_validation_loss)
    )
    print( sys.stderr, ('The code for file ' +os.path.split(__file__)[1]
                          +' ran for %.1fs' % ((end_time - start_time))))


    print('Best epoch is %i' % best_epoch )
    print('Now we starting computing the RMSE and error ratio')

    sae = newpath + '/best_model_epoch_' +str(best_epoch) +'.pkl'

    W1,W2,b1,b2,G,W3,W4,b3,b4,G_share = load_numerical_params_sae(sae)

    h1 = get_h1(missing1,data_test,W1,b1,G)
    h2 = get_h2(h1,W2,b2,G_share)
    h3 = get_h3(h2,W3,b3,G_share)
    reconstruction = get_reconstruct(missing1,h3,W4,b4,G)

    print(reconstruction)

    np.savetxt(newpath+ '/output_' +str(best_epoch) + '.txt',reconstruction,delimiter=',')


    f = open(newpath +'/AE_' +str(best_epoch) + '.txt', 'w')


    for i in range(int(numMod)):
        np.savetxt(newpath +'/Raw_' +str(i) + '.txt',
        raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],delimiter=',')
        np.savetxt(newpath+'/Recstru_' +str(i) + '_' +str(best_epoch) + '.txt',
        denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],trainstats_list[i]),delimiter=',')



        print(f, 'AE RMSE for Modality', i, str(RMSE(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                       trainstats_list[i]))))


        f.write('AE RMSE for Modality'+ '\t' + str(i) +'\t'+ str(RMSE(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                          trainstats_list[i]))) + '\n')


        print(f, 'AE error ratio for Modality', i, str(error_ratio(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                       trainstats_list[i]))))

        f.write('AE error ratio for Modality'+ '\t' + str(i) +'\t'+ str(error_ratio(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                          trainstats_list[i]))) + '\n')

if __name__ == '__main__':
    ###param_list consists : decay_weight(lambda),sparsity(rho) and sparse_weight(beta)
    param_list = [[1e-5,0.01,1e-5],[1e-5,0.01,1e-5]]
    dataset = '../Data/train_HumIllumDF_new.npy'
    validationset = '../Data/test_HumIllumDF_new.npy'
    test_SAE(data=dataset,validationdata = validationset,param_list=param_list)


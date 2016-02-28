__author__ = '1001925'


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from normLib import normActiv,denormActiv,sigmoid,tanh
from utils_copy import ravel_param, unravel_param,save,load,RMSE,error_ratio,load_data,load_sae_test_data
from load_models import load_numerical_params_ae,load_tensor_params_ae,load_data_sae,load_numerical_params_sae

from load_models import get_h1,get_h2,get_h3,get_reconstruct

import os
import sys
import timeit
import utils
import numpy as np
import pickle
from AE_RMSE_Missing import test_AE as build_block

#theano.config.optimizer= 'fast_compile'

class SAE(object):

    def __init__(self,
                 ae1,
                 ae2,
        input = None,
        indi_matrix = None,
        missing = None,
        param_list = None,
                 ):

        self.x = input
        self.indi_matrix = indi_matrix
        self.param_list = param_list
        self.missing = missing


        self.W1,self.W4,self.b1,self.b4,self.G,self.G_decay1,self.multi_sparsity1,self.multi_sparse_weight1= \
            load_tensor_params_ae(ae1)
        self.W2,self.W3,self.b2,self.b3,self.G_share,self.G_decay2,self.multi_sparsity2,self.multi_sparse_weight2 = \
            load_tensor_params_ae(ae2)

        # print('Checking the values here')
        #
        # W1,W2,b1,b2,G = load_numerical_params_ae(ae1)
        # print(W1,W2,b1,b2,G)
        #
        G_decay1 = self.G_decay1.get_value(borrow = True)
        print(G_decay1.shape)
        G_decay2 = self.G_decay2.get_value(borrow = True)
        print(G_decay2.shape)
        W1 = self.W1.get_value(borrow = True)
        print(W1.shape)
        W4 = self.W4.get_value(borrow = True)
        print(W4.shape)
        W2 = self.W2.get_value(borrow = True)
        print(W2.shape)
        W3 = self.W3.get_value(borrow = True)
        print(W3.shape)
        # Actually, the value of the model is correctly passed to the sae model

        norm_indi = T.sum(self.indi_matrix, axis = 0)
        self.norm_indi = norm_indi



        #self.params = [self.W1]
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
            #return T.tanh(T.dot(h3, self.W4*self.G.T) + self.b4)
        else:
            #return T.dot(h3, self.W2*self.G.T) + self.b2
            return T.tanh(T.dot(h3, self.W4*self.G.T) + self.b4)

    def finetuning(self,learning_rate):

        print('...Doing finetuning gradient descent here...')

        h1 = self.get_h1()
        h2 = self.get_h2(h1)
        h3 = self.get_h3(h2)
        z = self.get_reconstructed(h3)

        residue = z - self.x


        if self.missing:


            cost_part1 = 0.5*T.sum(((self.indi_matrix*residue)**2)/self.norm_indi)\
                         +0.5*(T.sum((self.W1*self.G_decay1)**2)+T.sum((self.W2*self.G_decay2)**2))\
                         +0.5*(T.sum((self.W3*self.G_decay2.T)**2)+T.sum((self.W4*self.G_decay1.T)**2))

            mean_activation1 = T.dot(self.indi_matrix.T, h1)/self.norm_indi

            KL1 = (1+mean_activation1)*(T.log(1+mean_activation1) -T.log(1+self.multi_sparsity1))\
                 + (1-mean_activation1)*(T.log(1-mean_activation1) - T.log(1-self.multi_sparsity1))

            mean_activation2 = T.mean(h2,axis=0)

            cost_part2 = 1/self.x.shape[1]*T.sum(self.multi_sparse_weight1 * KL1)

            KL2 = (1+mean_activation2)*(T.log(1+mean_activation2) -T.log(1+self.multi_sparsity2))\
                 + (1-mean_activation2)*(T.log(1-mean_activation2) - T.log(1-self.multi_sparsity2))

            cost_part3 = T.sum(self.multi_sparse_weight2 * KL2)

            cost = cost_part1 + cost_part2 + cost_part3
            #
            gparams = T.grad(cost_part1, self.params)
            # generate the list of updates
            updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ]


        else:

            cost_part1 = 0.5*T.sum((residue**2)/self.x.shape[0])\
                         +0.5*(T.sum((self.W1*self.G_decay1)**2)+T.sum((self.W2*self.G_decay2)**2))\
                         +0.5*(T.sum((self.W3*self.G_decay2.T)**2)+T.sum((self.W4*self.G_decay1.T)**2))

            mean_activation1 = T.mean(h1,axis=0)

            KL1 = (1+mean_activation1)*(T.log(1+mean_activation1) -T.log(1+self.multi_sparsity1))\
                 + (1-mean_activation1)*(T.log(1-mean_activation1) - T.log(1-self.multi_sparsity1))

            mean_activation2 = T.mean(h2,axis=0)

            cost_part2 = T.sum(self.multi_sparse_weight1 * KL1)

            KL2 = (1+mean_activation2)*(T.log(1+mean_activation2) -T.log(1+self.multi_sparsity2))\
                 + (1-mean_activation2)*(T.log(1-mean_activation2) - T.log(1-self.multi_sparsity2))

            cost_part3 = T.sum(self.multi_sparse_weight2 * KL2)

            cost = cost_part1 + cost_part2 + cost_part3

            gparams = T.grad(cost_part1, self.params)
            # generate the list of updates
            updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ]

        return (cost, updates)

    def get_cost(self):

        h1 = self.get_h1()
        h2 = self.get_h2(h1)
        h3 = self.get_h3(h2)
        z = self.get_reconstructed(h3)

        residue = z - self.x

        if self.missing:

            cost_part1 = 0.5*T.sum(((self.indi_matrix*residue)**2)/self.norm_indi)

        else:

            cost_part1 = 0.5*T.sum((residue**2)/self.x.shape[0])

        return cost_part1



def test_SAE(data = '',validationdata='',param_list= [], missingrate1 = 0.2,n_hidden1 = 288,n_hidden2 = 100,
             missing1= True, missing2 = False,share1 = False,share2 = True,
              missingrate2 = 0,learningrate = 0.1, training_epochs = 1200,batch_size = 30,
             output_folder = 'two_mod_sae_hidden'):


    output_folder_sae = output_folder + '_' + str(n_hidden2) + '_3_' +  str(missingrate1)
    print(missingrate1)

    newpath = '../Result/'+ output_folder_sae
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    output_folder1= output_folder + '_' + str(n_hidden2) +  '_1'
    output_folder2=output_folder + '_' + str(n_hidden2) +  '_2'
    ae1, h1, h_valid, indi_matrix_test,indi_matrix_valid_test,indi_matrix_final_test = \
        build_block(data,validationdata, param_list,n_hidden = n_hidden1,share= share1,missing = missing1,
            missing_rate = missingrate1,learning_rate=learningrate, training_epochs= training_epochs,
            batch_size=batch_size, output_folder=output_folder1,order = 1)
        # build_block(data,validationdata, param_list,n_hidden1,share1,missing1, missingrate1,
        #               learningrate, training_epochs,
        #               batch_size, output_folder='m_HI_1',order = 1)

    print('Fininshed training the first auto encoder')


    ae2,h2,h2_valid,indi_matrix_test2,indi_matrix_valid_test2 = \
        build_block(h1, h_valid, param_list,n_hidden2,share2,missing = missing2, missing_rate = missingrate2,
                      learning_rate=learningrate, training_epochs= training_epochs,
                      batch_size = 30, output_folder=output_folder2,order = 2)

    print('Fininshed training the second auto encoder')


    datasets,indi_matrix,data_test,n_train_batches,numMod,raw,trainstats_list,visible_size_Mod =\
        load_data_sae(param_list,data,indi_matrix_test,batch_size,train = True)


    valid_batch_size = 306
    validset,valid_indi_matrix, n_valid_batches =\
        load_data_sae(param_list,validationdata,indi_matrix_valid_test,valid_batch_size,train = False)


    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')
    y = T.matrix('y')
    # end-snippet-2


    sae = SAE(
        ae1=ae1,
        ae2=ae2,
        input = x,
        indi_matrix = y,
        missing = missing1,
        param_list = param_list,
    )

    # datasets = np.load(data)
    #
    # indi_matrix = [0]*raw.shape[1]
    # for i in range(raw.shape[1]):
    #     indi_matrix[i] = np.random.binomial(1, 1-missingrate1,(raw.shape[0],1))
    # indi_matrix = np.concatenate(indi_matrix,axis = 1)
    # datasets = theano.shared(np.asarray(datasets,dtype=theano.config.floatX),name='datasets', borrow=True)
    # indi_matrix = theano.shared(np.asarray(indi_matrix,dtype = theano.config.floatX ),name='indi_matrix', borrow=True)

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
            #print(a)
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
                    # print('Iteration %i' % (iter * patience_increase))
                    # print('Patience %i' % patience)
                    # print('saving the model for epoch %i' % epoch)
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

    for i in range(best_epoch -1):
        path = newpath + '/best_model_epoch_' +str(i+1) +'.pkl'
        if os.path.exists(path):
            os.remove(path)

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
                                                       trainstats_list[i]),indi_matrix_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],missingrate1)))

        f.write('AE error ratio for Modality'+ '\t' + str(i) +'\t'+ str(error_ratio(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                          trainstats_list[i]),indi_matrix_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],missingrate1)) + '\n')



    ###############
    # Test MODEL #
    ###############
    print('...We testing the performance of the finetuning model here...')
    raw_test,data_real_test,teststats_list = \
        load_sae_test_data(data,numMod,missingrate1,indi_matrix_final_test)

    h11 = get_h1(missing1,data_real_test,W1,b1,G)
    h21 = get_h2(h11,W2,b2,G_share)
    h31 = get_h3(h21,W3,b3,G_share)
    reconstruction_test = get_reconstruct(missing1,h31,W4,b4,G)

    print(reconstruction_test)

    np.savetxt(newpath+ '/output_test_' +str(best_epoch) + '.txt',reconstruction_test,delimiter=',')


    f = open(newpath +'/AE_test_' +str(best_epoch) + '.txt', 'w')


    for i in range(int(numMod)):
        np.savetxt(newpath +'/Raw_test_' +str(i) + '.txt',
        raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],delimiter=',')
        np.savetxt(newpath+'/Recstru_test_' +str(i) + '_' +str(best_epoch) + '.txt',
        denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],teststats_list[i]),delimiter=',')



        print(f, 'AE RMSE for Modality', i, str(RMSE(raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                       teststats_list[i]))))


        f.write('AE RMSE for Modality'+ '\t' + str(i) +'\t'+ str(RMSE(raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                          teststats_list[i]))) + '\n')


        print(f, 'AE error ratio for Modality', i, str(error_ratio(raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                       teststats_list[i]),indi_matrix_final_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],missingrate1)))

        f.write('AE error ratio for Modality'+ '\t' + str(i) +'\t'+ str(error_ratio(raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                          teststats_list[i]),indi_matrix_final_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],missingrate1)) + '\n')


if __name__ == '__main__':
    ###param_list consists : decay_weight(lambda),sparsity(rho) and sparse_weight(beta)
    param_list = [[1e-5,0.01,1e-5],[1e-5,0.01,1e-5]]
    dataset = '../Data/train_HumTempDF_new.npy'
    validationset = '../Data/test_HumTempDF_new.npy'

    dataset1 = '../Data/train_HumIllumDF_new.npy'
    validationset1 = '../Data/test_HumIllumDF_new.npy'

    n_hidden1 = 288

    hidden_list = [288]
    for j in range(len(hidden_list)):
        n_hidden2 = hidden_list[j]
        a = [0,.1,.2,.3,.4]
        for i in range(len(a)):
            missingrate = a[i]
            print(missingrate)
            test_SAE(data=dataset,validationdata = validationset,param_list=param_list,missingrate1=missingrate,
                     n_hidden1 = n_hidden1,n_hidden2=n_hidden2,missing1= True, missing2 = False,share1 = False,share2 = True,
                  missingrate2 = 0,learningrate = 0.1, training_epochs = 1200,batch_size = 30,
                 output_folder = 'two_mod_sae_hidden_HT')

        b = [.5,.6,.7]
        for i in range(len(b)):
            missingrate = b[i]
            print(missingrate)
            test_SAE(data=dataset,validationdata = validationset,param_list=param_list,missingrate1=missingrate,
                     n_hidden1 = n_hidden1,n_hidden2=n_hidden2,missing1= True, missing2 = False,share1 = False,share2 = True,
                  missingrate2 = 0,learningrate = 0.1, training_epochs = 1200,batch_size = 120,
                 output_folder = 'two_mod_sae_hidden_HT')

        c = [.8,.9]
        for i in range(len(c)):
            missingrate = c[i]
            print(missingrate)
            test_SAE(data=dataset,validationdata = validationset,param_list=param_list,missingrate1=missingrate,
                     n_hidden1 = n_hidden1,n_hidden2=n_hidden2,missing1= True, missing2 = False,share1 = False,share2 = True,
                  missingrate2 = 0,learningrate = 0.1, training_epochs = 1200,batch_size = 300,
                 output_folder = 'two_mod_sae_hidden_HT')



        d = [0,.1,.2,.3,.4]
        for i in range(len(d)):
            missingrate = d[i]
            print(missingrate)
            test_SAE(data=dataset1,validationdata = validationset1,param_list=param_list,missingrate1=missingrate,
                     n_hidden1 = n_hidden1,n_hidden2=n_hidden2,missing1= True, missing2 = False,share1 = False,share2 = True,
                  missingrate2 = 0,learningrate = 0.1, training_epochs = 1200,batch_size = 30,
                 output_folder = 'two_mod_sae_hidden_HI')

        e = [.5,.6,.7]
        for i in range(len(e)):
            missingrate = e[i]
            print(missingrate)
            test_SAE(data=dataset1,validationdata = validationset1,param_list=param_list,missingrate1=missingrate,
                     n_hidden1 = n_hidden1,n_hidden2=n_hidden2,missing1= True, missing2 = False,share1 = False,share2 = True,
                  missingrate2 = 0,learningrate = 0.1, training_epochs = 1200,batch_size = 120,
                 output_folder = 'two_mod_sae_hidden_HI')

        f = [.8,.9]
        for i in range(len(f)):
            missingrate = f[i]
            print(missingrate)
            test_SAE(data=dataset1,validationdata = validationset1,param_list=param_list,missingrate1=missingrate,
                     n_hidden1 = n_hidden1,n_hidden2=n_hidden2,missing1= True, missing2 = False,share1 = False,share2 = True,
                  missingrate2 = 0,learningrate = 0.1, training_epochs = 1200,batch_size = 300,
                 output_folder = 'two_mod_sae_hidden_HI')


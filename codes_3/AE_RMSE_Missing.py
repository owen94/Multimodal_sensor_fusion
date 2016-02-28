
import os
import sys
import timeit
import utils
import numpy
import pickle



import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from normLib import normActiv,denormActiv,sigmoid,tanh
from utils_copy import ravel_param, unravel_param,save,load,RMSE,error_ratio,load_data,load_test_data
from load_models import load_numerical_params_ae

try:
    import PIL.Image as Image
except ImportError:
    import Image

theano.config.optimizer= 'fast_compile'
class AE(object):
    """Auto-Encoder class (AE)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input = None,
        indi_matrix = None,
        bias_matrix = None,
        n_visible = 0,
        n_hidden = 0,
        W1 = None,
        W2 = None,
        bhid=None,
        bvis=None,
        missing = False,
        param_list = None,
        share = False
    ):

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.missing = missing

        self.param_list = param_list

        self.indi_matrix = indi_matrix

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        norm_indi = T.sum(self.indi_matrix, axis = 0)
        self.norm_indi = norm_indi
        self.share = share
        self.numMod = len(param_list)
        print('numMod is : ')
        print(self.numMod )

        # create a Theano random generator that gives symbolic random values

        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input


        if not bias_matrix:
            initial_bias_matrix = numpy.asarray(
                numpy.zeros((n_visible, n_hidden)),
                dtype=theano.config.floatX
            )
            bias_matrix = theano.shared(value=initial_bias_matrix, name='bias_matrix', borrow=True)

        if not W1:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU

            random_state = numpy.random.RandomState(None)
            r = numpy.sqrt(6)/numpy.sqrt(n_hidden + n_visible+1)
            initial_W1 = random_state.rand(n_visible,n_hidden)*2*r-r
            initial_W1 = numpy.asarray(initial_W1, dtype=theano.config.floatX)

            # initial_W1 = numpy.asarray(
            #     numpy_rng.uniform(
            #         low=- numpy.sqrt(6. / (n_hidden + n_visible)),
            #         high= numpy.sqrt(6. / (n_hidden + n_visible)),
            #         size=(n_visible, n_hidden)
            #     ),
            #     dtype=theano.config.floatX
            # )
            print('Initializing w1')
            W1 = theano.shared(numpy.asarray(initial_W1,dtype=theano.config.floatX),name='W1', borrow=True)


        if not W2:
            random_state = numpy.random.RandomState(None)
            r = numpy.sqrt(6)/numpy.sqrt(n_hidden + n_visible+1)
            initial_W2 = random_state.rand(n_hidden,n_visible)*2*r-r
            initial_W2 = numpy.asarray(initial_W2, dtype=theano.config.floatX)

            # initial_W2 = numpy.asarray(
            #     numpy_rng.uniform(
            #         low=-numpy.sqrt(6. / (n_hidden + n_visible)),
            #         high=numpy.sqrt(6. / (n_hidden + n_visible)),
            #         size=(n_hidden,n_visible)
            #     ),
            #     dtype=theano.config.floatX
            # )

            W2 = theano.shared(value=initial_W2, name='W2', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W1 = W1
        self.bias_matrix = bias_matrix
        # b corresponds to the bias of the hidden
        self.b1 = bhid
        # b_prime corresponds to the bias of the visible
        self.b2 = bvis
        self.W2 = W2
        print(type(self.W1))

        #self.params_missing = []
        #self.params = []
        #self.params_missing = [self.bias_matrix,self.W1, self.W2, self.b1, self.b2]
        self.params_missing = [self.W1, self.W2, self.b1, self.b2]
        self.params = [self.W1, self.W2, self.b1, self.b2]

        if self.numMod > 1:
            W_single_size = self.n_visible/self.numMod
            b_single_size = self.n_hidden/self.numMod
            a = numpy.ones((W_single_size,b_single_size))
            if self.share:
                b = numpy.ones((W_single_size,b_single_size))
            else:
                b = numpy.zeros((W_single_size,b_single_size))
            c = numpy.ones((1,b_single_size))
            ## we initialize the indicator matrix G here, which indicate whether there
            ## is an edge from input node i to hidden node j.
            ## if there is only one modality, we just initialize G as all ones,
            ## and it makes no change of the original auto-encoder.
            ## should G be initialized as shared variable????
            G1 = numpy.concatenate((a,b),axis = 1)
            G2 = numpy.concatenate((b,a),axis = 1)
            G = numpy.concatenate((G1,G2),axis = 0)

            G11 = self.param_list[0][0]*numpy.concatenate((a,b),axis = 1)
            G22 = self.param_list[1][0]*numpy.concatenate((b,a),axis = 1)
            G_decay = numpy.concatenate((G11,G22),axis = 0)

            multi_sparse_rate1 = self.param_list[0][1] * c
            multi_sparse_rate2 = self.param_list[1][1] * c
            multi_sparse_weight1 = self.param_list[0][2] * c
            multi_sparse_weight2 = self.param_list[1][2] * c
            multi_sparsity = numpy.concatenate((multi_sparse_rate1,multi_sparse_rate2),axis = 1)
            multi_sparse_weight = numpy.concatenate((multi_sparse_weight1,multi_sparse_weight2),axis = 1)

        else:
            G = numpy.ones((self.n_visible,self.n_hidden))
            G_decay = self.param_list[0][0]*numpy.ones((self.n_visible,self.n_hidden))
            multi_sparsity = self.param_list[0][1]*numpy.ones((1,self.n_hidden))
            multi_sparse_weight = self.param_list[0][2]*numpy.ones((1,self.n_hidden))

        # print(G.shape)
        if self.missing:
            multi_sparsity = numpy.repeat(multi_sparsity,n_visible,axis = 0)
            multi_sparse_weight = numpy.repeat(multi_sparse_weight,n_visible,axis = 0)

        # print(G)
        # print(multi_sparsity)
        # print(multi_sparse_weight)

        G = theano.shared(numpy.asarray(G,dtype=theano.config.floatX),name='G', borrow=True)
        G_decay = theano.shared(numpy.asarray(G_decay,dtype=theano.config.floatX),name='G_decay', borrow=True)
        multi_sparsity = theano.shared(numpy.asarray(multi_sparsity,dtype=theano.config.floatX),name='sparsity', borrow=True)
        multi_sparse_weight = theano.shared(numpy.asarray(multi_sparse_weight,dtype=theano.config.floatX),name='sparse_weight', borrow=True)

        self.G = G
        self.G_decay = G_decay
        self.multi_sparsity = multi_sparsity
        self.multi_sparse_weight = multi_sparse_weight

    def get_hidden_values(self):

        if self.missing:
            # bias_matrix = T.dot(self.indi_matrix,self.bias_matrix)
            # hidden_values = T.tanh(T.dot(self.x, self.W1*self.G) + bias_matrix + self.b1)
            hidden_values = T.tanh(T.dot(self.x, self.W1*self.G) + self.b1)

        else:
            hidden_values = T.tanh(T.dot(self.x, self.W1*self.G) + self.b1)
        return hidden_values


    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        if self.missing:
            return T.dot(hidden, self.W2*self.G.T) + self.b2
            #return T.tanh(T.dot(hidden, self.W2*self.G.T) + self.b2)
        else:
            #return T.dot(hidden, self.W2*self.G.T) + self.b2
            return T.tanh(T.dot(hidden, self.W2*self.G.T) + self.b2)

    def get_cost_updates(self,learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        y = self.get_hidden_values()

        z = self.get_reconstructed_input(y)

        residue = self.x - z*self.indi_matrix

        if self.missing:
            print('...This is the missing case...')

            cost_part1 = 0.5*T.sum(((self.indi_matrix*residue)**2)/self.norm_indi)\
                         +0.5*(T.sum((self.W1*self.G_decay)**2)+T.sum((self.W2*self.G_decay.T)**2))

            mean_activation = T.dot(self.indi_matrix.T, y)
            mean_activation = mean_activation/self.norm_indi


            KL = (1+mean_activation)*(T.log(1+mean_activation) -T.log(1+self.multi_sparsity))\
                 + (1-mean_activation)*(T.log(1-mean_activation) - T.log(1-self.multi_sparsity))

            cost_part2 = 1/self.x.shape[1]*T.sum(self.multi_sparse_weight * KL)

            cost = cost_part1 + cost_part2

            gparams_missing = T.grad(cost, self.params_missing)
            # generate the list of updates
            updates = [
            (params_missing, params_missing - learning_rate * gparams_missing)
            for params_missing, gparams_missing in zip(self.params_missing, gparams_missing)
            ]

        else:
            cost_part1 = 0.5*T.sum((residue**2)/self.x.shape[0])\
                         +0.5*(T.sum((self.W1*self.G_decay)**2)+T.sum((self.W2*self.G_decay.T)**2))

            mean_activation = T.mean(y,axis=0)

            KL = (1+mean_activation)*(T.log(1+mean_activation) -T.log(1+self.multi_sparsity))\
                 + (1-mean_activation)*(T.log(1-mean_activation) - T.log(1-self.multi_sparsity))

            cost_part2 = T.sum(self.multi_sparse_weight * KL)
            cost = cost_part1 + cost_part2
            gparams = T.grad(cost, self.params)
            # generate the list of updates
            updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ]

        return (cost, updates)

    def get_cost(self,get_reconstruction = False):

        y = self.get_hidden_values()

        z = self.get_reconstructed_input(y)


        if get_reconstruction:
            return z

        else:
            residue = self.x - z*self.indi_matrix

            if self.missing:

                cost_part1 = T.sum(((self.indi_matrix*residue)**2)/self.norm_indi)

            else:
                cost_part1 = T.sum((residue**2)/self.x.shape[0])

            return cost_part1


def test_AE(data='',validationdata= '', param_list= None,n_hidden = 288,share= False,missing = True,
            missing_rate = 0.2,learning_rate=0.08, training_epochs= 10,
            batch_size= 100, output_folder='dA_plots',order = 1):


    newpath = '../Result/'+ output_folder + '_' + str(missing_rate)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    ###################################
    # Initializing training dataset    #
    ####################################
    datasets,indi_matrix,data_test,indi_matrix_test,n_train_batches,numMod,raw,trainstats_list,visible_size_Mod = \
        load_data(param_list,data,batch_size,missing_rate,train = True,order = order)

    ####################################
    # Initializing validation dataset  #
    ####################################
    valid_batch_size = 306
    validationset,indi_matrix_validation,valid_test,indi_matrix_valid_test,n_valid_batches = \
        load_data(param_list,validationdata,valid_batch_size, missing_rate,train = False,order = order)


    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')
    y = T.matrix('y')# the data is presented as rasterized images
    # end-snippet-2

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))


    # indi_matrix = theano.shared(numpy.asarray(indi_matrix,dtype=theano.config.floatX),name='indi_matrix', borrow=True)
    #
    # datasets = datasets * indi_matrix

    ae = AE(
        numpy_rng = rng,
        theano_rng= theano_rng,
        input=x,
        indi_matrix = y,
        bias_matrix = None,
        n_visible = raw.shape[1],
        n_hidden = n_hidden ,
        W1 = None,
        W2 = None,
        bhid=None,
        bvis=None,
        missing = missing,
        param_list = param_list,
        share = share
    )

    cost,updates = ae.get_cost_updates(learning_rate)

    train_ae = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: datasets[index * batch_size: (index + 1) * batch_size],
            y: indi_matrix[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='warn',
    )


    validate_cost = ae.get_cost()

    validate_ae = theano.function(
        [index],
        validate_cost,
        givens={
            x: validationset[index * valid_batch_size: (index + 1) * valid_batch_size],
            y: indi_matrix_validation[index * valid_batch_size: (index + 1) * valid_batch_size]
        },
        on_unused_input='warn',
    )



    ############
    # TRAINING #
    ############
    # go through training epochs

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
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

    best_validation_loss = numpy.inf
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    best_epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        c= []
        for minibatch_index in range(int(n_train_batches)):

            a = train_ae(minibatch_index)
            c.append(a)
            #print(a)
            # iteration number indicate how many batches we have already runned on
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_ae(i)
                                     for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)

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
                    save(newpath + '/best_model_epoch_' +str(epoch) +'.pkl', ae)

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



    ####################################
    # computing RMSE and error ratio #
    ####################################
    print('Best epoch is %i' % best_epoch )
    print('Now we starting computing the RMSE and error ratio')

    for i in range(best_epoch -1):
        path = newpath + '/best_model_epoch_' +str(i+1) +'.pkl'
        if os.path.exists(path):
            os.remove(path)

    ae = newpath + '/best_model_epoch_' +str(best_epoch) +'.pkl'

    W1,W2,b1,b2,G = load_numerical_params_ae(ae)


    bias_matrix = None


    y = get_hidden_values(data_test,indi_matrix_test,W1,b1,G,bias_matrix,missing)

    y2 = get_hidden_values(valid_test,indi_matrix_valid_test,W1,b1,G,bias_matrix,missing)

    h1 = newpath + '/h1.npy'

    h_valid = newpath + '/h_valid.npy'

    numpy.save(h1,y)
    numpy.save(h_valid,y2)


    reconstruction = get_reconstructed_input(y,W2,b2,G,missing)

    print(reconstruction)

    numpy.savetxt(newpath+ '/output_' +str(best_epoch) + '.txt',reconstruction,delimiter=',')


    f = open(newpath +'/AE_' +str(best_epoch) + '.txt', 'w')

    if order == 1:

        for i in range(int(numMod)):
            numpy.savetxt(newpath +'/Raw_' +str(i) + '.txt',
            raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],delimiter=',')
            numpy.savetxt(newpath+'/Recstru_' +str(i) + '_' +str(best_epoch) + '.txt',
            denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],trainstats_list[i]),delimiter=',')



            print(f, 'AE RMSE for Modality', i, str(RMSE(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                           trainstats_list[i]))))


            f.write('AE RMSE for Modality'+ '\t' + str(i) +'\t'+ str(RMSE(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                              trainstats_list[i]))) + '\n')


            print(f, 'AE error ratio for Modality', i, str(error_ratio(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                       trainstats_list[i]),indi_matrix_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],missing_rate)))

            f.write('AE error ratio for Modality'+ '\t' + str(i) +'\t'+ str(error_ratio(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                              denormActiv(reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                          trainstats_list[i]),indi_matrix_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],missing_rate)) + '\n')

            print('we are done here')
    else:

        for i in range(int(numMod)):
            numpy.savetxt(newpath +'/Raw_' +str(i) + '.txt',
            raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],delimiter=',')
            numpy.savetxt(newpath+'/Recstru_' +str(i) + '_' +str(best_epoch) + '.txt',
           reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod],delimiter=',')



            print(f, 'AE RMSE for Modality', i, str(RMSE(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod])))


            f.write('AE RMSE for Modality'+ '\t' + str(i) +'\t'+ str(RMSE(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod])) + '\n')


            print(f, 'AE error ratio for Modality', i, str(error_ratio(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod]
                                                           )))

            f.write('AE error ratio for Modality'+ '\t' + str(i) +'\t'+ str(error_ratio(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  reconstruction[:,i*visible_size_Mod:(i+1)*visible_size_Mod])) + '\n')

    if order == 1:

        print('...This is the missing case, we predict the reconstruction error ratio for the test data...')

        raw_test,data_real_test,teststats_list,indi_matrix_final_test = load_test_data(data,numMod,missing_rate)

        y1 = get_hidden_values(data_real_test,indi_matrix_final_test,W1,b1,G,bias_matrix,missing)

        reconstruction_test = get_reconstructed_input(y1,W2,b2,G,missing)

        print(reconstruction_test)

        numpy.savetxt(newpath+ '/output_test_' +str(best_epoch) + '.txt',reconstruction_test,delimiter=',')

        f = open(newpath +'/AE_test_' +str(best_epoch) + '.txt', 'w')

        for i in range(int(numMod)):
            numpy.savetxt(newpath +'/Raw_test_' +str(i) + '.txt',
            raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],delimiter=',')
            numpy.savetxt(newpath+'/Recstru_test_' +str(i) + '_' +str(best_epoch) + '.txt',
            denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],teststats_list[i]),delimiter=',')


            print(f, 'AE RMSE for Modality', i, str(RMSE(raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                           teststats_list[i]))))


            f.write('AE RMSE for Modality'+ '\t' + str(i) +'\t'+ str(RMSE(raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                              teststats_list[i]))) + '\n')


            print(f, 'AE error ratio for Modality', i, str(error_ratio(raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                           teststats_list[i]),indi_matrix_final_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],missing_rate)))

            f.write('AE error ratio for Modality'+ '\t' + str(i) +'\t'+ str(error_ratio(raw_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                  denormActiv(reconstruction_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],
                                                              teststats_list[i]),indi_matrix_final_test[:,i*visible_size_Mod:(i+1)*visible_size_Mod],missing_rate)) + '\n')

        return  ae, h1, h_valid, indi_matrix_test,indi_matrix_valid_test,indi_matrix_final_test

    if order == 2:

        return ae, h1, h_valid, indi_matrix_test,indi_matrix_valid_test





def get_hidden_values(x,indi_matrix,W1,b1,G,bias_matrix = None,missing = False):

    if missing:
        # bias_matrix = numpy.dot(indi_matrix,bias_matrix)
        # hidden_values = tanh(numpy.dot(x, W1*G) + bias_matrix + b1)
        hidden_values = numpy.tanh(numpy.dot(x, W1*G) + b1)
    else:
        hidden_values = numpy.tanh(numpy.dot(x, W1*G) + b1)
    return hidden_values


def get_reconstructed_input(hidden,W2,b2,G,missing = False):

    if missing:
        return numpy.dot(hidden, W2*G.T) + b2
        #return numpy.tanh(numpy.dot(hidden, W2*G.T) + b2)
    else:
        #return numpy.dot(x, W1*G) + bias_matrix + b1
        return numpy.tanh(numpy.dot(hidden, W2*G.T) + b2)

if __name__ == '__main__':
    ###param_list consists : decay_weight(lambda),sparsity(rho) and sparse_weight(beta)
    param_list = [[1e-5,0.01,1e-5]]
    dataset = '../Data/train_tempDF_new.npy'
    validationset = '../Data/test_tempDF_new.npy'

    a = [.1]
    for i in range(len(a)):
        missingrate = a[i]
        test_AE(dataset,validationset, param_list,n_hidden = 144,share= False,missing = True,
            missing_rate = missingrate,learning_rate=0.1, training_epochs= 1000,
            batch_size= 300, output_folder='single_mod_ae',order = 1)

""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


import numpy
import pickle
import theano
import theano.tensor as T
from normLib import normActiv,denormActiv,sigmoid,tanh


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

def ravel_param(W1,W2,b1,b2,bias_matrix = None,missing = False):

    if missing:
        return numpy.concatenate((W1.ravel(),W2.ravel(),b1.ravel(),b2.ravel(),bias_matrix.ravel()))

    else:
        return numpy.concatenate((W1.ravel(),W2.ravel(),b1.ravel(),b2.ravel()))



def unravel_param(theta,hiddenSize,visibleSize,missing = False):

    # Convert theta to the (W1, W2, b1, b2) matrix/vector format
    W1 = theta[0:hiddenSize*visibleSize].reshape(visibleSize,hiddenSize)
    W2 = theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize].reshape(hiddenSize,visibleSize)
    b1 = theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize]
    b2 = theta[2*hiddenSize*visibleSize+hiddenSize:2*hiddenSize*visibleSize+2*hiddenSize]
    if missing:
        bias_matrix = theta[2*hiddenSize*visibleSize+hiddenSize:].reshape(visibleSize, hiddenSize)
        return ( W1,W2,b1,b2,bias_matrix)

    else:
        return ( W1,W2,b1,b2)

def save(filename,bob):
    pickle_file = open(filename,'wb')
    pickle.dump(bob, pickle_file)
    pickle_file.close()

def load(filename):
    pickle_file = open(filename,'rb')
    bob = pickle.load(pickle_file)
    pickle_file.close()
    return bob

def RMSE(data1,data2,rmsemodnum=1):
    if rmsemodnum == 1:
      return numpy.sqrt(numpy.mean((data2-data1)**2))
    else:
      onemod = data1.shape[1]/rmsemodnum
      data1_list = []
      data2_list = []
      for i in range(rmsemodnum):
          data1_list.append(data1[:,i*onemod:(i*onemod+onemod)])
          data2_list.append(data2[:,i*onemod:(i*onemod+onemod)])
      rmse = []
      for i in range(rmsemodnum):
          rmse.append(numpy.sqrt(numpy.mean((data2_list[i]-data1_list[i])**2)))
      return rmse

def error_ratio(data1, data2):
    a = numpy.sqrt((data1 - data2)**2)
    b = numpy.sqrt(data1**2)
    return numpy.sum(a)/numpy.sum(b)


def load_data(param_list,data,batch_size,missing_rate,train = True,order=1):

    print('Loading training data................')
    numMod = len(param_list)
    raw = numpy.load(data)
    visible_size = numpy.shape(raw)[1]
    visible_size_Mod = int(visible_size/numMod)
    train_list = [0]*numMod
    trainstats_list = [0]*numMod


    ##for the first autoencoder, we need normalization, however, for the second one, we did not need

    if order == 1:

        for i in range(numMod):
            train_list[i], trainstats_list[i] = normActiv(raw[:,i*visible_size_Mod:(i+1)*visible_size_Mod])
        datasets = numpy.concatenate(train_list, axis=1)

    else:
        datasets = raw

    print(datasets)

    indi_matrix = [0]*raw.shape[1]
    for i in range(raw.shape[1]):
        indi_matrix[i] = numpy.random.binomial(1, 1-missing_rate,(raw.shape[0],1))
    indi_matrix = numpy.concatenate(indi_matrix,axis = 1)

    datasets = datasets * indi_matrix
    data_test = datasets
    indi_matrix_test = indi_matrix

    print(indi_matrix)

    datasets = theano.shared(numpy.asarray(datasets,dtype=theano.config.floatX),name='datasets', borrow=True)
    indi_matrix = theano.shared(numpy.asarray(indi_matrix,dtype = theano.config.floatX ),name='indi_matrix', borrow=True)
    n_train_batches = datasets.get_value(borrow=True).shape[0] / batch_size

    if train:

        return datasets,indi_matrix,data_test,indi_matrix_test,n_train_batches,numMod,raw,trainstats_list,visible_size_Mod

    else:
        return datasets,indi_matrix,data_test,indi_matrix_test,n_train_batches


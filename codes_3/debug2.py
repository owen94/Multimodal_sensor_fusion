


import numpy as np

dataset = '../Data/test_humDF_new.npy'
validationset = '../Data/test_tempDF_new.npy'

a = np.load(dataset)
b = np.load(validationset)

c = np.concatenate((a,b),axis = 1)

print(c.shape)
np.save('../Data/test_HumTempDF_new.npy',c)
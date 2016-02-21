# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:17:52 2014

@author: OCC, ZWY

This library is currently a collection of all
normalization & denormalization methods.
The methods are grouped together by categories, 
i.e. e.g. denormSigmoid is placed right after
normSigmoid.

For ease of specifying a particular method,
normActiv and denormActiv accepts as arguments data
sphering options and activation function options.

This library is expected to be used by SAE and SSAE class.
"""

from __future__ import division
import numpy as np
from numpy import fmax, fmin, std    # normTruncate
from numpy import array, mean, multiply, matrix  # normLinear

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logit(x):
    return np.log(x/(1-x))
    
def tanh(x):
  return (1-np.exp(-2*x))/(1+np.exp(-2*x))
  
def tanhInv(x):
  return 0.5*np.log((1+x)/(1-x))

def softplus(x):
  return np.log(1+np.exp(x))
  
def softplusInv(x):
  return np.log(np.exp(x)-1)


def normActiv(data,dataStatistics=dict({}),activation='tanh',norm='sphere',by='row'): # by = 'row','col','all' for norm='linear'
    activationDict = dict({'sigmoid':sigmoid,'tanh':tanh,'softplus':softplus})
    normDict = dict({'linear':normLinear,'sphere':normSphere,'trunc':normTruncate})
    activationOpt = activationDict[activation]    
    normOpt = normDict[norm]    
    if len(dataStatistics)==0:
        data,dataStatistics = normOpt(data,by=by)
        data = activationOpt(data)
        return data,dataStatistics
    else:
        data = activationOpt(normOpt(data,dataStatistics=dataStatistics,by=by))
        return data     
  
def denormActiv(data,dataStatistics,activation='tanh',norm='sphere'):
    activationDict = dict({'sigmoid':logit,'tanh':tanhInv,'softplus':softplusInv})
    normDict = dict({'linear':denormLinear,'sphere':denormSphere,'trunc':denormTruncate})
    activationInvOpt = activationDict[activation]    
    denormOpt = normDict[norm]
    return denormOpt(activationInvOpt(data),dataStatistics)    

def normTruncate(data,dataStatistics=dict({}),by=None):
    truncMean = mean(data,axis=0)
    data = data-truncMean
    if len(dataStatistics)==0:
        truncDev = 3*std(data)
    else:
        truncDev = dataStatistics['truncDev']
    data = fmax(fmin(data,truncDev),-truncDev)/truncDev    
    data = (data+1)*0.4+0.1
    if len(dataStatistics)==0:
        dataStatistics = dict({'truncMean':truncMean,'truncDev':truncDev})
        return data,dataStatistics
    else:
        return data

def denormTruncate(data,dataStatistics):        
    truncMean = dataStatistics['truncMean']
    truncDev = dataStatistics['truncDev']    
    return ((data-0.1)/0.4-1)*truncDev+truncMean


def normLinear(data,dataStatistics=dict({}),by='row'):
    if len(dataStatistics)==0:
      if by == 'row':
        dataMean = matrix(mean(data,axis=1)).T
      elif by == 'col':
        dataMean = mean(data,axis=0)
      else:
        dataMean = mean(data)
    else:
        dataMean = dataStatistics['dataMean']
    if len(dataStatistics)==0:
      if by == 'row':
        dataDev = matrix(std(data,axis=1)).T
      elif by == 'col':
        dataDev = std(data,axis=0)
      else:
        dataDev = std(data)
    else:
        dataDev = dataStatistics['dataDev']
    data = array((data-dataMean)/(dataDev*3))
    if len(dataStatistics)==0:
        dataStatistics = dict({'dataMean':dataMean,'dataDev':dataDev})
        return data,dataStatistics
    else:
        return data
        
def denormLinear(data,dataStatistics):
    dataMean = dataStatistics['dataMean']
    dataDev = dataStatistics['dataDev']
    return array(multiply(data,(dataDev*3))+dataMean)
    
      
def normSphere(data,dataStatistics=dict({}),by=None):
    if len(dataStatistics)==0:
        dataRowMean = matrix(mean(data,axis=1)).T
        dataRowDev =  matrix(std(data-dataRowMean,axis=1))
        newdata = array((data-dataRowMean)/(dataRowDev*3))
        newdataColMean = mean(newdata,axis=0)
        data = array(newdata-newdataColMean)
        dataDev = std(data)
        dataStatistics = dict({'dataRowMean':dataRowMean,'dataRowDev':dataRowDev,'newdataColMean':newdataColMean,'dataDev':dataDev})
        return data*0.17/dataDev,dataStatistics
    else:
        dataRowMean = dataStatistics['dataRowMean']
        dataRowDev = dataStatistics['dataRowDev']
        newdataColMean = dataStatistics['newdataColMean']
        dataDev = dataStatistics['dataDev']
        data = array((data-dataRowMean)/(dataRowDev*3)-newdataColMean)*0.17/dataDev
        return data
        
def denormSphere(data,dataStatistics):
    dataRowMean = dataStatistics['dataRowMean']
    dataRowDev = dataStatistics['dataRowDev']
    newdataColMean = dataStatistics['newdataColMean']
    dataDev = dataStatistics['dataDev']
    return array(multiply(data*dataDev/0.17+newdataColMean,(dataRowDev*3))+dataRowMean)
    
    
    
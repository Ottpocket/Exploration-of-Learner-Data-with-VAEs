# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 07:52:15 2020

@author: andre
"""
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow import keras as k
import pandas as pd


a_true = np.transpose(np.loadtxt('A.csv', delimiter = ','))
b_true = np.transpose(np.loadtxt('B.csv', delimiter = ','))
theta_true = np.loadtxt('Theta.csv', delimiter = ',')
qmat = np.loadtxt('Q_matrix.csv', delimiter = ',')















   



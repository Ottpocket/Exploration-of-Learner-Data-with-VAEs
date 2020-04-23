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


def AVRB(correct, guess, rm_zero = True):
    correct = np.reshape(correct, [-1])
    guess = np.reshape(guess, [-1])
    if rm_zero == True:
        correct = np.delete(correct, np.where(correct == 0))
        guess = np.delete(guess, np.where(guess == 0))   
    return np.mean(np.abs(correct - guess)) 

def RMSE(correct, guess, rm_zero = True):
    correct = np.reshape(correct, [-1])
    guess = np.reshape(guess, [-1])
    if rm_zero == True:
        correct = np.delete(correct, np.where(correct == 0))
        guess = np.delete(guess, np.where(guess == 0))
    return np.sqrt(np.mean((correct-guess)**2))

def Corr(correct, guess, rm_zero = True):    
    #correct = a_true[:,0]
    #guess  = A_list[epoch][:,0]
    correct = np.reshape(correct, [-1])
    guess = np.reshape(guess, [-1])
    if rm_zero == True:
        correct = np.delete(correct, np.where(correct == 0))
        guess = np.delete(guess, np.where(guess == 0))    
    return pearsonr(correct, guess)[0] 



###############################################################################
#add_vae_loss: adds the kl loss for the distribution.
#INPUTS: 
    #model: the model used
    #loc: the location parameter
    #scale: the scale parameter
    #dist: the distribution used
###############################################################################
def add_vae_loss(model, loc, scale, dist):

    #no penalty added for ae
    if dist is None:
        pass
    #Normal with cov matrix of 0
    #loc = mu, scale = log_sigma
    elif dist == 'norm':
        kl_loss = 1 + 2 * scale - tf.math.square(loc) - tf.math.square(tf.exp(scale))
        kl_loss = -0.5 * tf.reduce_sum(kl_loss)
        model.add_loss(kl_loss)
    #gamma dist
    elif dist == 'gamma':
        #loc:= log_alpha
        #scale:= log_beta
        A = tf.ones_like(loc) #location parameter of prior
        B = tf.ones_like(loc)
        alpha = tf.math.exp(loc)
        kl_loss = -1 * alpha * scale + tf.math.lgamma(alpha) + (A - alpha) * (tf.math.digamma(alpha) - scale)
        kl_loss = kl_loss - alpha * (B / scale - 1)
        model.add_loss(-1 * kl_loss)
    #Multivariate Normal with non-I covariance matrix
    elif dist == 'mvn':
        #formula taken from https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        k_ = scale.shape[1]
        inv_cor = tf.linalg.inv(scale)
        mu = k.layers.Reshape((k_,1))(loc)
        kl_loss = tf.linalg.trace(inv_cor) + tf.reduce_sum(tf.linalg.matmul(tf.linalg.matmul(mu,inv_cor, transpose_a = True), mu), axis = [1,2])
        kl_loss = kl_loss - k_ #+ tf.linalg.logdet(scale) #DONE FOR DEBUGGING ONLY
        kl_loss = -.5 * tf.reduce_sum(kl_loss)
        model.add_loss(kl_loss)
    #beta dist
    elif dist == 'beta':
        pass

    elif dist == 'laplace':
        kl_loss = loc + tf.math.log(2.) +tf.math.exp(-1.* loc / tf.math.exp(scale))* tf.math.exp(scale)
        kl_loss = tf.reduce_sum(kl_loss)
        model.add_loss(kl_loss)

###############################################################################
#create_hidden_layer: creates the hidden layer of the VAE / AE
#INPUTS:
    #prev_layer: the previous layer in the network
    #dist: the distribution of the VAE/AE.  Can be 'norm', None, 
###############################################################################
def create_hidden_layer(prev_layer, dist):
    if dist is None:
        theta = k.layers.Dense(3, name = 'theta')(prev_layer)
        return k.Model(inputs=prev_layer, outputs = theta)
    elif dist == 'norm':
        mu = k.layers.Dense(3, name = 'mu')(prev_layer)
        log_sigma = k.layers.Dense(3, name = 'log_sigma')(prev_layer)
        epsilon =  tf.random.normal(tf.shape(log_sigma), dtype = tf.float32, mean = 0,
                                   stddev=1.0, name = 'epsilon')
        theta = mu + tf.exp(log_sigma) * epsilon
        return k.Model(inputs=prev_layer, outputs = [theta,mu,log_sigma])





###############################################################################
#get_stats_over_time: gets AVRB, RMSE, and CORR from a_1, a_2, a_3, and b for each
# epoch of training
#input:
#   A_list: list of (28x3) a matrices for each epoch of training
#   B_list: list of (28x1) b vectors '''''''
#   a_true: true values of 'a' from Data_Gen.py
#   b_true: true vales of 'b'  '''''''
#   matrix: do you want individual a rows or the whole thing as a?  default False for backwards compat    
#Output:
#   a_1: dataframe where rows are the epoch number and the cols are epoch, AVRB, RMSE, and Corr
#   a_2,a_3,b: dataframes of the same type
###############################################################################
def get_stats_over_time(A_list, B_list, a_true, b_true, qmat, matrix = False):
    b = []
    num_questions = A_list[0].shape[1]
    if matrix == False:
        a_1 = []
        a_2 = []
        a_3 = []
    else:
        a = []
    col_names = ['Epoch', 'AVRB', 'RMSE', 'Corr']
    a_true = a_true * qmat
    for epoch in range(len(A_list)):
        A_list[epoch] = qmat * A_list[epoch]
        b_row =   [epoch, AVRB(b_true, tf.reshape( B_list[epoch], [num_questions]), False), RMSE(b_true, tf.reshape( B_list[epoch], [num_questions]), False), 
                       Corr(b_true, tf.reshape( B_list[epoch], [num_questions]), False)]
        b.append(b_row)
        if matrix ==False:
            a_1_row = [epoch, AVRB(a_true[:,0], A_list[epoch][:,0]), RMSE(a_true[:,0], A_list[epoch][:,0]), Corr(a_true[:,0], A_list[epoch][:,0])]
            a_2_row = [epoch, AVRB(a_true[:,1], A_list[epoch][:,1]), RMSE(a_true[:,1], A_list[epoch][:,1]), Corr(a_true[:,1], A_list[epoch][:,1])]
            a_3_row = [epoch, AVRB(a_true[:,2], A_list[epoch][:,2]), RMSE(a_true[:,2], A_list[epoch][:,2]), Corr(a_true[:,2], A_list[epoch][:,2])]
            a_1.append(a_1_row)
            a_2.append(a_2_row)
            a_3.append(a_3_row)            
        else:
            a_row = [epoch, AVRB(a_true, A_list[epoch]), RMSE(a_true, A_list[epoch]), Corr(a_true, A_list[epoch])]           
            a.append(a_row)
    b_df = pd.DataFrame(data = b, columns = col_names)
    if matrix == False:
        a_1_df = pd.DataFrame(data = a_1, columns = col_names)
        a_2_df = pd.DataFrame(data = a_2, columns = col_names)
        a_3_df = pd.DataFrame(data = a_3, columns = col_names)
        return (a_1_df, a_2_df, a_3_df, b_df)
    else:
        a_df = pd.DataFrame(data = a, columns = col_names)
        return (a_df, b_df)

#Saving the Thetas 
#takes in train/val_student_thetas, and outputs cols 
#   [studentid, meanTheta1, meanTHeta2, meanTheta3]
def get_theta(list_of_theta_training_batches):
    student_thetas_mini_batches = []
    for i in list_of_theta_training_batches:
        student_thetas_mini_batches.append(np.hstack([ i[0][:,0:1], i[1]]))
    student_thetas = pd.DataFrame(np.vstack(student_thetas_mini_batches))
    return student_thetas.groupby(0).agg({1:'mean', 2:'mean', 3:'mean'}).sort_values(by=[0])

###############################################################################
#get_theta_stats_v2: gives AVRB, RMSE, and Corr between the correct thetas
# and the mean of the learned thetas from the network
#INPUT:
    #correct: (num_students, num_hidden_skills) the ground truth thetas
    #guess: (num_students*num_tests, num_hidden_skills) the guess from the nn
    #   of the students' hidden knowledge
    #studtest: (num_students*num_tests, 2) cols: student, test_num
#OUTPUT:
    #AVRB:
    #RMSE:
    #Corr:
###############################################################################    
def get_theta_stats_v2(correct, guess, studtest):
    cols = ['student', 'test']
    for i in range(guess.shape[1]):
        cols.append('skill{}'.format(i+1))
    guess_df = pd.DataFrame(data = np.hstack([studtest, guess]), columns = cols)
    guess_agg = guess_df.groupby('student').agg({'skill1':'mean', 'skill2':'mean', 'skill3':'mean'}).values
    return (AVRB(correct, guess_agg), RMSE(correct, guess_agg), Corr(correct, guess_agg))


###############################################################################
#Qmat_semi_sigmoid: recreates equation (4) from paper and includes the implicit
# Q-matrix.  We added b instead of subtracting it.  
#INPUTS:
    #Q_mat: the qmatrix
    #thetas: the hidden knowledge given by the encoder
#OUTPUTS:
    #logits: the outputs of equation (4) on the paper with b added not 
    #   subtracted. [None, 28]
###############################################################################
#help found at https://www.tensorflow.org/tutorials/customization/custom_layers
# and https://www.tensorflow.org/guide/keras/custom_layers_and_models
class Qmat_semi_sigmoid(k.layers.Layer):    
    def __init__(self):
        super(Qmat_semi_sigmoid, self).__init__()
        self.qmat = tf.Variable(initial_value = qmat, trainable = False, 
                                dtype = tf.float32, name = 'Qmat')
    
    def build(self, input_shape):
        self.log_A = self.add_weight(shape=(3,qmat.shape[1]),
                                 initializer='random_normal',
                                 trainable=True, name = 'log_A')
        self.B = self.add_weight(shape=(qmat.shape[1],),
                                 initializer='random_normal',
                                 trainable=True, name = 'B')

    def call(self, inputs):
        thetas = inputs
        return tf.math.sigmoid(tf.matmul(thetas, self.qmat * tf.exp(self.log_A)) - self.B)



###############################################################################
#stochastic_layer: function that creates the random variables of the vae
#INPUTS:
    #prev: the previous layer of the nn.
    #dist: the distribution used.  Can be 'gamma', 'mvn', 'norm', or None
    #num_hidden: the number of hidden variables
#OUTPUTS:
    #location: the location parameter of the distribution
    #scale: ' scale ' ''' ''''''. 
    #thetas: the students hidden knowledge
###############################################################################
def stochastic_layer(prev, dist, num_hidden = 3):
    assert prev.shape[0] is None
    assert prev.shape[1] == 10
    #for ae layer
    if dist == 'None':
        loc = tf.Variable([0], trainable = False)#dne
        scale = tf.Variable([0], trainable = False)#dne
        thetas = k.layers.Dense(num_hidden)(prev)
        return (loc, scale, thetas)
    
    #Normal with cov matrix of 0
    #loc = mu
    #scale = log_sigma
    elif dist == 'norm':
        loc = k.layers.Dense(num_hidden, name = 'loc')(prev) #mu
        scale = k.layers.Dense(num_hidden, name = 'scale')(prev) # log_sigma
        epsilon =  tf.stop_gradient(tf.random.normal(tf.shape(scale), dtype = tf.float32, mean = 0,
                                   stddev=1.0, name = 'epsilon'))
        thetas = loc + tf.exp(scale) * epsilon 
        return (loc, scale, thetas)
    
    #gamma dist, mean = loc / scale
    #from GREP paper, Ruiz, et al 2016
    #loc = log_alpha
    #scale = log_beta
    elif dist == 'gamma':
        loc = k.layers.Dense(num_hidden, name = 'loc')(prev) #log_alpha
        scale = k.layers.Dense(num_hidden, name = 'scale')(prev)#log_beta
        
        z = tf.random.gamma(shape = (), alpha = tf.math.exp(loc), beta = tf.math.exp(scale))
        dig_a = tf.math.digamma(tf.math.exp(loc))
        denom = tf.math.sqrt(tf.math.polygamma(tf.ones_like(loc),tf.math.exp(loc)))
        epsilon = tf.math.log(z) - dig_a + scale
        epsilon = epsilon / denom
        epsilon_stop = tf.stop_gradient(epsilon)
        thetas = denom * epsilon_stop + dig_a - scale
        return (loc, scale, tf.math.exp(thetas))
    #Multivariate Normal with non-I covariance matrix
    elif dist == 'mvn':
        #prev = hidden1
        #num_hidden = 3
        loc = k.layers.Dense(num_hidden, name = 'loc')(prev) #mu
        #very clunky.  
        scale = k.layers.Dense(num_hidden*num_hidden, name = 'scale_raw')(prev)#log of the square root of the cov matrix
        scale = tf.math.exp(k.layers.Reshape([num_hidden, num_hidden])(scale) )#
        scale = tf.linalg.LinearOperatorLowerTriangular(scale).to_dense()#lower triangular square root of cov mat
        diag_scaler = tf.constant([[.5,1,1],[1,.5,1],[1,1,.5]],
                                  dtype = tf.float32, name = 'diag_scaler') #maker the diagonal not 2x as much 
        scale = diag_scaler * (scale + tf.linalg.matrix_transpose(scale)) #sqrt of cov matrix
        epsilon = tf.stop_gradient(tf.random.normal([num_hidden,1], dtype = tf.float32, mean = 0,
                                   stddev=1.0, name = 'epsilon'))
        thetas = loc + k.layers.Reshape((3,))(tf.linalg.matmul(tf.math.exp(scale), epsilon))
        scale = tf.linalg.matmul(scale,scale, name = 'scale_true')# correct code
        #scale = k.layers.Dot([1,2], name = 'scale_true')([scale,scale]) #problematic for above line
        return(loc, scale, thetas)
    #Laplace with parameters mu,b
    elif dist == 'laplace':
        loc = k.layers.Dense(num_hidden, name = 'loc')(prev) #mu
        scale = k.layers.Dense(num_hidden, name = 'scale')(prev) # log_b
        exp_one= tf.random.gamma(tf.shape(scale), alpha = 1 , beta = 1)
        exp_two= tf.random.gamma(tf.shape(scale), alpha = 1 , beta = 1)
        laplace= tf.stop_gradient(exp_one - exp_two)
        thetas= loc + laplace * tf.math.exp(scale)
        return (loc, scale, thetas)
    #beta dist
    elif dist == 'beta':
        pass
    else:
        print('you did not enter a valid distribution!')

def xent(y_true, y_pred):
    X_hat_clipped = tf.compat.v1.clip_by_value(y_pred, 1e-10, 0.99999)
    encode_decode_loss_ = y_true * tf.math.log(1e-10 + X_hat_clipped) + (1 - y_true) * tf.math.log(1e-10 + 1 - X_hat_clipped) #Bernoulli is p(x|z)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss_, 1)
    return tf.reduce_sum(encode_decode_loss) 

   
###############################################################################
#Table_1: converts output of get_stats_over_time to Table1 of paper
#Input:
    #dfa1,...,dfb: output of get_stats_over_time function
    #dist: did this come from what distribution?
#Output:
    #df: Model_1 for either an ae or vae given the source of the data
###############################################################################
def Table_1(dfa1, dfa2, dfa3, dfb, dist, best_row = None):
    if dist is None:
        dist = 'AE'
        
    if best_row is None:
        last_row = len(dfa1.values) - 1
    else:
        last_row = best_row
    a1 = dfa1.values[last_row,1:]
    a2 = dfa2.values[last_row,1:]
    a3 = dfa3.values[last_row,1:]
    b  =  dfb.values[last_row,1:]
    df = pd.DataFrame(np.transpose(np.vstack([a1,a2,a3,b])), columns = ['a1','a2','a3','b'])
    df.insert(0, "Model", [dist]*3, True)
    df.insert(5,"Statistic", ['AVRB', 'RMSE', 'CORR'])
    return df


###############################################################################
#Table_2: creates table 2 from Paper
#INPUT: 
    #train_theta: the thetas produced from training.  not yet aggregated
    #val_theta: the thetas produced from validation.  not yet aggregated
    #vae: boolean.  True if vae, False o.w.
    #True_thetas: uses theta_true   
###############################################################################    
def Table_2(train_theta, val_theta, vae,True_thetas = theta_true):
    train_theta = get_theta(train_theta)
    val_theta = get_theta(val_theta)
    join_them_all_up = pd.concat([train_theta,val_theta]).sort_values(by=[0])
    #True_thetas = theta_true   
    est_thetas = join_them_all_up.values
    theta_1 = [AVRB(True_thetas[:,0], est_thetas[:,0]), RMSE(True_thetas[:,0], est_thetas[:,0]), Corr(True_thetas[:,0], est_thetas[:,0])]
    theta_2 = [AVRB(True_thetas[:,1], est_thetas[:,1]), RMSE(True_thetas[:,1], est_thetas[:,1]), Corr(True_thetas[:,1], est_thetas[:,1])]
    theta_3 = [AVRB(True_thetas[:,2], est_thetas[:,2]), RMSE(True_thetas[:,2], est_thetas[:,2]), Corr(True_thetas[:,2], est_thetas[:,2])]
    df = pd.DataFrame(np.transpose(np.array([theta_1,theta_2,theta_3])), columns = ['Theta1','Theta2','Theta3'])
    if vae:
        df.insert(0, "Model", ['VAE']*3, True)
    else:
         df.insert(0, "Model", ['AE']*3, True)
    df.insert(4,"Statistic", ['AVRB', 'RMSE', 'CORR'])
    return df

    












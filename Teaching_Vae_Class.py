# -*- coding: utf-8 -*-
"""
Teaching_VAE:
    
Here we will have all the functions necessary to create a Vae with no outside
dependencies.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as k



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




###############################################################################
#Create_Teaching_Vae: creates the vae
#INPUTS:
    #dist: the hidden distribution of the vae
    #qmat: the qmatrix
    #num_questions: # of questions in each test
    #architecture_type: which architecture to use before the stochastic layer
    #dropout_rate: how much dropout in the layer specified by architecture
#OUTPUTS:
    #model: a keras model of the vae
###############################################################################
class Teaching_Vae:
    def __init__(self, dist, qmat, num_questions, architecture_type = 0, 
                 dropout_rate = 0.0, activation = 'sigmoid'):
        #constants
        self.qmat = qmat.astype('float32')
        class Qmat_semi_sigmoid(k.layers.Layer):    
            def __init__(self):
                super(Qmat_semi_sigmoid, self).__init__()
                self.qmat_ = tf.Variable(initial_value = qmat.astype('float32'), trainable = False, 
                                        dtype = tf.float32, name = 'Qmat')            
            def build(self, input_shape):
                self.log_A = self.add_weight(shape=(3, num_questions),
                                         initializer='random_normal',
                                         trainable=True, name = 'log_A')
                self.B = self.add_weight(shape=( num_questions,),
                                         initializer='random_normal',
                                         trainable=True, name = 'B')
                assert self.log_A.shape == self.qmat_.shape
            def call(self, inputs):
                thetas = inputs
                return tf.math.sigmoid(tf.matmul(thetas, self.qmat_ * tf.exp(self.log_A)) - self.B)
        
        #Creating the model
        self.input_ = k.Input(shape = (num_questions,), name = 'Encoder_Input', dtype = 'float32')    
        #hidden1 = k.layers.Dense(10, name = 'Encoder_hidden', activation = 'sigmoid')(self.input_)
        hidden1 = self.Architecture(layers = self.input_, dropout_rate = dropout_rate, 
                                    type_ = architecture_type, activ = activation)
        loc, scale, self.theta = stochastic_layer(prev = hidden1, dist = dist)
        X_hat = Qmat_semi_sigmoid()(self.theta)
        self.model = k.Model(inputs = self.input_, outputs = X_hat)
        
        #Adding kl-divergence for the distribution
        add_vae_loss(model = self.model, loc = loc, 
                                          scale = scale, dist = dist)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer, loss= xent)
        
    def train(self, data, epochs = 100):
        thetas_grabber = k.Model(inputs = self.input_, outputs = self.theta)#used to get the hidden thetas at each epoch
        class GetEmbeddings(k.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                #index of qmat_semi_sigmoid layer
                index = np.reshape(np.where(['qmat' in var.name for var in self.model.layers])[0], ())
                logs['log_A'] = self.model.layers[index].get_weights()[0]
                logs['B'] = self.model.layers[index].get_weights()[1]
                logs['thetas'] = thetas_grabber(data[:,2:])
        
        #min_delta used to be 0.
        early_stopping = k.callbacks.EarlyStopping(monitor='val_loss', min_delta=10., 
                                                  patience=5, restore_best_weights = True)
        nanstop = tf.keras.callbacks.TerminateOnNaN()
        #Training the model
        H = self.model.fit(data[:,2:], data[:,2:], validation_split = .05,
                           epochs=epochs, batch_size = 128, callbacks=[GetEmbeddings(), early_stopping, nanstop])
        return H
        
    def plot_model(self):
        return k.utils.plot_model(self.model, 'my_first_model.png',show_shapes=True)
    
    ###########################################################################
    #Architecture: a function that will specify what NN architectures
    # will be used between the input layer and the stochastic layer. 
    # To be tested with the Experiment Table Function.
    #Input:
    #   layers: the previous layer of the NN   
    #   dropout_rate: fraction of inputs to drop in dropout layers.
    #   type_:   0) same as paper    Test->10->hidden
    #           1) dropout          Test->10->Dropout(n%)-> hidden
    #           2) 2 level%2 drop   Test-> test%2 -> Drop -> prev %2->Drop-> hidden
    #           3) 3 level %2 drop  Test-> test%2 -> Drop -> prev %2->Drop->prev %2->Drop-> hidden
    #   activ: the activation.  The paper used the 'sigmoid'.  
    #Output:
    #   layers: the completed middle layers
    ###########################################################################
    def Architecture(self, layers, dropout_rate = 0.0, type_ = 0, activ = 'sigmoid'):
        num_questions = layers.shape[1]
        hidden_neurons = np.ceil(num_questions / 2)
        if type_ ==0:
            layers = k.layers.Dense(10, name = 'Encoder_hidden_1', activation = activ)(layers)
        if type_ ==1:
            layers = k.layers.Dense(10, name = 'Encoder_hidden_1', activation = activ)(layers)    
            #layers = k.layers.BatchNormalization(name = 'Batch_Norm_1')(layers) #This creates high variance.  Commented out
            layers = k.layers.Dropout(rate = dropout_rate, name = 'Dropout_1')(layers)
        else:
            for i in range(type_):
                layers = k.layers.Dense(hidden_neurons, name = 'Encoder_hidden_{}'.format(i), activation = activ)(layers) 
                #layers = k.layers.BatchNormalization(name = 'Batch_Norm_{}'.format(i))(layers) #Creates too high variance
                layers = k.layers.Dropout(rate = dropout_rate, name = 'Dropout_{}'.format(i))(layers)
                hidden_neurons = np.ceil(hidden_neurons / 2)
        return layers
###############################################################################
#xent: gives the cross entropy betwen the predictions and their true values
###############################################################################
def xent(y_true, y_pred):
    X_hat_clipped = tf.compat.v1.clip_by_value(y_pred, 1e-10, 0.99999)
    encode_decode_loss_ = y_true * tf.math.log(1e-10 + X_hat_clipped) + (1 - y_true) * tf.math.log(1e-10 + 1 - X_hat_clipped) #Bernoulli is p(x|z)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss_, 1)
    return tf.reduce_sum(encode_decode_loss) 

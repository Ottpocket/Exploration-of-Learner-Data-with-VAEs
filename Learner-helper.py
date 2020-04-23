# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 08:05:27 2020

@author: andre
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
import os
import pandas as pd
os.chdir('C:/Users/andre/Desktop/Ecole/TF/Educational Assessment')
import Teaching_vaes_helper


###############################################################################
#get_stats: obtains the RMSE and CORR for the A, B, and Theta estimations
#INPUTS:
    #h: the history dict from the keras model
    #qmat: the qmatrix
    #amat: 
    #b:
    #students
    #tests
    #questions
    #network_num: what number neural network is this?
    #studtest: the first two rows of data which given the student num and the test num
    #dist:
#OUTPUTS:
    #df_row: df with cols [students, tests, questions, dist, A_RMSE, A_Corr, Theta_RMSE, Theta_Corr, B_RMSE, B_Corr]
###############################################################################
def Get_stats(H, qmat, amat, bvec, students, thetas, tests, questions, network_num, studtest, dist):
    #H = history_dict
    #studtest = data.values[:,0:2]
    A_list = [np.exp(a) for a in H.history['log_A']]
    B_list = H.history['B']
    th = H.history['thetas']
    dfa, dfb = Teaching_vaes_helper.get_stats_over_time(A_list, B_list, amat, bvec, qmat, matrix= True)
    th_avrb, th_RMSE, th_Corr = Teaching_vaes_helper.get_theta_stats_v2(thetas, th[-1], studtest)
    df_row = [students, tests, questions, dist, network_num, dfa['AVRB'][dfa.index[-1]], dfa['RMSE'][dfa.index[-1]],
              dfa['Corr'][dfa.index[-1]], dfb['AVRB'][dfb.index[-1]], dfb['RMSE'][dfb.index[-1]],
              dfb['Corr'][dfb.index[-1]], th_avrb, th_RMSE, th_Corr]
    return (dfa, dfb, df_row)






###############################################################################
#Create_Teaching_Vae: creates the vae
#INPUTS:
    #dist: the hidden distribution of the vae
    #qmat: the qmatrix
    #num_questions: # of questions in each test
#OUTPUTS:
    #model: a keras model of the vae
###############################################################################
class Teaching_Vae:
    def __init__(self, dist, qmat, num_questions):
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
        hidden1 = k.layers.Dense(10, name = 'Encoder_hidden', activation = 'sigmoid')(self.input_)
        loc, scale, self.theta = Teaching_vaes_helper.stochastic_layer(prev = hidden1, dist = dist)
        X_hat = Qmat_semi_sigmoid()(self.theta)
        self.model = k.Model(inputs = self.input_, outputs = X_hat)
        self.cov = k.Model(inputs = self.input_, outputs = self.model.get_layer('scale_true').output)#deletet this. testing mvn

        #Adding kl-divergence for the distribution
        Teaching_vaes_helper.add_vae_loss(model = self.model, loc = loc, 
                                          scale = scale, dist = dist)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer, loss= Teaching_vaes_helper.xent)
        
    #Returns the history object
    #
    def train(self, data, epochs = 100):
        thetas_grabber = k.Model(inputs = self.input_, outputs = self.theta)#used to get the hidden thetas at each epoch
        class GetEmbeddings(k.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                #index of qmat_semi_sigmoid layer
                index = np.reshape(np.where(['qmat' in var.name for var in self.model.layers])[0], ())
                logs['log_A'] = self.model.layers[index].get_weights()[0]
                logs['B'] = self.model.layers[index].get_weights()[1]
                logs['thetas'] = thetas_grabber(data[:,2:])

        early_stopping = k.callbacks.EarlyStopping(monitor='val_loss', min_delta=10, 
                                                  patience=5)
        nanstop = tf.keras.callbacks.TerminateOnNaN()
        #Training the model
        H = self.model.fit(data[:,2:], data[:,2:], validation_split = .05,
                           epochs=epochs, batch_size = 128, callbacks=[GetEmbeddings(), early_stopping, nanstop])
        return H

    def covariance(self, data):
        return self.cov.predict(data[:,2:])
    def layer_names(self):
        return [var.name for var in self.model.layers]
    def plot_model(self):
        return k.utils.plot_model(self.model, 'my_first_model.png',show_shapes=True)

###############################################################################
#Create_data: simulates data for student assessment.
#INPUTS:
    #num_students: (int) # of students taking the assessment
    #num_questions: (int) # of quesions in the assessment 
    #num_tests: (int) # of times the student has taken a test
#OUTPUTS:
    #Q_mat: the expert estimation of which skills pertain to which question
    #A: how much a skill effects a question
    #B: difficulty of each question
    #Theta: the student's hidden knowledge of a subject
    #data: the student responses for each question for each test
###############################################################################
def Create_data(num_students, num_questions, num_tests, num_skills):
    J = num_skills #number of hidden skills
    K = num_students #number of students
    I = num_questions #number of questions in the assessment
    
    #Q matrix is expert prepared matrix of whether a item i requires skill j
    Q_mat = np.random.binomial(n=1,p=.5, size = [J,I])
    
    #Discrimination parameters: how important is skill j for item i 
    A = np.random.uniform(low=0.25, high = 1.75, size = [J,I])
    
    #Theta: hidden skills for each student
    Theta = np.random.normal(loc = 0.0, scale=1.0, size = [K,J])
    np.savetxt('Theta.csv',Theta, delimiter=',')
    
    #B: the difficulty of each question
    B= np.random.uniform(low=-3.0, high = 3.0, size = [1, I])
    
    hidden = -1 * np.dot(Theta, (Q_mat * A)) + B# Equation 1 from the paper
    
    def sigmoid(x):
        return pow((1 + np.exp(x)), -1)
    
    prob_answers = sigmoid(hidden)#the probability a question is answered correctly
    
    data_rows = [] #[student, test #, q1, q2,...,qnum_questions]
    col_names = ['student','test_num']
    for question in range(I):
        col_names.append('Q{}'.format(question+1))
    for student in range(prob_answers.shape[0]):
        for test_num in range(num_tests):
            row = [None]*(num_questions + 2)#[student, test #, q1,q2,...,qnum_questions]
            row[0] = student
            row[1] = test_num
            for question in range(prob_answers.shape[1]):
                row[question+2] = np.random.binomial(n=1,p=prob_answers[student, question], size = None)
            data_rows.append(row)    
            
    data = pd.DataFrame(data = data_rows, columns = col_names)  
    return (Q_mat, A, B, Theta, data)
    
###############################################################################
#Experiment_table: returns an experiment for neural networks
#INPUTS:
    #num_students: (list) # of students in data
    #num_tests: (list) # of times each student took the test
    #num_questions: (list) # of questions in an assessment
    #num_networks: (int) # of nns to be tested for each of the above
    #which_dists: (list of strings) the distribution used
#OUTPUTS:
    #raw_df: a df with each neural network training as a row
    #agg_df: a df with each row as the average of the num_times a nn was ran
###############################################################################
def Experiment_table(num_students, num_tests, num_questions, num_networks, which_dists):
    df_list = []
    dfa_list = []
    dfb_list = []
    col_names = ['students', 'tests', 'questions', 'dist', 'network_num', 'A_AVRB', 'A_RMSE', 'A_Corr', 
                 'B_AVRB', 'B_RMSE', 'B_Corr', 'th_avrb', 'th_RMSE', 'th_Corr']
    current_iteration=0
    tot_iterations=len(num_students) * len(num_tests) * len(num_questions) * num_networks * len(which_dists)
    for students in num_students:
        for tests in num_tests:
            for questions in num_questions:
                for dist in which_dists:
                    for networks in range(num_networks):
                    
                        #just for testing
                        if False:
                            students = 1000
                            tests = 5
                            questions = 50
                            dist = 'norm'
                            networks = 1
                        print('\nCreating data for network {} of {}'.format(current_iteration, tot_iterations))
                        #create_data
                        qmat, amat, bvec, thetas, data = Create_data(num_students = students, num_questions = questions, num_tests= tests, num_skills = 3)
                        
                        #create_network
                        model = Teaching_Vae(dist = dist, qmat = qmat, num_questions = questions)
                        #get the relevant stats from the trained network
                        history_dict = model.train(data = data.values.astype('float32'))
                        dfa, dfb, df_row = Get_stats(H = history_dict, qmat = qmat, amat = amat, 
                                           bvec = bvec, students = students, thetas = thetas, tests = tests,
                                           questions = questions, network_num = networks, dist = dist,
                                           studtest = data.values[:,0:2])
                        df_list.append(df_row)
                        dfa_list.append(dfa)
                        dfb_list.append(dfb)
                        current_iteration = current_iteration + 1
                        
    df_raw = pd.DataFrame(df_list, columns = col_names)
    df_agg = df_raw.groupby(['students','tests','questions', 'dist']).agg({'A_AVRB':'mean', 
                           'A_RMSE':'mean', 'A_Corr':'mean', 'B_AVRB':'mean', 
                           'B_RMSE':'mean', 'B_Corr':'mean', 'th_avrb':'mean', 
                           'th_RMSE':'mean', 'th_Corr':'mean'})
    return (df_raw, df_agg, dfa_list, dfb_list)


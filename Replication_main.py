# -*- coding: utf-8 -*-
"""
Replication __main__
This function creates a NN as described in Autoencoders for Education 
Assessment, and creates the graphs just like that paper.  Specifically, we 
replicate table 1, figure 3, figure 4, table 2, and figure 5.  
@author: andre
"""

import os
os.chdir("C:/Users/andre/Documents/GitHub/Exploration-of-Learner-Data-with-VAEs")
from Experiment_table_Function import Experiment_table
from Teaching_Vae_Class import Teaching_Vae
from Data_Gen import Create_data
from Graphical_Functions import Get_stats, Table_1

#Creating the data for the network
num_students = 10000
num_questions = 28
num_tests = 10
num_skills = 3
dist = 'norm'
Q_mat, A, B, theta, data = Create_data(num_students = num_students, num_questions = num_questions,
                                    num_tests = num_tests, num_skills = num_skills)

#create the network and train it
vae = Teaching_Vae(dist = 'norm', qmat = Q_mat, num_questions = 28)
H = vae.train(data)

#getting the statistics from the neural network
dfa, dfb, dfrow = Get_stats(H = H, qmat = Q_mat, amat = A, bvec = B, 
                            students = num_students, thetas = theta, tests = num_tests,
                            questions = num_questions, network_num = 1, 
                            studtest = data[:,0:2], dist = dist)

#Replicating the Figures from the paper
tab1, fig3, fig4 tab2, fig5, vae, data_list = replication(num_students, num_questions, num_tests,
                                          num_skills, dist)

###############################################################################
#Replication_of_Paper_Figures: Function that trains a NN and replicates the data
# for all figures and tables in the paper
#INPUT:
#   Input_data = list of data if you have data you want to train the networks
#                on.  If None, ignored. Format: [Q_mat, A, B, theta, data]
#   num_students: the number of simulated students.
#   num_questions: the number of questions in a test
#   num_tests: the number of tests given for each student
#   num_skills: the number of hidden skills being tested
#   dist: the distribution of the stochastic layer in the VAE
#OUTPUT:
#   tab1: the first table in the paper
#   fig3: 
#   fig4: '''
#   tab2: '''
#   fig5:
#   models: [vae, ae] list of the fully trained vae and ae
#   data_list: [Qmat, A, B, theta, data] used in simulation
###############################################################################
def Replication_of_Paper_Figures(Input_data = None, num_students = 10000, 
                                 num_questions= 28, num_tests = 10, num_skills = 3,
                                 dist= 'norm'):
    #Obtaining data for the Network
    if Input_data is not None:
        Q_mat, A, B, theta, data = Input_data
    else:
        Q_mat, A, B, theta, data = Create_data(num_students, num_questions,
                                    num_tests, num_skills)
    #Initializing the networks
    vae = Teaching_Vae(dist = 'norm', qmat = Q_mat, num_questions = A.shape[1]) 
    H_vae = vae.train(data)
    
    ae = Teaching_Vae(dist = 'None', qmat = Q_mat, num_questions = A.shape[1])
    H_ae = ae.train(data)
    
    
    tab1_vae = Table_1(dfa1_vae, dfa2_vae, dfa3_vae, dfb_vae, dist)
    
    
    
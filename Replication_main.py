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
from Graphical_Functions import Get_stats

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

# -*- coding: utf-8 -*-
"""
Replication __main__
This function creates a NN as described in Autoencoders for Education 
Assessment, and creates the graphs just like that paper.  Specifically, we 
replicate table 1, figure 3, figure 4, table 2, and figure 5.  
@author: andre
"""
import pandas as pd
import numpy as np
import os
os.chdir("C:/Users/andre/Documents/GitHub/Exploration-of-Learner-Data-with-VAEs")
from Teaching_Vae_Class import Teaching_Vae
from Data_Gen import Create_data
from Graphical_Functions import Table_1, get_stats_over_time, get_theta_stats_v2


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
#   Input_Vae: a vae trained outside the function.  If empty, the function will
#               train a vae.
#   Input_Ae = an ae trained outside the function.  If empty, the function will
#               train an ae.
#OUTPUT:
#   tab1_data: the data for the first table in the paper
#   fig3_data: '''' 
#   fig4_data: '''
#   tab2_data: '''
#   fig5_data:
#   vae: the trained vae.
#   ae:  the trained ae
#   data_list: [Qmat, A, B, theta, data] used in simulation
###############################################################################
def Replication_of_Paper_Figures(Input_data = None, num_students = 10000, 
                                 num_questions= 28, num_tests = 10, num_skills = 3,
                                 dist= 'norm', vae = None, ae = None):
    #Obtaining data for the Network
    if Input_data is not None:
        Q_mat, A, B, theta, data = Input_data
    else:
        Q_mat, A, B, theta, data = Create_data(num_students, num_questions,
                                    num_tests, num_skills)
    #Initializing the networks
    if vae is None:
        vae = Teaching_Vae(dist = 'norm', qmat = Q_mat, num_questions = A.shape[1]) 
        H_vae = vae.train(data)
    else:
        H_vae = vae.train(data, epochs = 1)
    if ae is None:
        ae = Teaching_Vae(dist = 'None', qmat = Q_mat, num_questions = A.shape[1])
        H_ae = ae.train(data)
    else:
        H_ae = ae.train(data, epochs = 1)
    #Extracting the statistics from the networks
    dfa1_vae , dfa2_vae,  dfa3_vae, dfb_vae = get_stats_over_time(A_list=[], B_list=[], a_true = A, 
                                            b_true= B, qmat = Q_mat, matrix = False, H = H_vae)
    dfa1_ae , dfa2_ae,  dfa3_ae, dfb_ae = get_stats_over_time(A_list=[], B_list=[], a_true = A, 
                                            b_true= B, qmat = Q_mat, matrix = False, H = H_ae)
    
    tab1_vae = Table_1(dfa1_vae, dfa2_vae, dfa3_vae, dfb_vae, dist)
    tab1_ae =  Table_1(dfa1_ae, dfa2_ae, dfa3_ae, dfb_ae, 'AE')
    tab1 = pd.concat([tab1_vae,tab1_ae])
    tab1 = tab1.groupby(['Statistic','Model']).agg('mean').reset_index()
    
    #Indicating which skill is used
    skill=[]
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            skill.append(i)
    skill = np.array(skill)[np.where(np.reshape(Q_mat, [-1])==1)[0]]
    
    #Getting rid of a_i,j which were masked by the Q matrix
    A_qmat= np.reshape(A * Q_mat, [-1,])
    A = np.delete(A_qmat, np.where(A_qmat == 0))
    A_ae = np.reshape(Q_mat * np.exp(H_ae.history['log_A'][-1]), [-1])
    A_ae = np.delete(A_ae, np.where(A_ae == 0))
    A_vae= np.reshape(Q_mat * np.exp(H_vae.history['log_A'][-1]), [-1])
    A_vae = np.delete(A_vae, np.where(A_vae==0))
    fig3 = pd.DataFrame({'True_Values': A, 
                         'Estimates_ae': A_ae,
                         'Estimates_vae': A_vae,
                         'skill_num': skill })
    
    fig4 = pd.DataFrame({'True_Values': np.reshape(B,[-1]),
                         'Estimates_ae': H_ae.history['B'][-1],
                         'Estimates_vae': H_vae.history['B'][-1]})
    
    #Vae stats
    tab2_vae, thetas_vae = get_theta_stats_v2(theta, H_vae.history['thetas'][-1] , 
                                              data[:,0:2],Table2 = True, model = 'Vae')
    tab2_ae, thetas_ae = get_theta_stats_v2(theta, H_ae.history['thetas'][-1] , 
                                              data[:,0:2],Table2 = True, model = 'ae')
    tab2 = pd.concat([tab2_vae, tab2_ae])
    tab2 = tab2.groupby(['Statistic','Model']).agg('mean')
    
    #fig5
    thetas_ae = pd.DataFrame(thetas_ae)
    thetas_ae.columns = ['Theta{}_ae'.format(i+1) for i in range(thetas_ae.shape[1])]    
    thetas_vae = pd.DataFrame(thetas_vae)
    thetas_vae.columns = ['Theta{}_vae'.format(i+1) for i in range(thetas_vae.shape[1])]
    thetas_true = pd.DataFrame(theta)
    thetas_true.columns = ['Theta{}_true'.format(i+1) for i in range(thetas_vae.shape[1])]
    fig5 = pd.concat([thetas_true,thetas_ae, thetas_vae], axis= 1)
    
    return(tab1, fig3, fig4, tab2, fig5, vae, ae, [Q_mat, A, B, theta, data])
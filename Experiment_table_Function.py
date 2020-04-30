# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:36:59 2020

@author: andre
"""
import pandas as pd
import os
os.chdir("C:/Users/andre/Documents/GitHub/Exploration-of-Learner-Data-with-VAEs")
from Data_Gen import Create_data
from Teaching_Vae_Class import Teaching_Vae
from Graphical_Functions import Get_stats

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
def Experiment_table(num_students, num_tests, num_questions, num_networks, 
                     which_dists, arches = [0], activations = ['sigmoid'], dropouts = [.1]):
    df_list = []
    dfa_list = []
    dfb_list = []
    col_names = ['students', 'tests', 'questions', 'dist', 'Arch_type', 'dropout_rate','network_num', 'A_AVRB', 'A_RMSE', 'A_Corr', 
                 'B_AVRB', 'B_RMSE', 'B_Corr', 'th_avrb', 'th_RMSE', 'th_Corr']
    current_iteration=0
    tot_iterations=len(num_students) * len(num_tests) * len(num_questions) * num_networks * len(which_dists) * len(arches) * len(activations) * len(dropouts)
    for students in num_students:
        for tests in num_tests:
            for questions in num_questions:
                for dist in which_dists:
                    for networks in range(num_networks):
                        for arch in arches:
                            for activation in activations:
                                for dropout in dropouts:
                                    
                                    print('\nCreating data for network {} of {}'.format(current_iteration, tot_iterations))
                                    #create_data
                                    qmat, amat, bvec, thetas, data = Create_data(num_students = students, num_questions = questions, num_tests= tests, num_skills = 3)
                                    
                                    #create_network
                                    model = Teaching_Vae(dist = dist, qmat = qmat, num_questions = questions,
                                                         architecture_type = arch, dropout_rate = dropout)
                                    
                                    #get the relevant stats from the trained network
                                    history_dict = model.train(data = data)
                                    dfa, dfb, df_row = Get_stats(H = history_dict, qmat = qmat, amat = amat, 
                                                       bvec = bvec, students = students, thetas = thetas, tests = tests,
                                                       questions = questions, network_num = networks, dist = dist,
                                                       studtest = data[:,0:2], arch_type = arch, dropout_rate = dropout)
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


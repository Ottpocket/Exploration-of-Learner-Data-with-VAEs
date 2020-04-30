# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:12:34 2020
Graphical Functions for illustrating facets of Teaching VAEs
@author: andre
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr


###############################################################################
#Basic statistical functions
###############################################################################
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
#get_stats: obtains the RMSE and CORR for the A, B, and Theta estimations
#INPUTS:
    #h: the history dict from the keras model
    #qmat: the qmatrix
    #amat: 
    #b:
    #students: number of students
    #tests: number of tests
    #questions: number of questions
    #network_num: what number neural network is this?
    #studtest: the first two rows of data which given the student num and the test num
    #dist:
    #arch_type: the type of architecture in the model.  an int from 0-3
    #dropout_rate: the percentage of neurons dropped by dropout
#OUTPUTS:
    #df_row: df with cols [students, tests, questions, dist, arch_type, dropout_rate, 
    #                      A_RMSE, A_Corr, Theta_RMSE, Theta_Corr, B_RMSE, B_Corr]
###############################################################################
def Get_stats(H, qmat, amat, bvec, students, thetas, tests, questions, network_num, 
              studtest, dist, arch_type, dropout_rate):
    #H = history_dict
    #studtest = data.values[:,0:2]
    A_list = [np.exp(a) for a in H.history['log_A']]
    B_list = H.history['B']
    th = H.history['thetas']
    dfa, dfb = get_stats_over_time(A_list, B_list, amat, bvec, qmat, matrix= True)
    th_avrb, th_RMSE, th_Corr = get_theta_stats_v2(thetas, th[-1], studtest)
    df_row = [students, tests, questions, dist, arch_type, dropout_rate, network_num, 
              dfa['AVRB'][dfa.index[-1]], dfa['RMSE'][dfa.index[-1]],
              dfa['Corr'][dfa.index[-1]], dfb['AVRB'][dfb.index[-1]], dfb['RMSE'][dfb.index[-1]],
              dfb['Corr'][dfb.index[-1]], th_avrb, th_RMSE, th_Corr]
    return (dfa, dfb, df_row)



###############################################################################
#get_stats_over_time: gets AVRB, RMSE, and CORR from a_1, a_2, a_3, and b for each
# epoch of training
#input:
#   A_list: list of (28x3) a matrices for each epoch of training
#   B_list: list of (28x1) b vectors '''''''
#   a_true: true values of 'a' from Data_Gen.py
#   b_true: true vales of 'b'  '''''''
#   matrix: do you want individual a rows or the whole thing as a?  default False for backwards compat    
#   H: the history from a trained NN.  If this is given, A_list, B_list will be 
#       autopopulated from H.
#Output:
#   a_1: dataframe where rows are the epoch number and the cols are epoch, AVRB, RMSE, and Corr
#   a_2,a_3,b: dataframes of the same type
###############################################################################
def get_stats_over_time(A_list, B_list, a_true, b_true, qmat, matrix = False, H = None):
    if H is not None:
        if matrix ==False:
            A_list = [np.transpose(np.exp(a)) for a in H.history['log_A']]
            qmat = np.transpose(qmat)
            a_true = np.transpose(a_true)
        else:
            A_list = np.exp(H.history['log_A'])
        B_list = np.exp(H.history['B'])
    b = []
    if H is not None:
        num_questions = A_list[0].shape[0]
    else:
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

###############################################################################
#Saving the Thetas 
#takes in train/val_student_thetas, and outputs cols 
#   [studentid, meanTheta1, meanTHeta2, meanTheta3]
###############################################################################
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
    #table2: if True, give the output in shape of Table2 of paper
#OUTPUT:
    #AVRB:
    #RMSE:
    #Corr:
###############################################################################    
def get_theta_stats_v2(correct, guess, studtest, Table2 = False, model= None):
    cols = ['student', 'test']
    for i in range(guess.shape[1]):
        cols.append('skill{}'.format(i+1))
    guess_df = pd.DataFrame(data = np.hstack([studtest, guess]), columns = cols)
    guess_agg = guess_df.groupby('student').agg({'skill1':'mean', 'skill2':'mean', 'skill3':'mean'}).values
    if Table2 == False:
        return (AVRB(correct, guess_agg), RMSE(correct, guess_agg), Corr(correct, guess_agg))
    else:
        theta1 = [AVRB(correct[:,0], guess_agg[:,0]), RMSE(correct[:,0], guess_agg[:,0]), Corr(correct[:,0], guess_agg[:,0])]
        theta2 = [AVRB(correct[:,1], guess_agg[:,1]), RMSE(correct[:,1], guess_agg[:,1]), Corr(correct[:,1], guess_agg[:,1])]
        theta3 = [AVRB(correct[:,2], guess_agg[:,2]), RMSE(correct[:,2], guess_agg[:,2]), Corr(correct[:,2], guess_agg[:,2])]
        df = pd.DataFrame({'Theta1': theta1, 'Theta2': theta2, 'Theta3':theta3,
                           'Statistic': ['AVRB','RMSE','Corr'], 'Model': [model]*3})
        return (df, guess_agg)


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
def Table_2(train_theta, val_theta, vae, True_thetas):
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











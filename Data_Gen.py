# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:30:39 2020

@author: andre
"""
import numpy as np
import pandas as pd

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

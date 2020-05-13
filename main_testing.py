# -*- coding: utf-8 -*-
"""
Main:  Playground for testing implemented functions
@author: andre
"""
import pandas as pd
import os
os.chdir("C:/Users/andre/Documents/GitHub/Exploration-of-Learner-Data-with-VAEs")
from Teaching_Vae_Class import Teaching_Vae
from Data_Gen import Create_data
from Experiment_table_Function import Experiment_table

###############################################################################
#Super Big Architecture Test for NNs.  Here I will test:
# 1) Do different activations have stronger correlation with targets? 
# 2) Do different layers of NNs have stronger baseline performance?
# 3) What is the optimal dropout for the models to improve?
###############################################################################
df_raw, df_agg, dfa_list, dfb_list = Experiment_table(num_students= [100], num_tests = [10], 
                                                      num_questions =[28], num_networks = 20, which_dists = ['norm'],
                                                      arches = [2], activations = ['sigmoid', 'relu'], dropouts = [0.3])
df_raw, df_agg, dfa_list, dfb_list = Experiment_table(num_students= [1000, 5000, 10000], num_tests = [1,10], 
                                                      num_questions =[30,50], num_networks = 5, which_dists = ['norm','laplace'],
                                                      arches = [1,2,3], activations = ['sigmoid', 'relu'], dropouts = [0.0,0.1,0.2])

raw = pd.read_csv('raw_testing.csv')
ag = raw.groupby(['students','tests','questions', 'Arch_type','dropout_rate']).agg({'th_Corr':{'mean','count'}, 'epochs':{'min','mean'}})
#Test of data
import seaborn as sns
sns.set(style="whitegrid")
plt.savefig()
raw['questions_tests'] = raw['questions'].apply(str)+'_' +raw['tests'].apply(str)
raw['drop_arch'] = raw['dropout_rate'].apply(str)+'_' +raw['Arch_type'].apply(str)  
g = sns.catplot(x='questions_tests', y='th_Corr', hue= 'drop_arch', data = raw,
                height=6, aspect = 3, kind='bar', palette='muted')
g.set_ylabels("Correlation")
g.savefig('JibberJabber.png')

#Creating the data for the network
num_students = 10000
num_questions = 28
num_tests = 10
num_skills = 3
dist = 'norm'
#Getting data for NNs
Q_mat, A, B, theta, data = Create_data(num_students = num_students, num_questions = num_questions,
                                    num_tests = num_tests, num_skills = num_skills)
input_dat = [Q_mat, A, B, theta, data]

#Training VAE
from Teaching_Vae_Class import Teaching_Vae
vae = Teaching_Vae(dist = 'norm', qmat = Q_mat, num_questions = num_questions,  
                   dropout_rate = 0.1, architecture_type = 2)
vae.plot_model()
H_vae = vae.train(data)
H_vae.history.keys()
a_1_df, a_2_df, a_3_df, b_df = get_stats_over_time([], [], A, B, Q_mat, matrix = False, H = H_vae)

#Training AE
ae = Teaching_Vae(dist = 'None', qmat = Q_mat, num_questions = num_questions)
H_ae = vae.train(data)
get_stats_over_time([], [], A, B, Q_mat, matrix = False, H = H_ae)


#Replicating the Figures from the paper
tab1, fig3, fig4, tab2, fig5, vae, ae, _ = Replication_of_Paper_Figures(input_dat, vae = vae, ae = ae)

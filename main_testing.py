# -*- coding: utf-8 -*-
"""
Main:  Playground for testing implemented functions
@author: andre
"""
import pandas as pd
import os
os.chdir("C:/Users/andre/Documents/GitHub/Exploration-of-Learner-Data-with-VAEs")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_color_codes("colorblind")
from Replication_main import Replication_of_Paper_Figures
from Teaching_Vae_Class import Teaching_Vae
from Data_Gen import Create_data
from Experiment_table_Function import Experiment_table
from scipy.stats import pearsonr

###############################################################################
#Super Big Architecture Test for NNs.  Here I will test:
# 1) Do different activations have stronger correlation with targets? 
# 2) Do different layers of NNs have stronger baseline performance?
# 3) What is the optimal dropout for the models to improve?
###############################################################################
#Testing to see if it works
df_raw, df_agg, dfa_list, dfb_list = Experiment_table(num_students= [10000], num_tests = [10], 
                                                      num_questions =[28], num_networks = 5, which_dists = ['norm'],
                                                      arches = [2], activations = ['sigmoid','tanh'], dropouts = [0.0])



###############################################################################
#Data for Activation Testing
###############################################################################
df_raw, df_agg, dfa_list, dfb_list = Experiment_table(num_students= [10000], num_tests = [10], 
                                                      num_questions =[28], num_networks = 20, which_dists = ['norm'],
                                                      arches = [0], activations = ['sigmoid','tanh','relu'], dropouts = [0.0])
df_raw.to_csv('activation_testing.csv', index=False)
act = pd.read_csv('./Experiment_Data/activation_testing.csv')
act.groupby('activations').agg({'th_Corr':{'mean', 'std'}})#no difference whatsoever

g = sns.catplot(x= 'activations',y='th_Corr' ,  
                data =act, kind = 'bar', palette= 'muted')
g.set_ylabels("Correlation")
g.fig.suptitle('Differences in Correlations for Activations')
g.savefig('Activations')
###############################################################################
#Data for Architecture Testing
###############################################################################
df_raw, df_agg, dfa_list, dfb_list = Experiment_table(num_students= [1000, 5000, 10000], num_tests = [1,10], 
                                                      num_questions =[30,50], num_networks = 5, which_dists = ['norm','laplace'],
                                                      arches = [1,2,3], activations = ['sigmoid', 'relu'], dropouts = [0.0,0.1,0.2])
raw = pd.read_csv('./Experiment_Data/raw_testing.csv')
#ag= raw.groupby(['students','tests','questions', 'Arch_type','dropout_rate']).agg({'th_Corr':{'mean','count'}, 'epochs':{'min','mean'}})
raw['questions_tests'] = raw['questions'].apply(str)+'_' +raw['tests'].apply(str)
raw['drop_arch'] = raw['dropout_rate'].apply(str)+'_' +raw['Arch_type'].apply(str)  
g = sns.catplot(x='questions_tests', y='th_Corr', hue= 'drop_arch', data = raw,
                height=6, aspect = 3, kind='bar', palette='muted')
g.set_ylabels("Correlation")
g.savefig('JibberJabber.png')



###############################################################################
#Data for Regularization Testing
###############################################################################


###############################################################################
#Plot of fig3
###############################################################################
tab1, fig3, fig4, tab2, fig5, vae, ae, data_list = Replication_of_Paper_Figures()

fig, ax =plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
sns.scatterplot(x = 'True_Values',y= 'Estimates_ae',  ax=ax[0], hue = 'skill_num', 
                palette = 'colorblind', data = fig3).set_title('AE Parameter Recovery')
sns.scatterplot(x = 'True_Values',y= 'Estimates_vae', ax=ax[1], hue = 'skill_num', 
                palette = 'colorblind', data= fig3).set_title('VAE Parameter Recovery')
ax[0].set_ylim([0.0,4])
ax[1].set_ylim([0.0,4])
fig.show()
fig.savefig('fig3.png')

###############################################################################
#Plot of fig 4
###############################################################################
fig, ax =plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
sns.scatterplot(x = 'True_Values',y= 'Estimates_ae',  ax=ax[0],  
                data = fig4).set_title('AE Parameter Recovery')
sns.scatterplot(x = 'True_Values',y= 'Estimates_vae',  ax=ax[1],  
                data = fig4).set_title('VAE Parameter Recovery')
ax[0].set_ylim([-4.0,4.0])
ax[1].set_ylim([-4.0,4.0])
fig.show()
fig.savefig('fig4.png')

###############################################################################
#Making Table 2 pretty
###############################################################################
Table2 = pd.DataFrame({'Statistic': ['AVRB', 'AVRB', 'CORR','CORR', 'RMSE','RMSE'],
                       'Model': ['VAE','AE','VAE','AE','VAE','AE'],
                       'Theta1': tab2['Theta1'],
                       'Theta2': tab2['Theta2'],
                       'Theta3': tab2['Theta3'],})

###############################################################################
#Plot of fig 5
###############################################################################
fig, ax =plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
sns.scatterplot(x = 'Theta1_true',y= 'Theta1_ae',  ax=ax[0],  
                data = fig5).set_title('AE prediction of 1st latent trait')
sns.scatterplot(x = 'Theta1_true',y= 'Theta1_vae',  ax=ax[1],  
                data = fig5).set_title('VAE prediction of 1st latent trait')
ax[0].set_ylim([-4.0,4.0])
ax[1].set_ylim([-4.0,4.0])
fig.show()
fig.savefig('fig5.png')


fig, ax =plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
sns.scatterplot(x = 'Theta2_true',y= 'Theta2_ae',  ax=ax[0],  
                data = fig5).set_title('AE prediction of 2nd latent trait')
sns.scatterplot(x = 'Theta2_true',y= 'Theta2_vae',  ax=ax[1],  
                data = fig5).set_title('VAE prediction of 2nd latent trait')
ax[0].set_ylim([-4.0,4.0])
ax[1].set_ylim([-4.0,4.0])
fig.show()
#fig.savefig('fig5_2.png')

fig, ax =plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
sns.scatterplot(x = 'Theta3_true',y= 'Theta3_ae',  ax=ax[0],  
                data = fig5).set_title('AE prediction of 3rd latent trait')
sns.scatterplot(x = 'Theta3_true',y= 'Theta3_vae',  ax=ax[1],  
                data = fig5).set_title('VAE prediction of 3rd latent trait')
ax[0].set_ylim([-4.0,4.0])
ax[1].set_ylim([-4.0,4.0])
fig.show()
#fig.savefig('fig5_3.png')


pearsonr(fig4['True_Values'], fig4['Estimates_vae'])
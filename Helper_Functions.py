# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:06:04 2020
A file of commonly used helper functions for data science.

@author: andre
"""
import pandas as pd
import numpy as np


###############################################################################
#Bin_Data: puts data into a dense bin with bins.
#Input:
#   dfs: list of data frames ie [train, test]
#   col: the numeric column to be binned
#   bins: a list of numbers that give the bins for the col.  
#NOTE: if bins = [0,1,2,3], the binned categories are (0,1], (1,2], (2,3] with 
#           everything else being a NaN
#Output
#   df2s: the list of dataframe plus 'col_bin' column appended to each
###############################################################################
def Bin_Data(dfs, col, bins, na_fill = -1):
    #dfs = [raw, test]
    bins.sort()
    col_name = col+'_bin'
    df2s = []
    for df2 in dfs:
        df = df2.copy()
        df[col_name]=np.nan
        min_ = bins[0]
        for i,bin_ in enumerate(bins):
            if i ==0:
                continue
            #print('Min: {}, Max: {}, bin#: {}'.format(min_, bin_, i))
            df.loc[( df[col] > min_) & (df[col] <= bin_), 'Age_bin'] = i
            min_ = bin_
        df = df.fillna(na_fill)
        df2s.append(df)
    return df2s


###############################################################################
#Find_Outliers: takes a data frame and uses Tukey method to locate outliers.
# Labels the outliers in an outlier column appended to the df
#Input:
#   df: the data frame in question
#   cols: a list of columns wanting to be checked for outliers.  Must be numeric columns
#Output:
#   df: the original dataframe with column "IsOutlier" that indicates if an 
#       observation had an outlier and why it was an outlier.
###############################################################################
def Find_Outliers(df, cols):
    df["IsOutlier"] = np.repeat(['no'],repeats = len(df))
    for col in cols:
        Q1 = np.nanpercentile(df[col], 25)
        Q3 = np.nanpercentile(df[col],75)
        IQR = Q3 - Q1
        tukey_outlier = 1.5*IQR
        df.loc[(df[col] < Q1 - tukey_outlier) | (df[col] > Q3 + tukey_outlier), "IsOutlier"] = col
    return df

###############################################################################
#NN_cleaner: cleans data st it can be effectively ran through a MLP.
#Input:
#   dfs = list of dfs ie [train, test]
#   delete = lsit of columns to drop
#   Normalize: list of columns to minus mean div by sd
#   to_one_hot: turn dense categorical col into one hot columns
#Output:
#   dfs: list of dataframes with columns cleaned
###############################################################################
def NN_cleaner(dfs, delete, Normalize, to_one_hot):
    df2s = []
    for df in dfs:
        #df = train_nn
        df2 = df.copy()
        df2.drop(labels = delete, axis = 1, inplace = True)#deleting bad cols
        for n in Normalize:
            df2.loc[:,n] = (df2[n] - np.mean(df2[n])) / np.std(df2[n])
        for col in to_one_hot:
            #col = 'Pclass'
            one_hots = pd.get_dummies(df[col])
            one_hots.columns = ['{}_{}'.format(col,i) for i in one_hots.columns]
            df2 = pd.concat([df2, one_hots], axis=1)
        df2.drop(labels = to_one_hot, axis = 1, inplace = True)#deleting bad cols
        df2s.append(df2)
    return df2s


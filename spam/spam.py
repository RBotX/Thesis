#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:27:37 2017

@author: dan
"""
import pandas as pd
import numpy as np
def tidy_split(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df

df=pd.DataFrame()    
offset=0
for i in range(0,15):
    t = open("/home/dan/Thesis/spam/task_b_lab/task_b_u%02d_eval_lab.tf"%(i),"r").read().splitlines()
    t=list(map(lambda x:x.strip(),t))
    tt=pd.DataFrame(t).reset_index(drop=True)
    dd = pd.DataFrame()
    dd['Label'] = tt.apply(lambda x:x[0].split(' ')[0],axis=1).reset_index(drop=True)
    dd['data'] = tt.apply(lambda x:" ".join(x[0].split(' ')[1:]),axis=1).reset_index(drop=True)    
    dd['row'] = list(range(offset+1,offset+dd.shape[0]+1))
    offset+=dd.shape[0]
    dd = tidy_split(dd, 'data', sep=' ').reset_index(drop=True)
    dd['col'] = dd.apply(lambda x:x['data'].split(':')[0],axis=1).reset_index(drop=True)
    dd['val']= dd.apply(lambda x:x['data'].split(':')[1],axis=1).reset_index(drop=True)
    
    dd['Family']="fam"+str(i)
    
    df = pd.concat([df,dd],axis=0)
    
    
    
df.to_csv("/home/dan/Thesis/spam/spam.csv")












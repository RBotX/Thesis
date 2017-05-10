#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:04:47 2017

@author: dan
"""
MATFILE="/home/dan/Thesis/LandmineData"
import scipy.io
import pandas as pd

mat = scipy.io.loadmat(MATFILE+'.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
finaldf=pd.DataFrame()
for i in range(0,data.shape[0]):
    df=pd.concat([pd.DataFrame(data.ix[i,'feature']),pd.DataFrame(data.ix[i,'label'])],axis=1)
    df.columns = ['f'+str(i) for i in range(1,df.shape[1])]+['Label']
    df['Family']="fam"+str(i+1)
    finaldf = pd.concat([finaldf,df],axis=0)    
finaldf.to_csv(MATFILE+".csv")    
              
              

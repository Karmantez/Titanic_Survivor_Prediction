#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[2]:


def process_null_data(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df


# In[3]:


def drop_unnecessary_features(df):
    df.drop(['Name','Ticket'],axis=1,inplace=True)
    return df


# In[4]:


def process_label_encoding(df):
    df['Cabin'] = df['Cabin'].str[:1]
    
    for feature in ['Cabin','Sex','Embarked']:
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df[feature])
        df[feature] = label_encoder.transform(df[feature])
    return df       


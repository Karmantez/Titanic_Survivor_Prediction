#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# In[3]:


def process_kfold(model, X_titanic_df, y_titanic_df, folds=5):
    kfold = KFold(n_splits=folds)
    scores = []

    for iter_count , (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):

        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]

        model.fit(X_train, y_train) 
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        print("KFold {0} Accuracy: {1:.4f}".format(iter_count, accuracy))     

    mean_score = np.mean(scores)
    print("Average Accuracy: {0:.4f}".format(mean_score)) 


# In[4]:


def processing(X_train, X_test, y_train, X_titanic_df, y_titanic_df, algorithm='dtc'):
    
    predict_ret = np.empty([1, 1])
    
    if algorithm == 'dtc':
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        predict_ret = dtc.predict(X_test)
        # process_kfold(dtc, X_titanic_df, y_titanic_df)
    elif algorithm == 'rfc':
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        predict_ret = rfc.predict(X_test)
        # process_kfold(rfc, X_titanic_df, y_titanic_df)
    elif algorithm == 'lr':
        lr = RandomForestClassifier()
        lr.fit(X_train, y_train)
        predict_ret = lr.predict(X_test)
        # process_kfold(lr, X_titanic_df, y_titanic_df)
    return predict_ret


# In[ ]:





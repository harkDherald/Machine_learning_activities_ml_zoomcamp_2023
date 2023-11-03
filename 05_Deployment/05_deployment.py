#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold


# In[2]:


# Data Preparation
df = pd.read_csv(r"C:\Users\Herald\Documents\ML_Zoomcamp\03_classification\Telco-Customer-Churn.csv")

# Clean columns names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Clean Categorical values
categorical_cols = list(df.dtypes[df.dtypes == 'object'].index)

for col in categorical_cols:
    df[col] = df[col].str.lower().str.replace(' ', '_')
    
# totalcharges is currently an object type and should be a float type
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce').fillna(0)
# Convert this to int 0 and 1
df.churn = (df.churn == 'yes').astype(int)


# In[3]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[4]:


# Group numerical categories
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [col for col in df_full_train.columns if col not in numerical and col != 'churn' and col != 'customerid']


# In[5]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[6]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    
    return y_pred


# In[7]:


C = 1.0
n_splits = 5


# In[8]:


# Lets loop on our kfold splits
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    
print('C= %s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[9]:


scores


# In[10]:


dv, model = train(df_full_train, df_full_train.churn.values, C=C)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# # Save the Model

# In[11]:


import pickle


# In[12]:


output_file = f'model_C={C}.bin'
output_file


# In[13]:


f_out = open(output_file, 'wb') # write to the file
pickle.dump((dv, model), f_out)
f_out.close()


# In[14]:


# A better way to write the code above (avoid forgetting to close the file)
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# # Load the Model

# In[1]:


import pickle


# In[2]:


model_file = 'model_C=1.0.bin'


# In[3]:


with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[4]:


dv, model


# In[12]:


customer = {'gender': 'female',
         'seniorcitizen': 0,
         'partner': 'yes',
         'dependents': 'no',
         'phoneservice': 'no',
         'multiplelines': 'no_phone_service',
         'internetservice': 'dsl',
         'onlinesecurity': 'no',
         'onlinebackup': 'yes',
         'deviceprotection': 'no',
         'techsupport': 'no',
         'streamingtv': 'no',
         'streamingmovies': 'no',
         'contract': 'month-to-month',
         'paperlessbilling': 'yes',
         'paymentmethod': 'mailed_check',
         'tenure': 1,
         'monthlycharges': 29.85,
         'totalcharges': 29.85}


# In[13]:


X = dv.transform([customer])


# In[14]:


model.predict_proba(X)[0,1]


# In[ ]:





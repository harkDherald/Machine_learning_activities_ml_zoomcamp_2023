#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

# Parameters
C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'


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

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Group numerical categories
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [col for col in df_full_train.columns if col not in numerical and col != 'churn' and col != 'customerid']

# Training
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    
    return y_pred

# Validation
print(f'Doing Validaton with C={C}')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('Validation results:')   
print('C= %s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# Training the final Model
print('Training the final Model')
dv, model = train(df_full_train, df_full_train.churn.values, C=C)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')

# Save the Model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}')
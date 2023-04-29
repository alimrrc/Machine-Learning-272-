#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:58:41 2023

@author: ali
"""
from sklearn.cluster import KMeans
import pandas as pd

# Load the dataset from a CSV file
df = pd.read_csv('X_train_G3tdtEn.csv')

# Get the data types of all columns
dtypes = df.dtypes

cols = df.columns

for col in cols:
    if "goods_code" in col or "model" in col or "ID" in col or col == 'Nb_of_items':
        df.drop(col, axis=1, inplace=True)

cols = df.columns

df.fillna('', inplace=True)

unique_values = {}
feature_cols = {}
feature_df = {}
features = ['item', 'price', 'make', 'purchas']

for feature in features:
    unique_values[feature] = []
    feature_cols[feature] = []
    
    for col in cols:
        if feature in col:
            feature_cols[feature].append(col)
    
    feature_df[feature] = df[feature_cols[feature]]
        
    if feature == 'item' or feature == 'make':  
        
        feature_df[feature] = feature_df[feature].apply(lambda x: x.str.replace(' ', ''))
        
        for i, row in feature_df[feature].iterrows():
            for val in row:
                if val != '':
                    if val not in unique_values[feature]:
                        unique_values[feature].append(val)
                    

new_cols = []

for feature in features:
    if feature == 'item' or feature == 'make':       
        for label_add_up in [' Number', ' Price']:
            for val in unique_values[feature]:
                new_cols.append(val + label_add_up)


data = pd.DataFrame(columns = new_cols, index = df.index)

count_item = {}
count_make = {}
price_item = {}
price_make = {}

for i, row in feature_df['item'].iterrows():
    count_item[i] = {}
    price_item[i] = {}
    
    for item in unique_values['item']:
        count_item[i][item] = 0
        price_item[i][item] = 0
    
    for index, val in row.iteritems():
        if val != '':
            i_th = int(index[-1])
            count_item[i][val] += float(feature_df['purchas'].iloc[i, i_th])
            price_item[i][val] += float(feature_df['price'].iloc[i, i_th])

for i, row in feature_df['make'].iterrows():
    count_make[i] = {}
    price_make[i] = {}
    
    for make in unique_values['make']:
        count_make[i][make] = 0
        price_make[i][make] = 0
    
    for index, val in row.iteritems():
        if val != '':
            i_th = int(index[-1])
            count_make[i][val] += float(feature_df['purchas'].iloc[i, i_th])
            price_make[i][val] += float(feature_df['price'].iloc[i, i_th])
            


# Initialize the K-Means algorithm with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit the model to the data
kmeans.fit(df)

# Predict the cluster labels for each data point
labels = kmeans.predict(df)

# Print the cluster labels
print(labels)

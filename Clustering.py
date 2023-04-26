#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:49:44 2023

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
    
    for i, row in feature_df[feature].iterrows():
        
        for val in row:
            if feature == 'item' or feature == 'make':
                raw_val = val.replace(" ", "")
            else:
                raw_val = val
                
            if raw_val not in unique_values[feature] and raw_val != '':
                unique_values[feature].append(raw_val)            

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

for row in range(len(df.index)):
    count_item[row] = {}
    price_item[row] = {}
    
    for item in unique_values['item']:
        count_item[row][item] = 0
        price_item[row][item] = 0
    
    for col in range(len(feature_df['item'].columns)):
        if feature_df['item'].iloc[row, col] != '':
            count_item[row][item] += feature_df['purchas'].iloc[row, col]
            price_item[row][item] += feature_df['price'].iloc[row, col]

for row in range(len(df.index)):
    count_make[row] = {}
    price_make[row] = {}
    
    for make in unique_values['make']:
        count_make[row][make] = 0
        price_make[row][make] = 0
    
    for col in range(len(feature_df['make'].columns)):
        if feature_df['make'].iloc[row, col] != '':
            count_make[row][make] += feature_df['purchas'].iloc[row, col]
            price_make[row][make] += feature_df['price'].iloc[row, col]
            


# Initialize the K-Means algorithm with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit the model to the data
kmeans.fit(df)

# Predict the cluster labels for each data point
labels = kmeans.predict(df)

# Print the cluster labels
print(labels)

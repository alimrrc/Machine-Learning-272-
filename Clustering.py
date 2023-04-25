#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:49:44 2023

@author: ali
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# Load the dataset from a CSV file
df = pd.read_csv('X_train_G3tdtEn.csv')

# Get the data types of all columns
dtypes = df.dtypes

cols = df.columns

# Get a list of columns that have object (string) data type
string_cols = list(dtypes[dtypes == "object"].index)
float_cols = list(dtypes[dtypes == "float"].index)

other_cols = [col for col in cols if col not in string_cols and col not in float_cols]
df.drop(columns=other_cols, inplace=True)

replacement_dict = {col: 'Nothing' for col in string_cols}
df.fillna(replacement_dict, inplace=True)

replacement_dict = {col: 0 for col in float_cols}
df.fillna(replacement_dict, inplace=True)

for col in string_cols:
    if "goods_code" in col:
        df.drop(col, axis=1, inplace=True)
        string_cols = [column for column in string_cols if column != col]
    else:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, int) else x)

# Create a OneHotEncoder object
encoder = OneHotEncoder()

# Fit the encoder to the categorical columns
encoder.fit(df[string_cols])

# Transform the categorical columns into one-hot encoded features
one_hot_encoded = encoder.transform(df[string_cols])

# Replace the original categorical columns with the one-hot encoded features
df.drop(columns=string_cols, inplace=True)

df = pd.concat([df, pd.DataFrame(one_hot_encoded.toarray(), columns=encoder.get_feature_names(string_cols))], axis=1)

# Initialize the K-Means algorithm with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit the model to the data
kmeans.fit(df)

# Predict the cluster labels for each data point
labels = kmeans.predict(df)

# Print the cluster labels
print(labels)

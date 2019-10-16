# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:33:29 2019

@author: David
"""

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor

def preprocess_data():
    training_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
    test_data = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
    
    training_data = training_data.dropna()

    full_data = pd.concat([training_data, test_data], sort=False)
    
    features = full_data[['Instance', 'Year of Record', 'Age', 'Gender', 'Country', 'Size of City', 'Profession', 'University Degree', 'Hair Color', 'Body Height [cm]' ]]
    
    labels = full_data['Income in EUR']

    features = encode_data(features, labels)
    
    train_features, test_features = split_features(features)
    
    train_features = train_features.drop(['Instance', 'index'], axis=1)
    test_features = test_features.drop(['Instance','index'], axis=1)
    
    labels = labels.dropna()
    train_labels = np.array(labels)
    
    train_features = scale_data(train_features)
    test_features = test_features.apply(lambda x: x.fillna(x.median()))
    test_features = scale_data(test_features)
   
    return train_features, test_features, train_labels

def encode_data(features, labels):    
    target_encoder = ce.TargetEncoder(cols=['Gender', 'Country', 'Profession', 'University Degree', 'Hair Color'])
    target_encoder.fit(features, labels)
    encoded_data = target_encoder.transform(features)
    return encoded_data.reset_index()
    
def split_features(features):
    for i, data in features.iterrows():
        if data['Instance'] == 111994:
            break
        index = i        
    train_features = features.iloc[:index+1, :]
    test_features = features.iloc[index+1:, :]
    return train_features, test_features

def scale_data(features):    
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)
    
def calculate_local_rmse(train_features, train_labels):
    train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_labels, test_size = 0.25, random_state = 42)
    cbr = train_model(train_features, train_labels)
    predictions = get_predictions(cbr, test_features)
    errors = predictions - test_labels
    print('Local RMSE CatBoost:', np.sqrt(((errors) ** 2).mean()))    

def train_model(features, labels):
    cbr = CatBoostRegressor(random_state=42, n_estimators=1000)
    cbr.fit(features, labels)
    return cbr    

def get_predictions(cbr, features):
    return cbr.predict(features)

def confirm_we_should_create_submission():
    response = input('Would you like to create a submission using this model? (y/n): ')
    if response == 'y' or response == 'Y':
        return True
    elif response == 'n' or response == 'N':
        return False
    else:
        print('Invalid user input, please try again...')
        return confirm_we_should_create_submission()

def train_predict_and_create_submission(train_features, test_features, train_labels):
    cbr = train_model(train_features, train_labels)
    predictions = get_predictions(cbr, test_features)
    create_submission(predictions)
    print('Submission Complete')
    print('Sample predictions (first 10): {}'.format(predictions[:10]))
    
def create_submission(predictions):
    submission_data = pd.read_csv('tcd ml 2019-20 income prediction submission file.csv')
    submission_data['Income'] = predictions
    submission_data.to_csv('tcd ml 2019-20 income prediction submission file.csv', encoding='utf-8', index = False)
    
    
if __name__=='__main__':
    print('Preprocessing data...')    
    train_features, test_features, train_labels = preprocess_data()
    print('Calculating local RMSE using CatBoostRegressor...')
    calculate_local_rmse(train_features, train_labels)
    if confirm_we_should_create_submission():
        train_predict_and_create_submission(train_features, test_features, train_labels)
    else:
        pass
    print('Progam Shutting Down')
    
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle

test_data = pd.read_csv("data/test.csv")

test_data = test_data[['id','Gender', 'Age',
       'Working Professional or Student', 'Academic Pressure',
       'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
       'Sleep Duration', 'Dietary Habits',
       'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
       'Financial Stress', 'Family History of Mental Illness', 'Depression']]

encoder = LabelEncoder()
test_data['Gender'] = encoder.fit_transform(test_data['Gender'])
test_data['Have you ever had suicidal thoughts ?'] = test_data['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
test_data['Family History of Mental Illness'] = test_data['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
test_data['Working Professional or Student'] = test_data['Working Professional or Student'].map({'Working Professional': 1, 'Student': 0})
test_data['Dietary Habits'] = encoder.fit_transform(test_data['Dietary Habits'])
test_data['Sleep Duration'] = encoder.fit_transform(test_data['Sleep Duration'])

test_data['CGPA'].fillna(test_data['CGPA'].mean(), inplace=True)
test_data['Work Pressure'].fillna(0, inplace=True)
test_data['Academic Pressure'].fillna(0, inplace=True)  # Assuming no pressure if missing
#test_data['Working Professional or Student'].fillna("unknown", inplace=True)
test_data['Study Satisfaction'].fillna(0, inplace=True)    
test_data['Financial Stress'].fillna(0, inplace=True)
test_data['Job Satisfaction'].fillna(0, inplace=True)

columns_to_combine = ['Age', 'Work/Study Hours', 'Financial Stress']

# Normalize the columns
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(test_data[columns_to_combine])

test_data['Combined_Stress_Feature'] = scaled_features.mean(axis=1)

features = ['Age',
       'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
       'Job Satisfaction','Dietary Habits',
       'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
       'Financial Stress']

data = test_data[features]
lgbm_path = "lgbm_model.pkl"
with open(lgbm_path, 'rb') as file:
    lgbm = pickle.load(file)
    
lgbm_pred = lgbm.predict(data)

test_data['Depression'] = lgbm_pred

test_data = test_data[['id', 'Depression']]
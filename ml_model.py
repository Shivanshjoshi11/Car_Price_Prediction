import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import warnings

# reading the csv file
df = pd.read_csv('data_train.csv')
df.head()
df = df.drop('New_Price', axis = 1)

df_raw = df.copy()

# separating the features and the label
data = df_raw.drop('Price', axis = 1)
data_labels = df_raw['Price']

# data preprocessing
def num_processing(data):
    col = ['Mileage', 'Engine', 'Power']
    data['Power'] = data['Power'].replace('null bhp',np.nan)
    for i in col:
        data[i]= data[i].apply(lambda x: list(str(x).split())[0]) 
        data[i] = data[i].replace({'nan': np.nan})
        data[i] = data[i].fillna(data[i].mean())
    data['Seats'] = data['Seats'].fillna(5).astype('int')
    data['Kilometers_Driven'] = data['Kilometers_Driven'].astype('float')
    
    data['Company'] = data.loc[:, 'Name'].apply(lambda x: list(x.split())[0])
    data = data.drop('Name', axis = 1)
    
    return data

# data transformation pipelines
def num_pipeline_transformer(data):
    num_attrs = ['Mileage', 'Engine', 'Power', 'Kilometers_Driven']
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    return num_attrs, num_pipeline


        

def pipeline_transformer(data):
    
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    cat_attrs = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Company']
    
    
    
    print(list(num_attrs))

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(sparse= False), list(cat_attrs)),
    ], remainder= 'passthrough')
    
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data, full_pipeline

processed_data = num_processing(data)
prepared_data, final_pipeline = pipeline_transformer(processed_data)

# training the model
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(prepared_data, data_labels)

final_model = forest_reg

# prediction function
def predict_price(config, model):
    global final_pipeline
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    prepared_df = final_pipeline.transform(df)
    print(prepared_df)
    y_pred = model.predict(prepared_df)
    return y_pred

import pickle
# loading the model in pickle
pickle.dump(final_model, open('model.pkl', 'wb'))

model= pickle.load(open('model.pkl', 'rb'))

# loading the final pipeline in pickle
pickle.dump(final_pipeline, open('final_pipeline.obj', 'wb'))

pipeline = pickle.load(open('final_pipeline.obj', 'rb'))

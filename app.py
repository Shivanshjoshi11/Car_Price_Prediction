import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from ml_model import * 
import pickle

app= Flask(__name__, template_folder= 'template')
model= pickle.load(open('model.pkl', 'rb'))

pipeline= pickle.load(open("final_pipeline.obj","rb"))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    input_features = [x for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    feature_name = ['Location', 'Year', 'Kilometers_Driven', 'Transmission'
    , 'Mileage', 'Engine', 'Power', 'Company']
    
    df = pd.DataFrame(features_value, columns= feature_name)
    
    for col in ['Year']:
        df[col] = df[col].astype('int')
    output = predict_price(df, model)
    return render_template('index.html', prediction_text= 'The Cost of The Car Should be Rs.{} Lakhs'.format(float(output)))
if __name__ == "__main__":
    app.run(debug= True)

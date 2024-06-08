from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('linear_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    airline = data['Airline']
    source = data['Source']
    destination = data['Destination']
    duration = float(data['Duration'])
    total_stops = int(data['Total_Stops'])

    # Creating a dataframe for the input features
    input_data = pd.DataFrame({
        'Airline': [airline],
        'Source': [source],
        'Destination': [destination],
        'Duration': [duration],
        'Total_Stops': [total_stops]
    })

    # Encoding the input features to match the training features
    input_data = pd.get_dummies(input_data, columns=['Airline', 'Source', 'Destination'], drop_first=True)
    
    # Ensuring all columns are present
    all_columns = model_columns()
    input_data = input_data.reindex(columns=all_columns, fill_value=0)
    
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted Price: {prediction:.2f}')

def model_columns():
    # List of columns used during training
    return [
        'Duration', 'Total_Stops', 
        'Airline_Airline_B', 'Airline_Airline_C',
        'Source_City_B', 'Source_City_C', 'Source_City_D',
        'Destination_City_F', 'Destination_City_G', 'Destination_City_H'
    ]

if __name__ == "__main__":
    app.run(debug=False, port = 8080, host="0.0.0.0")

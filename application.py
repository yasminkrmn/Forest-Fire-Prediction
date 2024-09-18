from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

# import logistic regression and standard scaler pickle
log_model = pickle.load(open('models/logregmodel.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        BUI =float(request.form.get('BUI'))
        FWI = float(request.form.get('FWI'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,FWI,Region]])
        result = log_model.predict(new_data_scaled)

        return render_template('home.html', results=result[0])

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0')
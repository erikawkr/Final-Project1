from multiprocessing.sharedctypes import Value
from optparse import Values
from flask import Flask, render_template, request, url_for, jsonify
import numpy as np
import pickle
import pandas as pd

# Load the pickled data from the file

file = open("model_LR.pkl", "rb")
file2 = open("model_dtr.pkl", "rb")
lr = pickle.load(file, encoding='latin1')
dtr = pickle.load(file2, encoding='latin1')
app = Flask(__name__, static_folder='static')

@app.route("/")
def main():
    return render_template("index.html")

# Desicion Tree
@app.route('/predict_1', methods=['POST'])
def predict_1():
    values = request.values

    X_new_dtr = {'transportation_name': [int(values['transport_name'])], 'type of car': [int(values['kind'])], 'start point': [int(values['start'])], 'end point': [int(values['finish'])], 'visibility': [int(values['visibility'])], 
                 'weather': [int(values['weather'])], 'distance': [int(values['distance'])], 'surge_multiplier': [int(values['surge_multiplier'])]}
    X_new_dtr = pd.DataFrame(X_new_dtr) 
    price_predict_dtr= dtr.predict(X_new_dtr)
    output_dtr = round(int(price_predict_dtr[0]), 2)
    prediction_txt_dtr ='Prediksi Tarif Decision Tree yaitu : $ {} Tingkat Akurasi Sebesar 94.62%'.format(output_dtr)

    return render_template('index.html', prediction_text_1='Prediksi Tarif Decision Tree yaitu : $ {} Tingkat Akurasi Sebesar 94.62%'.format(output_dtr))

    # return jsonify(prediction_txt_dtr)

@app.route('/predict_2', methods=['POST'])
def predict_2():
   
    # features_2 = [y for y in request.values()]
    values = request.values
    # final_features_2 = [int(values['transport_name']), int(values['kind']), int(values['start']), int(values['finish']), int(values['visibility']), int(values['weather']), int(values['distance']), int(values['surge_multiplier'])]
    # prediction_2 = rf.predict(final_features_2)

    X_new = {'transportation_name': [int(values['transport_name'])], 'type of car': [int(values['kind'])], 'start point': [int(values['start'])], 'end point': [int(values['finish'])], 'visibility': [int(values['visibility'])], 
             'weather': [int(values['weather'])], 'distance': [int(values['distance'])], 'surge_multiplier': [int(values['surge_multiplier'])]}
    X_new = pd.DataFrame(X_new) 
    price_predict_lr= lr.predict(X_new)
    output_2 = round(int(price_predict_lr[0]), 2)

    return render_template('index.html', prediction_text_2='Prediksi Tarif Linear Regression yaitu : $ {} Tingkat Akurasi Sebesar 52%'.format(output_2))

if __name__ == '__main__':
    app.run(debug=True)
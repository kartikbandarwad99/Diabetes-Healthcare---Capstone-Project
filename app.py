from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
from model import imputation,scaler

model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = np.array(int_features)
        #features=imputation(final_features)
        x=scale.transform(final_features.reshape(1,-1))
        prediction = model.predict(x)
        return render_template('result.html',prediction=prediction)
    

if __name__ == '__main__':
	app.run(debug=True)
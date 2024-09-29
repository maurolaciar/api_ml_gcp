from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_value = int(request.form['input_value'])
    single_feature = [input_value]
    prediction = model.predict([single_feature])
    prediction = prediction.tolist()
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

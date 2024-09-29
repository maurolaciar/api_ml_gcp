from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

@app.post('/api/predict')
async def predict(input: int):
    single_feature = [input]
    prediction = model.predict([single_feature])
    prediction = prediction.tolist()
    return prediction
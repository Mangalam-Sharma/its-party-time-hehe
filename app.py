import pickle

import numpy as np
from flask import Flask, request, jsonify

model=pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return "hello world"

@app.route("/predict",methods = ["POST"])
def predict():

    tags = request.form.get("tags")
    age = request.form.get("age")
    shift = request.form.get("shift")
    input_query = np.array([[tags,age,shift]])

    result=model.predict(input_query)[0]

    return jsonify({'time' : str(result)})
if __name__ == "__main__":
    app.run(debug=True)
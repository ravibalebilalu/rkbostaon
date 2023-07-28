import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("research/model.pkl","rb"))
sc = pickle.load(open("research/scaling.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api",methods = ["POST"])

def predict_api():
    data = request.json['data']
    data = np.array(list(data.values())).reshape(1,-1)
    sc_data = sc.transform(data)
    pred = model.predict(sc_data)
    return jsonify(pred[0])


if __name__ == "__main__":
    app.run(debug = True)
    #1:51



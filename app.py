
from flask import Flask,request,jsonify
import pickle
import numpy as np
app=Flask(__name__)

@app.route('/predict',methods=["post"])
def predict():
    data=request.get_json()
    data_list=data['features']
    data_np=np.array(data_list).reshape(1,-1)
    model=pickle.load(open("grid.pkl", "rb"))
    y_pred=model.predict(data_np)
    return jsonify({"p":y_pred.tolist()})


if __name__=="__main__":
    app.run(debug=True)
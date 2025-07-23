import logging
from flask import Flask,request,jsonify
import pickle
import numpy as np
app=Flask(__name__)

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s [%(levelname)s] %(message)s')
model = pickle.load(open(r"C:\Users\dumalaks\PycharmProjects\PyColorMap\FlaskProject\grid.pkl", "rb"))
@app.route('/predict',methods=["post"])
def predict():
    logging.info("Flask app is running")
    data=request.get_json()
    logging.info("Getting Data from the websever")
    data_list=data['features']
    data_np=np.array(data_list).reshape(1,-1)
    logging.info("fetching the data to train the model")

    logging.info("Loading the model")
    y_pred=model.predict(data_np)
    print(y_pred.tolist())
    logging.info("prediction output")
    return jsonify({"p":y_pred.tolist()})


if __name__=="__main__":
    app.run(debug=True)
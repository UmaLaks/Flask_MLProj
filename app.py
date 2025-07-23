"""from flask import Flask,request,url_for,redirect,render_template

app=Flask(__name__)

@app.route('/')
def home():
    return render_template(r"page.html",name="uma")

@app.route("/<name>")
def enterpage(name):
    return f"hello{name}"
@app.route("/success/<name>/<float:balance>")
def success(name,balance):
    return f"hello{name,balance}"

@app.route("/login",methods=["POST","GET"])
def login():
    if request.method=="POST":
        print("in hello")
        user=request.form["nm"]
        #return redirect(url_for("success",name=user))
        return render_template("page.html",name="uma")
    if request.method=="GET":
       # user=request.get["nm"]

        user=request.form.get('nm')

        #return redirect(url_for("success",name=user))
        return render_template(r"page.html", name="uma")



if __name__=="__main__":
    app.run(debug=True,port=50001)
"""

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
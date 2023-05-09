import numpy as np
from flask import Flask, request, render_template, redirect,url_for
import pickle

app=Flask(__name__)
model=pickle.load(open('heart\model\model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output=round(prediction[0],2)
    return redirect(url_for('score',output=output))

@app.route('/score/<int:output>')
def score(output):
    return render_template('result.html',result=output)

if __name__ == "__main__":
    app.run(debug=True)
import numpy as np
from flask import Flask, request, jsonify, render_template


#question1 = "Do they enjoy eating the dessert?"
#question2 = "Do they like hiking in the desert?"
# 1 means it is duplicated, 0 otherwise
#print(get_predict(question1 , question2, 0.7, model, vocab, verbose=True))


app = Flask(__name__)

from model import *

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    ques = [q for q in request.form.values()]
    prediction = get_predict(ques[0] , ques[1], 0.6, model, vocab, verbose=True)
    output = prediction[0]
    return render_template('index.html', prediction_text='Result : {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)

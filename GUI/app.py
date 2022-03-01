from flask import Flask, render_template, request, url_for, flash, redirect
import OurFunctions as F
import pickle
import numpy as np
import torch

tfidf = pickle.load(open("tfidf.pkl",'rb'))
CovidModel = torch.load('CovidNN.pkl')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sssjjjjjccccc'
output = ['']

@app.route('/',methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        tweet = request.form['tweet']
        if not tweet:
            output[0] = ''
        else:
            processedText = F.preprocess(request.form['tweet'])
            data = torch.tensor(tfidf.transform([processedText]).toarray()[0])
            modeloutput = CovidModel(data.float())
            FakeOrTrue = torch.argmax(modeloutput)
            if(FakeOrTrue == 0):
                output[0] = 'Fake'
            else :
                output[0] = 'Real'
            
    return render_template('index.html',result=output)



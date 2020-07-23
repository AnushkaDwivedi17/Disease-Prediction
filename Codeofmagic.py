# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:59:36 2020

@author: anushka
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, make_response
import json
from flask_cors import cross_origin

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World'

@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)
    res = processRequest(req)

    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def processRequest(req):
    result = req.get("queryResult")
    parameters = result.get("parameters")
    user_symptoms = parameters.get("Disease")
    
    symptom=np.zeros([526],dtype=float)
    finaldataset=pd.read_csv('finaldataset.csv')
    labels=finaldataset['prognosis']
    fdc=finaldataset
    fdc.drop('prognosis',axis=1,inplace=True)
    x_train,x_test,y_train,y_test=train_test_split(fdc,labels,test_size=0.25,random_state=20)
    model=MultinomialNB()
    model.fit(x_train,y_train)
    Alldiseases=model.classes_.tolist()
    
    indexes=[]
    for i in range(len(x_train.columns)):
        if x_train.columns[i] in user_symptoms:
            indexes.append(i)
        for i in indexes:
            symptom[i]=1
        top3=[]
        probab=model.predict_proba([symptom]).tolist()  
        for j in range(3):
            max=-10000000000
            h=0
            for i in range(len(probab[0])):
                if probab[0][i]>max:
                    max=probab[0][i]
                    h=i 
            k=[]
            k.append(Alldiseases[h])
            k.append(probab[0][h])
            top3.append(k)
            probab[0][h]=-1
            
    result=""
    for i in top3:
        result+="Probability of "+str(i[0])+"is "+"{:2.3f}".format((i[1]*100))+'%'+'\n'
    
    fulfillmentText = result
    return {
        "fulfillmentText": fulfillmentText
    }
if __name__ == '__main__':
    app.run()

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:59:36 2020

@author: anushka dwivedi
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, make_response
import json
from flask_cors import cross_origin

User_Symptoms=[]
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
    global User_Symptoms
    if req.get("queryResult").get("action") == "add_symptom":
        result = req.get("queryResult")
        parameters = result.get("parameters")
        ans=[]
        ans=ans+parameters.get("Disease")
        User_Symptoms=User_Symptoms+ans
        
    elif req.get("queryResult").get("action") == "add_symptom.no":
        user_symptoms = User_Symptoms
        
        symptom=np.zeros([132],dtype=float)
        finaldataset=pd.read_csv('Training.csv')
        finaldataset.columns=finaldataset.columns.str.replace('_',' ')
        labels=finaldataset['prognosis']
        fdc=finaldataset
        fdc.drop('prognosis',axis=1,inplace=True)
        x_train,x_test,y_train,y_test=train_test_split(fdc,labels,test_size=0.25,random_state=20)
        model=RandomForestClassifier()
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
            result+="Probability of "+str(i[0])+" is "+"{:2.3f}".format((i[1]*100))+'%'+'\n'
        User_Symptoms=[]
        fulfillmentText = result
        return {
            "fulfillmentText": fulfillmentText
        }
if __name__ == '__main__':
    app.run()

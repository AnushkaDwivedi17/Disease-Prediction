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
#from sklearn.neural_network import MLPClassifier
from flask import Flask, request, make_response
import json
from flask_cors import cross_origin
import pickle

User_Symptoms=[]
model_symptoms=['itching','skin rash','nodal skin eruptions','continuous sneezing','shivering','chills','joint pain','stomach pain','acidity','ulcers on tongue','muscle wasting','vomiting','burning micturition','spotting  urination','fatigue','weight gain','anxiety','cold hands and feets','mood swings','weight loss','restlessness','lethargy','patches in throat','irregular sugar level','cough','high fever','sunken eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish skin','dark urine','nausea','loss of appetite','pain behind the eyes','back pain','constipation','abdominal pain','diarrhoea','mild fever','yellow urine','yellowing of eyes','acute liver failure','fluid overload','swelling of stomach','swelled lymph nodes','malaise','blurred and distorted vision','phlegm','throat irritation','redness of eyes','sinus pressure','runny nose','congestion','chest pain','weakness in limbs','fast heart rate','pain during bowel movements','pain in anal region','bloody stool','irritation in anus','neck pain','dizziness','cramps','bruising','obesity','swollen legs','swollen blood vessels','puffy face and eyes','enlarged thyroid','brittle nails','swollen extremeties','excessive hunger','extra marital contacts','drying and tingling lips','slurred speech','knee pain','hip joint pain','muscle weakness','stiff neck','swelling joints','movement stiffness','spinning movements','loss of balance','unsteadiness','weakness of one body side','loss of smell','bladder discomfort','foul smell of urine','continuous feel of urine','passage of gases','internal itching','toxic look (typhos)','depression','irritability','muscle pain','altered sensorium','red spots over body','belly pain','abnormal menstruation','dischromic  patches','watering from eyes','increased appetite','polyuria','family history','mucoid sputum','rusty sputum','lack of concentration','visual disturbances','receiving blood transfusion','receiving unsterile injections','coma','stomach bleeding','distention of abdomen','history of alcohol consumption','fluid overload.1','blood in sputum','prominent veins on calf','palpitations','painful walking','pus filled pimples','blackheads','scurring','skin peeling','silver like dusting','small dents in nails','inflammatory nails','blister','red sore around nose','yellow crust ooze']

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
        model=pickle.load(open('rfcmodel.pkl','rb'))
        Alldiseases=model.classes_.tolist()

        indexes=[]
        for i in range(len(model_symptoms)):
            if model_symptoms[i] in user_symptoms:
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

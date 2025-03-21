# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template

# Flask constructor takes the name of 
# current module (__name__) as argument.
application = Flask(__name__)
app=application

#import pickle files
scaler_model = pickle.load(open('models/scaler.pkl','rb'))
reg_model= pickle.load(open('models/ridgemodel.pkl','rb'))


# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/', methods=['GET','POST'])
# ‘/’ URL is bound with hello_world() function.
def predict_data():
    if request.method == 'POST':
        temperature=float(request.form.get('temperature'))
        RH=float(request.form.get('RH'))
        WS=float(request.form.get('WS'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled = scaler_model.transform([[temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        reuslt = reg_model.predict(new_data_scaled)
        return render_template("index.html", results=reuslt[0])
    else:
        return render_template("index.html")




# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(host='0.0.0.0', port='8080')
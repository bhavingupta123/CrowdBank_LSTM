#import libraries
import numpy as np
from flask import Flask, redirect, request, jsonify, render_template
#import pickle
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import random

#Initialize the flask App
app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
model=load_model('lstm.h5')

d=[]

#default page of our web-app
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():

    '''
    For rendering results on HTML GUI
    '''
    #print(request.form.values())
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    #output = round(prediction[0], 2)

    d=[[x for x in request.form.values()]]
    df = pd.DataFrame(d, columns = ["check_acc","mon","credit_his","purpose","Credit_amo","saving_amo","Pre_employ","installrate","p_status","guatan","pre_res","property","age","installment","Housing","existing_cards","job","no_people","telephn","for_work"])
    
    df['mon'] = df['mon'].astype(np.int64)
    df['Credit_amo']=df['Credit_amo'].astype(np.int64)
    df['installrate']=df['installrate'].astype(np.int64)
    df['pre_res']=df['pre_res'].astype(np.int64)
    df['age']=df['age'].astype(np.int64)
    df['existing_cards']=df['existing_cards'].astype(np.int64)
    df['no_people']=df['no_people'].astype(np.int64)

    check_acc_categories=['A11','A12','A13','A14']
    credit_his_categories=['A30','A31','A32','A33','A34']
    purpose_categories=['A40','A41','A42','A43','A44','A45','A46','A47','A48','A49','A410']
    saving_amo_categories=['A61','A62','A63','A64','A65']
    Pre_employ_categories=['A71','A72','A73','A74','A75']
    p_status_categories=['A91','A92','A93','A94','A95']
    guatan_categories=['A101','A102','A103']
    property_categories=['A121','A122','A123','A124']
    installment_categories=['A141','A142','A143']
    Housing_categories=['A151','A152','A153']
    job_categories=['A171','A172','A173','A174']
    telephn_categories=['A191','A192']
    for_work_categories=['A201','A202']
    df['check_acc'] = df['check_acc'].astype(pd.CategoricalDtype(categories= check_acc_categories))
    df['credit_his'] = df['credit_his'].astype(pd.CategoricalDtype(categories= credit_his_categories))
    df['purpose'] = df['purpose'].astype(pd.CategoricalDtype(categories= purpose_categories))
    df['saving_amo'] = df['saving_amo'].astype(pd.CategoricalDtype(categories= saving_amo_categories))
    df['Pre_employ'] = df['Pre_employ'].astype(pd.CategoricalDtype(categories= Pre_employ_categories))
    df['p_status'] = df['p_status'].astype(pd.CategoricalDtype(categories= p_status_categories))
    df['guatan'] = df['guatan'].astype(pd.CategoricalDtype(categories= guatan_categories))
    df['property'] = df['property'].astype(pd.CategoricalDtype(categories= property_categories))
    df['installment'] = df['installment'].astype(pd.CategoricalDtype(categories= installment_categories))
    df['Housing'] = df['Housing'].astype(pd.CategoricalDtype(categories= Housing_categories))
    df['job'] = df['job'].astype(pd.CategoricalDtype(categories= job_categories))
    df['telephn'] = df['telephn'].astype(pd.CategoricalDtype(categories= telephn_categories))
    df['for_work'] = df['for_work'].astype(pd.CategoricalDtype(categories= for_work_categories))
    dummies = pd.get_dummies(df, columns=['check_acc','credit_his','purpose','saving_amo','Pre_employ','p_status','guatan','property','installment','Housing','job','telephn','for_work'])
    
    X=np.array(dummies)
    print(type(X))
#print(X.shape)
    
    print(X)
    
    scaler_train=joblib.load('scaler.save')
    b=scaler_train.transform(X)
    print(b)

    b=np.reshape(b,(b.shape[0],1,b.shape[1]))
    
    ans=model.predict(b)
    ans=1000-(ans*1000)
    if(ans>900):
        ans=ans-random.randint(150,200)
    elif ans<100:
        ans=ans+random.randint(150,200)

    #return '{}'.format(int(ans))
    #data = {'username': 'Pang', 'site': 'stackoverflow.com'}
    #return render_template('',prediction_text='Credit Score : {} '.format(int(ans)))
    return redirect("http://localhost:8080/borrower.html?cs={}".format(int(ans)))


if __name__ == "__main__":
    app.run(debug=True)
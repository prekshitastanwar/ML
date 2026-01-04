import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import precision_score,recall_score,f1_score
import numpy as np
ZIP=("C:/Users/Prekshita/Downloads/archive.zip")
with zipfile.ZipFile(ZIP,'r') as z:
    file=z.namelist()[0]
    spam=pd.read_csv(z.open(file))

spam = spam[spam['Category'] != '{"mode":"full"']
train,test=train_test_split(spam,test_size=0.3,random_state=42)
X_train=train['Message']
X_test=test['Message']
y_train=train['Category']
y_test=test['Category']

spam_keywords=['free','prize','membership','money','cash','credit','link','expire','expires',
'subscription','charged','confirm','urgent','complimentary','claim','msg',
'won','code','password','contact','private','bonus','bonus','points','unredeemed','won',
'txt','text','ans','ansr','join','verify','customer','service','operator','selected','winner',
'stop','reply','inviting','guaranteed','ur','call','awarded','draw','wkly','offer',
'unlimited','date','upgrade','loyalty','apply','income','trial',
'refund','debt','paid','instant','limited','click','here','discount','cashback','exclusive',
'risk','lottery','award','special','hurry','unlocked','activate','processed','deposit','pre',
'dload','vid','results','unsubscribe','delivered','subscriber','lucky','surprise','sale',
'receipt','chat','matched','sex','sexy','hot','babe','babes','match','area','local','alone',
'hotel','optout','entry','/','double','latest']

def email_to_vector(text):
    text=text.lower()
    vector=np.array(text.split())
    return vector
def score(text):
    vector=email_to_vector(text)
    score=0
    for i in vector:
        if i in spam_keywords:
            score+=1
    return score
#Function should be applied to array
def email_score(text_column):
    return text_column.apply(score).values.reshape(-1,1)
rfc=RandomForestClassifier(random_state=42,class_weight='balanced')
email_pipeline=make_pipeline(FunctionTransformer(email_score,validate=False),rfc)
spam_prepared=email_pipeline.fit(X_train,y_train)
kfolds=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
prediction=cross_val_predict(email_pipeline,X_train,y_train,cv=kfolds)
prediction_score=cross_val_score(email_pipeline,X_train,y_train,cv=kfolds,scoring='accuracy')
print("Prediction Accuracy")
print(prediction_score)
print("Precision Score")
print(precision_score(prediction,y_train,pos_label='spam'))
print("Recall Score")
print(recall_score(prediction,y_train,pos_label='spam'))
print("F1 Score")
print(f1_score(y_train,prediction,pos_label='spam'))

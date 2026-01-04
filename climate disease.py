import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

data=pd.read_csv("C:/Users/Prekshita/Downloads/climate_disease_dataset.csv")
print(data.columns)
y1=pd.qcut(data['malaria_cases'],q=2,labels=[0,1])
X=data.drop(columns=['malaria_cases','dengue_cases'],axis=1)
data['']
X_train,X_test,y1_train,y1_test=train_test_split(X,y1,test_size=0.25,random_state=42,stratify=y1)
ohc=OneHotEncoder()
features=['country','region']
preprocessing=ColumnTransformer(
    [('cat',OneHotEncoder(),features)])

rfc = RandomForestClassifier(random_state=42,class_weight='balanced')

pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('classifier', rfc)])
y_pred = cross_val_predict(pipeline,X_train,y1_train,cv=5)
print(y1.value_counts())
print(classification_report(y1_train, y_pred,zero_division=0))

X_train=preprocessing.fit_transform(X_train)
X_test=preprocessing.transform(X_test)
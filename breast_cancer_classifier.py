import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
cancer_data=pd.read_csv("C:/Users/Prekshita/Downloads/Cancer_Data.csv")
cancer_data=cancer_data.drop(columns='Unnamed: 32',axis=1)
cancer_data['diagnosis'] = cancer_data['diagnosis'].map({'M': 1, 'B': 0})
target=cancer_data['diagnosis']
data=cancer_data.drop('diagnosis',axis=1)

X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.25,random_state=42,stratify=target)
features=cancer_data.select_dtypes(include=[np.number])
model_1=make_pipeline(PCA(random_state=42),RandomForestClassifier(random_state=42))

params={"randomforestclassifier__n_estimators":np.arange(100,500),
        "randomforestclassifier__min_samples_split":np.arange(10,30),
        "pca__n_components":np.arange(2,30)}
random_search=RandomizedSearchCV(model_1,params,cv=3,random_state=10,n_iter=10)
random_search.fit(X_train,y_train)
model=random_search.best_estimator_

def cross_validation(model):
    skfold=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
    score=0
    for train_index,test_index in skfold.split(X_train,y_train):
        X_train_folds=X_train.iloc[train_index]
        X_test_folds=X_train.iloc[test_index]
        y_train_folds=y_train.iloc[train_index]
        y_test_folds=y_train.iloc[test_index]
        model.fit(X_train_folds,y_train_folds)
        y_pred=model.predict(X_test_folds)
        n_correct=sum(y_pred==y_test_folds)
        score+=n_correct/len(y_pred)
    print("Report:",score/5)

cross_validation(model)
y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))
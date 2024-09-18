#################################################################
# Forest Fires Prediction - MODEL
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle

#---------------------------- 1. Importing Dataset ---------------------------
df = pd.read_csv("forest_fires_cleaned_dataset.csv")

#---------------------------- 2. Modelling ---------------------------
X = df.drop('Classes', axis=1)
y = df['Classes']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

# Standard Scaling

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


X_train.shape, X_test_scaled.shape
y_train.shape, y_test.shape

logistic=LogisticRegression()
logistic.fit(X_train,y_train)
y_pred=logistic.predict(X_test_scaled)
print(y_pred)

score=accuracy_score(y_pred,y_test)
print(score)
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))

#---------------------------- 3. Hyperparameter Tuning And Cross Validation ---------------------------
# Grid SearchCV
model=LogisticRegression()
penalty=['l1', 'l2', 'elasticnet']
c_values=[100,10,1.0,0.1,0.01]
solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
params={'C':c_values,'penalty':penalty,'solver':solver}
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold()
grid=GridSearchCV(estimator=model,param_grid=params,scoring='accuracy',cv=cv,n_jobs=-1)
grid

grid.fit(X_train,y_train)

grid.best_params_ # {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg'}

print(grid.best_score_) # 0.98

y_pred=grid.predict(X_test_scaled)

score=accuracy_score(y_pred,y_test)
print(score) # 0.45
print(classification_report(y_pred,y_test))
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))


# Randomized SearchCV

model=LogisticRegression()
randomcv=RandomizedSearchCV(estimator=model,param_distributions=params,cv=5,scoring='accuracy')
randomcv.fit(X_train,y_train)
randomcv.best_params_ # {'solver': 'liblinear', 'penalty': 'l2', 'C': 1.0}
print(randomcv.best_score_) #0.97
y_pred=randomcv.predict(X_test_scaled)
score=accuracy_score(y_pred,y_test)
print(score) # 0.77
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))

# Pickle Model
# pickle.dump(randomcv, open('logregmodel.pkl','wb'))
# pickle.dump(scaler, open('scaler.pkl', 'wb'))

#with open('logregmodel.pkl', 'rb') as f:
#    loaded_classifier = pickle.load(f)

#loaded_classifier.predict(X_test_scaled)
#accuracy_score(y_test, loaded_classifier.predict(X_test_scaled))

#with open('scaler.pkl', 'rb') as f:
#    loaded_scaler = pickle.load(f)

#loaded_scaler.transform(X_test)


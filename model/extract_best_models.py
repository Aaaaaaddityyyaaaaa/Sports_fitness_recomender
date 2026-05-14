import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score ,r2_score
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_excel("5000dataset.xlsx")


y=df["Performance_Metric"]
x= df.drop(labels=["Athlete_ID","Performance_Metric"] , axis=1)

xtrain  , xtest , ytrain ,ytest =  train_test_split(x,y,train_size=0.8,random_state=42)



scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

joblib.dump(scaler,"model/artifact/scaler.pkl")

print("preprocessedand saved scaler")


lr = LinearRegression()
lr.fit(xtrain, ytrain)

print("trained now evaluating") 

ypredicted = lr.predict(xtest)



r2 = r2_score(ytest , ypredicted)

joblib.dump(lr,"model/artifact/my_best_model.pkl")

print("saved model")


print(f"R2 score : {r2}")
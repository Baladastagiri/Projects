import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("D:/Downloads/emp_sal.csv")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.svm import SVR
svr_regressor=SVR(kernel='poly',degree=4,gamma='scale',C=10.0)
svr_regressor.fit(x,y)

svr_model_pred=svr_regressor.predict([[6.5]])
print(svr_model_pred)

#KNN
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model=KNeighborsRegressor(n_neighbors=5,weights='distance',leaf_size=30)
knn_reg_model.fit(x,y)

knn_reg_pred=knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

#Decission Tree

from sklearn.tree import DecisionTreeRegressor
dtr_reg_model=DecisionTreeRegressor()
dtr_reg_model.fit(x,y)

dtr_reg_pred=dtr_reg_model([[6.5]])
print(dtr_reg_pred)

#Random forest
from sklearn.ensemble import RandomForestRegressor
rfr_reg_model=RandomForestRegressor(n_estimators=7,random_state=0)
rfr_reg_model.fit(x,y)

rfr_reg_pred=rfr_reg_model.predict([[6.5]])
print(rfr_reg_pred)

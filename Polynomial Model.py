import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"d:\Downloads\emp_sal.csv")

x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_model_pred=lin_reg.predict([[6.5]])
print(lin_model_pred)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=6)
x_poly=poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



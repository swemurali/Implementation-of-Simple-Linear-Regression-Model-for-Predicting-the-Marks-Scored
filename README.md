# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.
2.Assign hours to X and scores to Y.
3.Implement training set and test set of the dataframe. 4.Plot the required graph both for test data and training data. 5.Find the values of MSE , MAE and RMSE. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.Suwetha
RegisterNumber:  212221230112
*/
import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![193340375-75a3d817-aa58-4715-b7b0-91185d6b8ff1](https://user-images.githubusercontent.com/94165336/197789061-a95223c8-028d-42f6-927e-740fb46226aa.png)
![193340377-682adf7d-ad7f-4fed-a700-910b4b5daaae](https://user-images.githubusercontent.com/94165336/197789111-a90533a0-3c4a-4e6c-b0f3-0c6d1a977e86.png)
![193340380-7f61afa0-cf71-4c91-aee6-1fffddd0eb13](https://user-images.githubusercontent.com/94165336/197789217-3e9b33d2-6f07-44b8-aa74-c3b2c4897279.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

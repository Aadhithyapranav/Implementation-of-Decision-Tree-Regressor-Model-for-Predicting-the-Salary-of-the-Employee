# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. OPEN jupyter notebook
2. import the  modules
3. write the code and generate the regressor model for the salary of the employee 

   

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: AADHITHYAA L
RegisterNumber: 212224220003
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()

x=df[["Position","Level"]]
x.head()

y=df["Salary"]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Aadhithyaa L")
print("212224220003")
print(y_pred)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean squared error:",mse)
print("root mean squared error:",rmse)
print("mean absolute error:",mae)
print("r2 score",r2)

dt.predict(pd.DataFrame([[5,6]],columns=["Position","Level"]))
 
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="659" height="297" alt="Screenshot 2025-09-22 133455" src="https://github.com/user-attachments/assets/6c64c6bf-c791-4326-9114-14bf7bd14798" />
<img width="603" height="332" alt="Screenshot 2025-09-22 133513" src="https://github.com/user-attachments/assets/56f1d806-514d-4554-84ad-60cfab52dcbf" />
<img width="494" height="264" alt="Screenshot 2025-09-22 133529" src="https://github.com/user-attachments/assets/44421e4a-19b3-43f0-9da4-7dcb8cd24b35" />
<img width="429" height="214" alt="Screenshot 2025-09-22 133542" src="https://github.com/user-attachments/assets/56056922-1bee-4b84-9af3-9620d6557fbb" />
<img width="885" height="325" alt="Screenshot 2025-09-22 133553" src="https://github.com/user-attachments/assets/a4674f82-0dd3-4ff2-b4a8-80f2e14f6ee0" />
<img width="857" height="446" alt="Screenshot 2025-09-22 133610" src="https://github.com/user-attachments/assets/8b5176b1-a4f6-4c14-b0b5-c519cfe6faad" />









## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```

Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: K Deekshitha
RegisterNumber: 2305002005

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
x_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_train,lr.predict(x_train),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
X_test
Y_test
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test, pred)
print(f'Mean Squared Error (MSE): {mse}')



```

## Output:

![image](https://github.com/user-attachments/assets/e301958b-fb89-47f8-9923-fae59a6c5aa7)
![image](https://github.com/user-attachments/assets/15410534-ff5a-4b5c-a060-a2ef170a0946)
![image](https://github.com/user-attachments/assets/9b5ed8c3-861c-4173-83a9-52cee4e7f44b)
![image](https://github.com/user-attachments/assets/85910581-f498-40ce-bcbf-f517d37a7727)
![image](https://github.com/user-attachments/assets/d51e5eb2-ab39-4643-b587-114b4391c3f2)
![image](https://github.com/user-attachments/assets/19713c0a-d918-464c-bdab-abad0b9dce11)
![image](https://github.com/user-attachments/assets/8c82ff0c-ef2d-472b-8e8e-2321100c2d2b)
![image](https://github.com/user-attachments/assets/cd4866ad-770c-4d39-a012-79cd62662827)
![image](https://github.com/user-attachments/assets/37d13a03-6fe3-467f-a943-9c3b486a8f78)
![image](https://github.com/user-attachments/assets/05e04ccd-4ea4-45a1-b9d9-c242191fdd51)
![image](https://github.com/user-attachments/assets/5ac1b0b2-873a-4f57-96a7-a60a0b7fe497)

## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.

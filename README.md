# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas as pd.

### Step2
Read the csv file.


### Step3
Get the value of X and y variables

### Step4
Create the linear regression model and fit.
### Step5
Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.
## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# Load the Boston dataset
boston = fetch_openml(name="boston", version=1, as_frame=False)

# Defining feature matrix (X) and response vector (y)
X = boston.data
y = boston.target.astype(float)

# Splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(X_train, y_train)

# Regression coefficients
print("Coefficients:\n", reg.coef_)

# Variance score (R^2 score)
print("Variance score: {:.2f}".format(reg.score(X_test, y_test)))

# Setting plot style
plt.style.use("fivethirtyeight")

# Plotting residual errors in training data
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green",
    s=10,
    label="Train data"
)

# Plotting residual errors in test data
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue",
    s=10,
    label="Test data"
)

# Plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

# Plot legend
plt.legend(loc="upper right")

# Plot title
plt.title("Residual Errors")

# Show the plot
plt.show()
```
## Output:
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/ae50c310-8491-4054-a8f3-ddd7680d51b1" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/8ed43215-5397-41e9-a523-09429a482513" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/259ed7b0-8bc3-4abe-9e5a-941f0fa55e9e" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.

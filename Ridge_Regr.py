import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Cleaned and setup through R 
df=np.loadtxt('/Users/pinesa/Desktop/subdf.csv',delimiter=',')

# Divide to predict first column
# sep to predict first column
newY=cbcl[:,0]
newX=cbcl[:,1:49]

# Train and test split from data frame
xtrain,xtest,ytrain,ytest=train_test_split(newX,newY,test_size=0.3,random_state=3)

# linear regression
lr = LinearRegression()
lr.fit(xtrain,ytrain)
pred_train_lr=lr.predict(xtrain)
print(np.sqrt(mean_squared_error(ytrain,pred_train_lr)))
print(r2_score(ytrain, pred_train_lr))

pred_test_lr= lr.predict(xtest)
print(np.sqrt(mean_squared_error(ytest,pred_test_lr))) 
print(r2_score(ytest, pred_test_lr))

# ridge regression
# may want to play with the alpha, the proper way would be with nested cross-validation
rr = Ridge(alpha=0.001)
rr.fit(xtrain, ytrain) 
pred_train_rr= rr.predict(xtrain)
print(np.sqrt(mean_squared_error(ytrain,pred_train_rr)))
print(r2_score(ytrain, pred_train_rr))

pred_test_rr= rr.predict(xtest)
print(np.sqrt(mean_squared_error(ytest,pred_test_rr))) 
print(r2_score(ytest, pred_test_rr))

# Assess coefficient weights from this "weights" file
np.savetxt('weights', rr.coef_, delimiter=',')


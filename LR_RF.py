#ML Project to predict the value of Y
#logS = y
#other columns are x
#We are going to use 4 vairables to predict on the logS

#Load Data
import pandas as pd
df=pd.read_csv(r"C:\Users\God\Downloads\gitdemo\first_ml_solubility\delaney_solubility_with_descriptors.csv")
# print(df.head())

##Data Preparation: Will make logS as Y and others as X
###Data Separation as X and Y

y=df['logS']
# print(y)

# Get all X by dropping logS
x=df.drop('logS', axis=1)
# print(x)


###Data splittinng in training and testing set using sklearns

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=100)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

## **Model Building**
###  **Linear Regression**

####Training the model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
# print(lr)

####Applying a model to make a prediction

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred  = lr.predict(x_test)

####Evalue Model Performance
from sklearn.metrics import mean_squared_error, r2_score
lr_train_men = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_men = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# print('LR MEN(TRAIN)', lr_train_men)
# print('LR R2(TRAIN)', lr_train_r2)
# print('LR MEN(TEST)', lr_test_men)
# print('LR R2(TEST)', lr_test_r2)

lr_results=pd.DataFrame(['Linear Regression:',lr_train_men, lr_train_r2,  lr_test_men, lr_test_r2]).transpose()
lr_results.columns=['Method', 'Training MEA', 'Training R2', 'Testing MEA', 'Testing R2']
# print(lr_results)


##Random Forest
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_test, y_train = train_test_split(x,y,test_size=0.2, random_state=100)

#Model Training
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

#Applying model to make prediction
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred  = rf.predict(x_test)

##Evaluate model peformance
from sklearn.metrics import mean_squared_error, r2_score
rf_train_men = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2  = r2_score(y_train, y_rf_train_pred)

rf_test_men  =   mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2   =   r2_score(y_test, y_rf_test_pred)

rf_results=pd.DataFrame(['Random Forest Regression:',rf_train_men, rf_train_r2,  rf_test_men, rf_test_r2]).transpose()
rf_results.columns=['Method', 'Training MEA', 'Training R2', 'Testing MEA', 'Testing R2']
# print(rf_results)

df_model = pd.concat([lr_results, rf_results], axis=0)
df_model = df_model.reset_index(drop=True)
print(df_model)

##Data Visualization of prediction results
import matplotlib.pyplot as plt
import numpy as np

# plt.figure(figsize=(5,5))
# plt.scatter(x*y_train, y*y_lr_train_pred, alpha=0.3, c='#7CAE05')
# z=np.polyfit(y_train, y_lr_train_pred, 1)
# p=np.poly1d(z)
# plt.plot(y_train, p(y_train), 'Y17640')
# plt.ylabel('Predict logS')
# plt.xlabel('Experimental logS')

plt.scatter(x*y_train, y*y_lr_train_pred)
plt.plot()

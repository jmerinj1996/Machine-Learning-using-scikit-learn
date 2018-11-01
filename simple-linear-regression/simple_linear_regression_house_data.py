#simple linear regression on kc_house_data

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

#import dataset into a pandas dataframe
df = pd.read_csv('kc_house_data.csv')

df.drop(['id','date'],1,inplace = True)
df.fillna(-99999, inplace = True)

X = np.array(df.drop(['price'],1))
X = preprocessing.scale(X)

#the last 20 records are kept to perform prediction
for_prediction = X[-20:]
X = X[:-20]

y = np.array(df['price'])
prediction_df = pd.DataFrame(y[-20:],columns=['Actual y'])
y = y[:-20]

#Cross-validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression(n_jobs = -1)

#Train
clf.fit(X_train,y_train)

#Test
accuracy = clf.score(X_test,y_test)
print('Accuracy:',accuracy)

#Predict
prediction_set = clf.predict(for_prediction)
prediction_df['Predicted y'] = prediction_set

#Visualization that compares the predicted price to the actual price
plt.plot(prediction_df.index,prediction_df['Actual y'],label = 'Actual y')
plt.plot(prediction_df.index,prediction_df['Predicted y'],label = 'Predicted y')
plt.legend(loc = 4)
plt.show()

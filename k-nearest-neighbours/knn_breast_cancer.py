# knn on breast cancer data

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace = True)

#drop the id column because it does not contribute anything
df.drop(['id'],1, inplace = True)

#Define X and y
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)

#train the classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

#test the classifier
accuracy = clf.score(X_test, y_test)
print(accuracy)

#predict
predict = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
predict = predict.reshape(len(predict),-1)
prediction = clf.predict(predict)
print(prediction)

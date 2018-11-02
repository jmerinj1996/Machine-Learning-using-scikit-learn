
Here we are using a simple housing dataset and performing linear regression on it.
Although we can infer from the result that linear regression may not be the best option for this data, this is just to give you an idea as to how easy it is to use scikit learn on real world datasets.
### Things to note:
* The dataset used here is the kc_house data from Kaggle
* The only two columns dropped are the date and the id column since they do not contribute to the classification.
* Preprocessing the featureset made the accuracy go higher.
* Before passing X, y to "fit", we need to covert them to arrays. Python does not have built-in arrays so we use the ones provided by the numpy module.
* The last 20 records were seperated for prediction.
* The accuracy on using a simple linear regression algorithm on the dataset is about 70%

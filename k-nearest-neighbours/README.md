Here we are using breast cancer data obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolber and performing k nearest neigbours on it
### Things to note:
* The dataset used here is breast cancer data
* The only column dropped is the id column since it do not contribute to the classification, infact it lowers the accuracy with included.
* Before passing X, y to "fit", we need to covert them to arrays. Python does not have built-in arrays so we use the ones provided by the numpy module.
* The accuracy on using a knn algorithm on the dataset is about 96%


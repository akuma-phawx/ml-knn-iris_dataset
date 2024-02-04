import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Example of binary classification using the cancer dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

neighbors_settings = range(1,11)
training_accuracy = []
test_accuracy = []

for n_neighbors in neighbors_settings:
    # Build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Fit the model
    clf.fit(X_train, y_train)
    # Record training accuracy
    training_accuracy.append(clf.score(X_train,y_train))
    # Recort generalization accuracy
    test_accuracy.append(clf.score(X_test,y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_linear_regression_wave()

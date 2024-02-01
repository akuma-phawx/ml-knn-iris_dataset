import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    iris_dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset.data, iris_dataset.target, random_state=0
    )
    # train the model
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # make a prediction
    X_new = np.array([[5, 2.9, 1, 0.2]])
    prediction = knn.predict(X_new)
    print(prediction)
    print(f"Prediction: {iris_dataset['target_names'][prediction[0]]}")

    # model evaluation
    y_pred = knn.predict(X_test)
    print('Test set predictions:\n', y_pred)
    print('Test set actual values:\n', y_test)
    print('Test set score {:.2f}'.format(knn.score(X_test, y_test)))
    print('Test set score {:.2f}'.format(np.mean(y_pred == y_test)))

if __name__ == "__main__":
    main()

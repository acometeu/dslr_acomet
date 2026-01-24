import numpy as np
import logreg_train
import utils


def predict_proba(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return logreg_train.sigmoid(X_b @ theta)


# def predict(X, theta, treshold=0.5):
#     return (predict_proba(X, theta) >= treshold)


def logreg_predict(thetas, my_data, results):
    hogwarts_house_dict = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]

    probas = [x for x in range(len(hogwarts_house_dict))]
    for i in range(len(hogwarts_house_dict)):
        probas[i] = predict_proba(my_data, thetas[i])

    predics = np.empty((results.shape), dtype=type(results))
    for j in range(len(my_data)):
        best_proba = 0
        index = 0
        for i in range(len(thetas)):
            if (probas[i][j] >= best_proba):
                best_proba = probas[i][j]
                index = i
        predics[j] = hogwarts_house_dict[index]
    
    utils.compare_lr_results(results, predics)


def main():
    pass


if (__name__ == "__main__"):
    main()
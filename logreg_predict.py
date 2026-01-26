import numpy as np
import pandas as pd
import sys
import logreg_train
import data_parsing



def predict_proba(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return logreg_train.sigmoid(X_b @ theta)


def logreg_predict(data_csv, weights_csv, parsing_method=data_parsing.replace_nan_value_by_0):

    thetas = pd.read_csv(weights_csv)
    my_data = data_parsing.get_students_scores(data_csv, parsing_method)
    results = data_parsing.get_student_houses(data_csv)

    hogwarts_house_dict = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    houses = pd.DataFrame(columns=['Hogwarts House'])


    probas = [x for x in range(len(hogwarts_house_dict))]
    for i in range(len(hogwarts_house_dict)):
        probas[i] = predict_proba(my_data, thetas[hogwarts_house_dict[i]])
        # probas[i] = predict_proba(my_data, thetas[i])

    predics = np.empty((results.shape), dtype=type(results))
    for j in range(len(my_data)):
        best_proba = 0
        index = 0
        for i in range(len(probas)):
            if (probas[i][j] >= best_proba):
                best_proba = probas[i][j]
                index = i
        predics[j] = hogwarts_house_dict[index]
        houses.loc[j] = hogwarts_house_dict[index]
    
    houses.index.name = 'Index'
    houses.to_csv('houses.csv', index=True)


def main():
    args = sys.argv
    print (args)
    
    logreg_predict('dataset_train.csv', 'weights.csv')


if (__name__ == "__main__"):
    main()
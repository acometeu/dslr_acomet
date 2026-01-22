import pandas as pd
import numpy as np
import logistic_regression
import logistic_regression_utils
import data_parsing


def main():
    # parse data
    data_pd = pd.read_csv('dataset_train.csv')
    data_pd.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1, inplace=True)
    hogwarts_house_dict = {"Gryffindor": 0, "Slytherin": 1, "Hufflepuff": 2, "Ravenclaw": 3}
    data_pd = data_parsing.replace_nan_value(data_pd, hogwarts_house_dict)

    # get results 
    results = data_pd['Hogwarts House']
    data_pd.drop('Hogwarts House', axis=1, inplace=True)

    # transform in numpy
    my_data = data_pd.to_numpy(dtype=np.float64)
    logistic_regression_utils.normalise_data(my_data)


    # generate all thetas
    hogwarts_house_dict_inv = {v: k for k, v in hogwarts_house_dict.items()}
    thetas = [x for x in range(len(hogwarts_house_dict_inv))]
    for i in range(len(hogwarts_house_dict_inv)):
        binomial_results = results.copy()
        binomial_results[binomial_results != hogwarts_house_dict_inv[i]] = 0
        binomial_results[binomial_results == hogwarts_house_dict_inv[i]] = 1
        binomial_results = binomial_results.to_numpy(dtype=np.float64)
        thetas[i] = logistic_regression.gradient_descent(my_data, binomial_results)


    # thetas = get 

    

    # test if predics good
    probas = [x for x in range(len(hogwarts_house_dict_inv))]
    for i in range(len(hogwarts_house_dict_inv)):
        probas[i] = logistic_regression.predict_proba(my_data, thetas[i])

    predics = np.empty((results.shape), dtype=type(results))
    for j in range(len(my_data)):
        best_proba = 0
        index = 0
        for i in range(len(thetas)):
            if (probas[i][j] >= best_proba):
                best_proba = probas[i][j]
                index = i
        predics[j] = hogwarts_house_dict_inv[index]
    
    logistic_regression_utils.compare_lr_results(results, predics)


if (__name__ == "__main__"):
    main()
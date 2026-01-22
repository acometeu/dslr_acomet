import pandas as pd
import numpy as np
import logistic_regression
import logistic_regression_utils
import data_parsing


def main():
    data_pd = pd.read_csv('dataset_train.csv')
    data_pd.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1, inplace=True)
    hogwarts_house_dict = {"Gryffindor": 0, "Slytherin": 1, "Hufflepuff": 2, "Ravenclaw": 3}
    data_pd = data_parsing.replace_nan_value(data_pd, hogwarts_house_dict)
    results = data_pd['Hogwarts House']
    
    data_pd.drop('Hogwarts House', axis=1, inplace=True)
    # print(f"results = {results}")


    # transform in numpy
    # my_results = results.to_numpy(dtype=np.float64)
    my_data = data_pd.to_numpy(dtype=np.float64)
    logistic_regression_utils.normalise_data(my_data)


    # temp
    hogwarts_house_dict_inv = {v: k for k, v in hogwarts_house_dict.items()}
    thetas = [x for x in range(len(hogwarts_house_dict_inv))]
    for i in range(len(hogwarts_house_dict_inv)):
        binomial_results = results.copy()
        binomial_results[binomial_results != hogwarts_house_dict_inv[i]] = 0
        binomial_results[binomial_results == hogwarts_house_dict_inv[i]] = 1
        binomial_results = binomial_results.to_numpy(dtype=np.float64)
        thetas[i] = logistic_regression.gradient_descent(my_data, binomial_results)

    

        # test if it works
        predics = logistic_regression.predict(my_data, thetas[i])
        np.set_printoptions(threshold=predics.size)
        logistic_regression_utils.compare_lr_results(binomial_results, predics)


    # test if it works
    # theta = logistic_regression.gradient_descent(my_data, my_results)
    # predics = logistic_regression.predict(my_data, theta)
    # np.set_printoptions(threshold=predics.size)
    # logistic_regression_utils.compare_lr_results(my_results, predics)


if (__name__ == "__main__"):
    main()
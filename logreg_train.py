import numpy as np
import pandas as pd
import sys
import data_parsing


def cost_function(theta, X, y):
    predictions = sigmoid(X @ theta)
    # to avoid warning when divided by 0 or by +inf
    predictions[predictions == 1] = 0.999
    predictions[predictions == 0] = 0.001
    cost = (np.log(predictions) * y) + (np.log(1 - predictions) * (1 - y))
    return (-sum(cost) / y.size)


def sigmoid(z):
    # to avoid warning and too large rounding
    z[z > 700] = 700
    z[z < -700] = -700
    return (1.0 / (1.0 + np.exp(-z)))


def calculate_gradient(theta, X, y):
    m = y.size
    return (X.T @ (sigmoid(X @ theta) - y) / m) # derivative


def gradient_descent(X, y, alpha=0.001, iter=1000):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]    # intercept (or bias)
    theta = np.zeros(X_b.shape[1])

    #save best values
    cost_save = np.finfo(np.float64).max
    loop_save = 0
    theta_save = theta.copy()

    for i in range(iter):
        # print(f"loop = {i}")
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha * grad

        # save best values
        cost = cost_function(theta, X_b, y)
        if (cost < cost_save):
            cost_save = cost
            loop_save = i
            theta_save = theta.copy()
        
        # print(f"cost = {cost}")


    print(f"Best solution find at loop {loop_save} with a cost of {cost_save}")
    print(f"So {i - loop_save} useless loop done !")
    return theta_save


def logreg_train(data_csv, parsing_method=data_parsing.replace_nan_value_by_0, alpha=0.001, iter=1000):
    X = data_parsing.get_students_scores(data_csv, parsing_method)
    y = data_parsing.get_student_houses(data_csv, parsing_method)

    hogwarts_house_dict = {0: "Gryffindor", 1: "Slytherin", 2: "Hufflepuff", 3: "Ravenclaw"}
    thetas = pd.DataFrame()

    for i in range(len(hogwarts_house_dict)):
        binomial_results = y.copy()
        binomial_results[binomial_results != hogwarts_house_dict[i]] = 0
        binomial_results[binomial_results == hogwarts_house_dict[i]] = 1
        binomial_results = binomial_results.to_numpy(dtype=np.float64)
        thetas[hogwarts_house_dict[i]] = gradient_descent(X, binomial_results, alpha, iter)

    thetas.to_csv('weights.csv', index=False)


def logreg_train_parse_args():
    args_dic = {'data_csv=': "dataset_train.csv", 'parse_method=': 'replace_nan_value_by_0', 'alpha=': "0.001", 'iter=': "1000"}

    i = 1
    while(i < len(sys.argv)):
        arg_found = False
        for arg in args_dic.keys():
            if (sys.argv[i].startswith(arg)):
                part = sys.argv[i].partition(arg)
                args_dic[arg] = part[2]
                arg_found = True
                break
        if (not arg_found):
            args_dic['data_csv='] = sys.argv[i]
        i += 1

    # transform args in correct type
    func_dic = {'replace_nan_value': data_parsing.replace_nan_value, 'pandas_remove_nan_line': data_parsing.pandas_remove_nan_line, 'replace_nan_value_by_0': data_parsing.replace_nan_value_by_0}
    args_dic['parse_method='] = func_dic[args_dic['parse_method=']]
    args_dic['iter='] = int(args_dic['iter='])
    args_dic['alpha='] = float(args_dic['alpha='])

    return(args_dic)



def main():

    args = logreg_train_parse_args()
    logreg_train(args['data_csv='], args['parse_method='], args['alpha='], args['iter='])


if (__name__ == "__main__"):
    main()
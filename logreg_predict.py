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


def logreg_predict_parse_args():
    args_dic = {'data_csv': "dataset_test.csv", 'weights_csv': "weights.csv", 'parse_method': 'replace_nan_value_by_0'}

    if (len(sys.argv) > 4):
        print("Error to many argument, 3 maximum")
        return

    i = 1
    while (i < len(sys.argv)):
        print(f"loop {i}")
        args_dic[list(args_dic.keys())[i - 1]] = sys.argv[i]
        i += 1

    # transform args in correct type
    func_dic = {'replace_nan_value': data_parsing.replace_nan_value, 'pandas_remove_nan_line': data_parsing.pandas_remove_nan_line, 'replace_nan_value_by_0': data_parsing.replace_nan_value_by_0}
    args_dic['parse_method'] = func_dic[args_dic['parse_method']]

    return(args_dic)


def main():
    ''' Arguments are optionals :
        arg 1 : data in csv file                    (default : 'dataset_test.csv')
        arg 1 : weights in csv file                 (default : 'weights.csv')
        arg 3 : parsing method used to filter data  (default : 'replace_nan_value_by_0')
        others choices : ('pandas_remove_nan_line', 'replace_nan_value')
    '''

    args = logreg_predict_parse_args()
    if (not args):
        return
    print (args)
    logreg_predict(args['data_csv'], args['weights_csv'], args['parse_method'])


if (__name__ == "__main__"):
    main()
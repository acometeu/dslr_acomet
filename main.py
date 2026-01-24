import pandas as pd
import numpy as np
import data_parsing
import logreg_train
import logreg_predict


def main():
    my_data = data_parsing.get_students_scores('dataset_train.csv', data_parsing.replace_nan_value_by_0)
    results = data_parsing.get_student_houses('dataset_train.csv')

    thetas = logreg_train.logreg_train(my_data, results)


    my_data = data_parsing.get_students_scores('dataset_train.csv', data_parsing.replace_nan_value_by_0)
    results = data_parsing.get_student_houses('dataset_train.csv')
    
    logreg_predict.logreg_predict(thetas, my_data, results)


if (__name__ == "__main__"):
    main()
import pandas as pd
import numpy as np
import utils


def hogwarts_house_to_value(data, hogwarts_house_dict):
    my_result = data['Hogwarts House'].map(hogwarts_house_dict)
    return (my_result)


def pandas_remove_nan_line(data):
    return (data[~data.isnull().any(axis=1) == True])


def replace_nan_value_by_0(data):
    positions = np.argwhere(data.isna().to_numpy())
    for i in positions:
        data.iloc[i[0], i[1]] = 0
    return(data)


def replace_nan_value(data):
    house_average = pandas_get_house_average(data)

    hogwarts_house_names_dict = {"Gryffindor":0, "Slytherin":1, "Hufflepuff":2, "Ravenclaw":3}
    positions = np.argwhere(data.isna().to_numpy())
    for i in positions:
        house_of_instance = data.loc[i[0], 'Hogwarts House']
        right_house_average = house_average[hogwarts_house_names_dict[house_of_instance]]
        data.iloc[i[0], i[1]] = right_house_average.iloc[i[1]]
    
    return data


def get_students_scores(data_csv, treat_nan_values=replace_nan_value_by_0):
    ''' parameter 1 : name of the data.csv file
        parameter 2 : (optional) name of the function used to treat nan values
        Return X : the score of each feature per index '''
    
    X = pd.read_csv(data_csv)
    X.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1, inplace=True)
    X = treat_nan_values(X)
    X.drop('Hogwarts House', axis=1, inplace=True)
    X = X.to_numpy(dtype=np.float64)
    utils.normalise_data(X)
    return (X)


def get_student_houses(data_csv, treat_nan_values=replace_nan_value_by_0):
    y = pd.read_csv(data_csv)
    if (treat_nan_values == pandas_remove_nan_line):
        y = pandas_remove_nan_line(y)
    y = y['Hogwarts House']
    return(y)



def pandas_get_house_average(data):
    hogwarts_house_names_list = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    house_average = list(range(len(hogwarts_house_names_list)))
    data_no_nan = pandas_remove_nan_line(data)

    for i in range(len(hogwarts_house_names_list)):
        house_notes = data_no_nan[data_no_nan['Hogwarts House'] == hogwarts_house_names_list[i]].copy()
        house_notes.drop('Hogwarts House', axis=1, inplace=True)
        house_average[i] = house_notes.sum() / len(house_notes)

    return(house_average)


def main():
    pass


if (__name__ == "__main__"):
    main()
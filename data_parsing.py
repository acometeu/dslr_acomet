import pandas as pd
import numpy as np


def hogwarts_house_to_value(data, hogwarts_house_dict):
    my_result = data['Hogwarts House'].map(hogwarts_house_dict)
    return (my_result)


def pandas_remove_nan_line(data):
    return (data[~data.isnull().any(axis=1) == True])


def pandas_get_house_average(data):
    hogwarts_house_names_list = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    house_average = list(range(len(hogwarts_house_names_list)))
    data_no_nan = pandas_remove_nan_line(data)

    for i in range(len(hogwarts_house_names_list)):
        house_notes = data_no_nan[data_no_nan['Hogwarts House'] == hogwarts_house_names_list[i]].copy()
        house_notes.drop('Hogwarts House', axis=1, inplace=True)
        house_average[i] = house_notes.sum() / len(house_notes)

    return(house_average)


def replace_nan_value(data, hogwarts_house_names_dict):
    house_average = pandas_get_house_average(data)

    positions = np.argwhere(data.isna().to_numpy())
    for i in positions:
        house_of_instance = data.loc[i[0], 'Hogwarts House']
        right_house_average = house_average[hogwarts_house_names_dict[house_of_instance]]
        data.iloc[i[0], i[1]] = right_house_average.iloc[i[1]]
    
    return data


def main():
    pass


if (__name__ == "__main__"):
    main()
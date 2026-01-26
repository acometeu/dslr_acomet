import pandas as pd
import numpy as np
import logreg_train
import logreg_predict
import utils


def main():
    logreg_train.logreg_train('dataset_train.csv')

    logreg_predict.logreg_predict('dataset_train.csv', 'weights.csv')

    utils.compare_lr_results('dataset_train.csv', 'houses.csv')


if (__name__ == "__main__"):
    main()
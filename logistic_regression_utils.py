import numpy as np


def compare_lr_results(true_results, test_results):
    accuracy = (true_results == test_results)
    good_results = accuracy[accuracy == True].size
    false_results = accuracy[accuracy == False].size
    print(f"Accuracy = {good_results * 100 / accuracy.size}%, (True : {good_results}, False : {false_results})")


def normalise_data(data):
    i = data.shape[1] - 1
    while i >= 0:
        max = np.max(data[:, i])
        min = np.min(data[:, i])
        range = abs(max - min)
        data[:, i] -= min
        data[:, i] = data[:, i] * 100 / range
        i -= 1
    return (data)


def main():
    pass


if (__name__ == "__main__"):
    main()
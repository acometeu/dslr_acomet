import numpy as np


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


def gradient_descent(X, y, alpha=0.001, num_iter=1000, tol=0.01):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]    # intercept (or bias)
    theta = np.zeros(X_b.shape[1])

    #save best values
    cost_save = np.finfo(np.float64).max
    loop_save = 0
    theta_save = theta.copy()

    for i in range(num_iter):
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


def predict_proba(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)


def predict(X, theta, treshold=0.5):
    return (predict_proba(X, theta) >= treshold)


def main():
    pass


if (__name__ == "__main__"):
    main()


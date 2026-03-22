# solve.py
# Students must complete the TODO sections

import numpy as np

# ============================================================
# OLS IMPLEMENTATIONS
# ============================================================

def ols_with_intercept(X, y):
    """
    Ordinary Least Squares WITH intercept.

    Parameters
    ----------
    X : numpy array (N,d)
        Feature matrix

    y : numpy array (N,)
        Target values

    Returns
    -------
    w  : slope vector (d,)
    w0 : intercept scalar
    """

    # TODO:
    # Step 1: Add a column of ones to X to represent intercept
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    # Step 2: Use the normal equation
    #
    # w = (X^T X)^(-1) X^T y
    w1 = np.linalg.inv(X1.T@X1)@X1.T@y
    
    #
    # Step 3: Separate intercept from weight vector
    w0 = w1[0]
    w = w1[1:]
    return w, w0

    raise NotImplementedError


def ols_no_intercept(X, y):
    """
    OLS WITHOUT intercept.

    Use the normal equation:

        w = (X^T X)^(-1) X^T y
    """

    # TODO:
    # Implement closed-form solution
    w = np.linalg.inv(X.T@X)@X.T@y
    return w

    raise NotImplementedError


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_with_intercept(X, w, w0):
    """
    Predict y = Xw + w0
    """

    # TODO:
    # return predicted values
    return X@w +w0

    raise NotImplementedError


def predict_no_intercept(X, w):
    """
    Predict y = Xw
    """

    # TODO
    return X@w

    raise NotImplementedError


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y, y_hat):
    """
    Compute the following metrics:

    1. Mean Squared Error (MSE)

        MSE = mean((y - y_hat)^2)

    2. Correlation

    3. Squared Correlation

    4. R^2 score
    """

    # TODO
    mse= np.mean((y-y_hat)**2)
    correlation = np.corrcoef(y, y_hat)[0,1]
    squared_correlation = correlation**2
    r2_score = 1 - (np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2))
    return mse, correlation, squared_correlation, r2_score
    raise NotImplementedError


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """
    Load dataset from CSV files.

    CSV format:

    size,bedrooms,age,distance,price

    First 4 columns = features
    Last column = target
    """

    train = np.loadtxt("C:\\Users\\ritan\\OneDrive\\Desktop\\rakshak\\ML-Assignment\\assignment2\\task2-OLS\\train.csv", delimiter=",", skiprows=1)
    test = np.loadtxt("C:\\Users\\ritan\\OneDrive\\Desktop\\rakshak\\ML-Assignment\\assignment2\\task2-OLS\\test.csv", delimiter=",", skiprows=1)

    # TODO:
    # Separate X and y
    X= train[:, :-1]
    y= train[:, -1]
    test_X= test[:, :-1]
    test_y= test[:, -1]

    return X, y, test_X, test_y

    raise NotImplementedError
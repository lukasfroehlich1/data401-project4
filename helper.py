import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV, LinearRegression



def basis_expansion(X, highest_order):
    cols = list(X.columns)[:]
    X = X.copy()
    
    for order in range(2, highest_order+1):
        for col in cols:
            X[col + "^" + str(order)] = X[col] ** order

    return X

def fit_model(X, y):
    # basis expansion showed no real improvement
    #X = basis_expansion(X, 1)
    #print(X.shape)
    lasso = LassoCV()
    lasso.fit(X, y)
    
    keep_vars = []

    for name, val in zip(X.columns, lasso.coef_):    
        if val != 0:
            keep_vars.append(name)
    
    linear = LinearRegression()
    linear.fit(X[keep_vars], y)
    print(linear.score(X[keep_vars], y))
    
    for name, val in sorted(zip(keep_vars, linear.coef_), key=lambda x: x[1], reverse=True):
        print((name, val))
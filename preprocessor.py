import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

'''OneHot Encoding of categorical features'''

class LogTransformer(BaseEstimator, TransformerMixin):
    '''Logarithmic transformation of skewed features'''
    
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables
        
    def fit(self, X,y=None):
        # we need this step to fit the sklearn pipeline
        return self
    
    def transform(self, X):
        # so that we do not over-write the original dataframe
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.log(X[feature] + 1)
        return X
    
    
class DropFeatures(BaseEstimator, TransformerMixin):
    '''Drop features not selected for modelling'''
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X,y=None):
        # we need this step to fit the sklearn pipeline
        return self
    
    def transform(self, X):
        # so that we do not over-write the original dataframe
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X
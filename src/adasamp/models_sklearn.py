# -*- coding: utf-8 -*-
"""Helper classes for simple adaptive sampling models based on scikit-learn."""


import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import sklearn.svm as svm
import sklearn.gaussian_process.kernels as kernels
from adasamp.models import RegressionModel, ClassificationModel


class AdvancedMultiOutputRegressor(BaseEstimator):
    """Regression model with multiple, independent outputs.
    
    For each output, a seperate sklearn-model is realized."""
    
    def __init__(self, model_constructor, predict_fun=None, **kwargs):
        super().__init__()
        self._model_constructor = model_constructor
        self._predict_fun = predict_fun
        self._kwargs = kwargs

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._y_dim = y.shape[1]
        self._model = MultiOutputRegressor(self._model_constructor(**self._kwargs))
        self._model.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, ['_X', '_y'])     
        if not return_std:
            predictions = self._model.predict(X)
        else:
            estimator_predictions = [estimator.predict(X, return_std=True) if self._predict_fun is None else self._predict_fun(estimator, X) for estimator in self._model.estimators_]
            mu = np.stack(list(estimator_predictions[i][0] for i in range(self._y_dim)),axis=1)
            sigma = np.stack(list(estimator_predictions[i][1] for i in range(self._y_dim)),axis=1) 
            predictions = mu, sigma
        return predictions

    def score(self, X, y):
        return self._model.score(X, y)
   
    def get_params(self, deep=True):
        return {"kwargs" : self._kwargs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
class MultivariateGPR(AdvancedMultiOutputRegressor):
    """Gaussian process regressor with multiple, independent outputs.
    
    An independent `GaussianProcessRegressor` for each output is realized."""
    
    def __init__(self, **kwargs):
        super().__init__(GaussianProcessRegressor, None, **kwargs)
        
        
class Y_Model_GPR(RegressionModel):
    """Gaussian process regressor with multiple, independent outputs including
    scaling.
    
    Realizes a pipeline: Standardization, independent 
    `GaussianProcessRegressor` for each output."""
    
    
    def __init__(self, kernel = 1.0*kernels.Matern(), n_restarts_optimizer=5, random_state=0):
        super().__init__()
        self._model = Pipeline([('Yscaler', preprocessing.StandardScaler()),
                                ('Yreg', MultivariateGPR(kernel=kernel, random_state=random_state, n_restarts_optimizer=n_restarts_optimizer))])

    def fit(self, X, Y):
        self._model.fit(X, Y)         
        
    def predict(self, X, return_std):
        return self._model.predict(X, return_std=return_std)  
        
    
class f_Model_SVM(ClassificationModel):
    """Support vector classifier inlcuding scaling.
    
    Realizes a pipeline: Standardization, `SVM` with `GridSearchCV` 
    hyperparameter optimization."""
    
    def __init__(self, kernel='rbf', cv_dict=dict(C=np.logspace(-1,3,10), gamma=np.logspace(-1,3,10)), n_splits=3, random_state=0):
        super().__init__() 
        self._model = Pipeline([('fscaler', preprocessing.StandardScaler()),
                                ('fclf', model_selection.GridSearchCV(svm.SVC(kernel=kernel, random_state=random_state, probability=True), 
                                                                      cv_dict, cv=model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)))])

    def fit(self, X, f):
        self._model.fit(X, f)        

    def predict(self, X):
        return self._model.predict(X)           

    def predict_true_proba(self, X):
        return self._model.predict_proba(X)[:,1].ravel()   
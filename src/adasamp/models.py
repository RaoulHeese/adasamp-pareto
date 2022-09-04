# -*- coding: utf-8 -*-
"""Wrappers for adaptive sampling models."""


from abc import ABC, abstractmethod


class RegressionModel(ABC):
    """Regression model wrapper.
    
    Estimator for the adaptive sampling regression problem of predicting 
    ``Y`` (goals) from ``X`` (features)."""
    
    def __init__(self):
        pass
        
    @abstractmethod
    def fit(self, X, Y):
        """Train estimator.
        
        Is called at least once before any prediction.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_samples, X_dim)
            Array of features (estimator inputs).
            
        Y : ndarray of shape (n_samples, Y_dim)
            Array of regression values (estimator targets).
        """
        
        raise NotImplementedError            
        
    @abstractmethod
    def predict(self, X, return_std):
        """Estimator prediction.
         
        It is assumed that the predictive distribution is a Gaussian specified 
        by a mean and a standard deviation.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_query_points, X_dim)
            Query points where the estimator is evaluated.
        
        return_std : bool
            If set to True, return both the means and the standard deviations.
            If set to False, only return the means.
            
        Returns
        -------
        
        Y_mu : ndarray of shape (n_query_points, Y_dim)
            Mean of the predictive distribution at the query points.
            
        Y_sigma : ndarray of shape (n_query_points, Y_dim)
           Standard deviation of the predictive distribution at the query 
           points. Covariances are not returned.
        """
        
        raise NotImplementedError  
        
        
class ClassificationModel(ABC):
    """Classification model wrapper.
    
    Estimator for the adaptive sampling classification problem of predicting 
    ``f`` (binary feasibilities) from ``X`` (features)."""
    
    def __init__(self):
        pass
        
    @abstractmethod
    def fit(self, X, f):
        """Train estimator.
        
        Is called at least once before any prediction. The class labels can be 
        either True or False and are automatically converted to integers 
        before the training.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_samples, X_dim)
            Array of features (estimator inputs).
            
        f : ndarray of shape (n_samples,)
            Array of binary classification labels (estimator targets).
        """
        
        raise NotImplementedError            
        
    @abstractmethod
    def predict(self, X):
        """Classifier prediction.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_query_points, X_dim)
            Query points where the estimator is evaluated.
            
        Returns
        -------
        
        f : ndarray of shape (n_query_points,)
            Predicted labels at the query points.
        """
        
        raise NotImplementedError   

    @abstractmethod
    def predict_true_proba(self, X):
        """Classifier probability prediction.
        
        Returns the predicted probability of a True label at the respective 
        query points.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_query_points, X_dim)
            Query points where the estimator is evaluated.
            
        Returns
        -------
        
        p : ndarray of shape (n_query_points,)
            Predicted probability for a True label at the query points. It is 
            assumed that ``0 <= p <= 1``.
        """

        raise NotImplementedError   
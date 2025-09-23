import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class logtransform():
    def __init__(self):
        pass

    def transform(self, data):
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Input data contains NaN or Inf values while transforming")
        return np.log1p(data)  # log1p for numerically stable log(1 + x)

    def inverse_transform(self, data):
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Input data contains NaN or Inf values while inverse transforming")
        return np.expm1(data)  # expm1 for inverse transformation
    
class sqrttransform():
    def __init__(self):
        pass

    def transform(self, data):
        return np.sqrt(data)

    def inverse_transform(self, data):
        return np.square(data)
    

class optimal_BoxCox():
    def __init__(self):
        pass
    

    def transform_boxcox(self,Y, lambdas):
        """
        Apply Box-Cox transformation to each column of Y using the corresponding lambda in lambdas.

        For a value y and parameter lambda:
          - if |lambda| is very small: return log(y)
          - else: return (y**lambda - 1) / lambda

        Parameters:
        -----------
        Y : numpy array of shape (n_samples, n_series)
        lambdas : array-like of length n_series

        Returns:
        --------
        X : Transformed data (numpy array with same shape as Y)
        """
        n, p = Y.shape
        X = np.empty_like(Y, dtype=np.float64)
        for j in range(p):
            if np.any(Y[:, j] <= 0):
                raise ValueError(f"Column {j} contains non-positive values, which are not allowed for Box-Cox transformation.")
            lam = lambdas[j]
            if np.abs(lam) < 1.1e-5:#1e-8:
                X[:, j] = np.log(Y[:, j])
            else:
                X[:, j] = (Y[:, j]**lam - 1) / lam
        return X

    def joint_boxcox_negloglik(self,lambdas, Y):
        """
        Computes the negative joint log-likelihood for the Box-Cox transformation parameters.

        The log-likelihood is:
          L(λ) = sum_j (λ_j - 1) * sum_i log(y_ij) - (n/2) * log|Σ(λ)|
        where Σ(λ) is the covariance matrix of the transformed data.

        Parameters:
        -----------
        lambdas : array-like of shape (p,)

        Returns:
        --------
        Negative log-likelihood (float)
        """
        n, p = Y.shape
        X = self.transform_boxcox(Y, lambdas)
        mu = X.mean(axis=0)
        X_centered = X - mu
        cov = np.dot(X_centered.T, X_centered) / n

        # Compute the log-determinant of the covariance matrix
        sign, logdet = np.linalg.slogdet(cov)
        
#         print(sign, logdet)
        if sign <= 0:
            return 1e10  # Penalize non-positive definite covariance

        # Jacobian adjustment for all series
        jacobian = sum((lambdas[j] - 1) * np.sum(np.log(Y[:, j])) for j in range(p))

        loglik = jacobian - (n / 2) * logdet
        return -loglik  # We minimize the negative log-likelihood

    def joint_boxcox(self,Y, initial_lambdas=None):
        """
        Jointly estimates the Box-Cox transformation parameters for a multivariate dataset.

        Parameters:
        -----------
        Y : numpy array or DataFrame of shape (n_samples, n_series)
            The data to transform (must be strictly positive).
        initial_lambdas : array-like (optional)
            Initial guess for lambda for each series. Defaults to ones.

        Returns:
        --------
        opt_lambdas : numpy array of optimal Box-Cox parameters
        X_trans : Transformed data (same shape as Y)
        """
        if isinstance(Y, pd.DataFrame):
            Y = Y.values
        n, p = Y.shape
        if np.any(Y <= 0):
            raise ValueError("All values must be positive for Box-Cox transformation.")
        if initial_lambdas is None:
            initial_lambdas = np.ones(p)  # start with no transformation

#         bounds = [(-4, 4)] * p  # one bound per feature
        res = minimize(self.joint_boxcox_negloglik, x0=initial_lambdas, args=(Y,), method='BFGS')

        opt_lambdas = res.x
        X_trans = self.transform_boxcox(Y, opt_lambdas)
        return opt_lambdas, X_trans

    def inverse_transform(self,X, lambdas):
        """
        Inverse Box-Cox transformation with a threshold for treating lambda as zero.
        """
        n, p = X.shape
        Y = np.empty_like(X, dtype=np.float64)
        for j in range(p):
            lam = lambdas[j]
            if np.abs(lam) <1.1e-5:
                Y[:, j] = np.exp(X[:, j])
            else:
                Y[:, j] = (lam * X[:, j] + 1) ** (1 / lam)

        return Y
    
class SkewAwareZ:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.mean_ = None
        self.std_ = None
        self.skew_ = None
        self.adjustment_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        centered = (X - self.mean_) / self.std_
        self.skew_ = np.mean(centered ** 3, axis=0)
        self.adjustment_ = self.std_ * (1 + self.alpha * np.abs(self.skew_))
        return self

    def transform(self, X):
        return (X - self.mean_) / self.adjustment_

    def inverse_transform(self, Z):
        return Z * self.adjustment_ + self.mean_

import numpy as np
from implementations import *
from preprocess import *
from helpers import *
from knn import *


class Preprocessor:
    fit_data = None
    cov_decomposition = None

    def __init__(self):
        pass

    def fit(self, data):
        """Initialize with training data -- will initialize so that all other transformations use this."""
        self.fit_data = data.copy()
        self.cov_decomposition = None

    def transfrom(self, tx, type, param=1.0, upper=95, lower=5):
        """
        Data Transformation
        using sigmoid / log/ inverse_log / clip/ boxcox
        """
        if type == "sigmoid":
            return sigmoid(tx)
        if type == "log":
            return np.log(tx + 1)
        if type == "inverse_log":
            return 1.0 / np.log(tx + 1)
        if type == "square":
            return np.sqrt(tx)
        if type == "clip":
            upper = np.percentile(self.fit_data, upper)
            lower = np.percentile(self.fit_data, lower)
            return np.clip(tx, upper, lower)
        if type == "boxcox":
            return (tx**param - 1) / param
        return tx

    def standardize(self, tx, col=True):
        """
        Standardization
        col: True: standardize per column, False: per row
        """
        if col:
            data_T = np.transpose(self.fit_data)
            tx_T_std = np.zeros((tx.shape[1], tx.shape[0]))
            tx_T = np.transpose(tx)
            for i in range(tx.shape[1]):
                if np.std(data_T[i]) == 0:
                    pass
                    # This can happen if e.g. we have the constant part of our polynomial where all items will just have a value of 1.
                    # The correct value in those cases is to leave it at zero
                else:
                    tx_T_std[i] = (tx_T[i] - np.mean(data_T[i])) / np.std(data_T[i])
            return np.transpose(tx_T_std)
        else:
            # If normalizing per row then don't rely on fitted
            tx_std = np.zeros((tx.shape[0], tx.shape[1]))
            for i in range(tx.shape[0]):
                tx_std[i] = (tx[i] - np.mean(tx[i])) / np.std(tx[i])
            return tx_std

    def normalize(self, tx, col=True):
        """
        Normalization
        col: True: normalize per column, False: per row
        """
        if col:
            data_T = np.transpose(self.fit_data)
            tx_T = np.transpose(tx)
            tx_T_norm = np.zeros((tx.shape[1], tx.shape[0]))
            for i in range(tx.shape[1]):
                tx_T_norm[i] = (np.max(data_T[i]) - tx_T[i]) / (
                    np.max(data_T[i]) - np.min(data_T[i])
                )
            return np.transpose(tx_T_norm)
        else:
            tx_norm = np.zeros((tx.shape[0], tx.shape[1]))
            for i in range(tx.shape[0]):
                tx_norm[i] = (np.max(tx[i]) - tx[i]) / (np.max(tx[i]) - np.min(tx[i]))
            return tx_norm

    def std_norm_preprocess(self, tx, func="std", col=True):
        """
        Helper function for choosing standardization / normalization,
        'std': standardization, 'norm': normalization)
        col: True: standardize or normalize per column, False: per row
        """
        if func == "std":
            tx_std = self.standardize(tx, col=True)
            return tx_std

        elif func == "norm":
            tx_norm = self.normalize(tx, col=True)
            return tx_norm

    def data_preprocess(self, tx, filling="median", mode="std", col=True):
        """
        Helper function dealing with different means of preprocessing.
        Input:
            tx: original feature matrix
            y: original label vector
            id : original id vector
            filling: methods for filling missing values ('mean', 'median', 'zero')
            mode:'std': standardization, 'norm': normalization
            col: True: standardize or normalize per column, False: per row
        """
        if filling != "knn" and filling != "knn_sample":
            for i in range(0, len(tx[0])):
                self.basic_replace_missing_value(tx, i, func=filling)
        else:
            data_standardize = tx.copy()
            fit_data_std = self.fit_data.copy()
            for i in range(0, len(tx[0])):
                self.basic_replace_missing_value(data_standardize, i, func="zero")
                self.basic_replace_missing_value(fit_data_std, i, func="zero")
            preproc = Preprocessor()
            preproc.fit(self.fit_data)
            data_standardize = preproc.std_norm_preprocess(
                data_standardize, func="std", col=True
            )
            fit_data_std = preproc.std_norm_preprocess(
                fit_data_std, func="std", col=True
            )
            print("knn or knn sample")
            if filling == "knn":
                tx = knn_auto_fill(tx, data_standardize, self.fit_data, fit_data_std)
            else:
                print("knn random")
                tx = knn_random_fill(tx, data_standardize, self.fit_data, fit_data_std)

        return tx

    def basic_replace_missing_value(self, data, col=0, func="median"):
        """
        Use median, mean, zero, LR to fill the missing data
        """
        if func == "median":
            rep = np.median(self.fit_data[self.fit_data[:, col] != -999][:, col])
            data[:, col] = np.where(data[:, col] == -999, rep, data[:, col])

        elif func == "zero":
            rep = 0
            data[:, col] = np.where(data[:, col] == -999, rep, data[:, col])

        elif func == "mean":
            rep = np.mean(self.fit_data[self.fit_data[:, col] != -999][:, col])
            data[:, col] = np.where(data[:, col] == -999, rep, data[:, col])

    def PCA(self, data, k):
        """
        Projects data to a k dimensional subspace using PCA
        Input:
        Data: Data to perform the transform on.
        k: Dimension of subspace to project to

        Returns:
        rank k representation of matrix
        W: Map to k dimensional space
        """
        # Generate covariance matrix:
        if self.cov_decomposition == None:
            input_data = self.standardize(self.fit_data).T
            covariance = np.cov(
                input_data
            )  # (input_data.T @ input_data) / (input_data.shape[0]-1)
            # Get eigenvalues and eigenvectors of covariance matrix
            eigenvals, eigenvecs = np.linalg.eigh(covariance)
            self.cov_decomposition = (eigenvals, eigenvecs)

        eigenvals, eigenvecs = self.cov_decomposition
        # Rank k PCA requires us to take the k eigenvectors with the corresponding largest eigenvalues. We need to sort them first
        sorted_indexes = (
            eigenvals.argsort()
        )  # Generates list of indexes which correspond to smallest to largest eigenvalues.
        largest_indexes = np.flip(sorted_indexes)[:k]  # Get the k largest indexes
        W = eigenvecs[:, largest_indexes]
        return self.standardize(data) @ W, W

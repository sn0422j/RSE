import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from typing import NoReturn

class RSE(BaseEstimator, RegressorMixin):
    """
    Representational Similarity Encoding Regressor.

    This class implements representational similarity encoding in Anderson et al., (2016). 

    Parameters
    ----------
    None

    Attributes
    ----------
    X_train_ : np.ndarray (n_samples, n_features)
        Stimulus Featues Matrix for Training.
    Y_train_ : np.ndarray (n_samples, n_features)
        Neural Activity Matrix for Training.

    References
    ----------
    Anderson, A. J., Zinszer, B. D., & Raizada, R. D. S. (2016). 
    Representational similarity encoding for fMRI: Pattern-based synthesis 
    to predict brain activity using stimulus-model-similarities. 
    NeuroImage, 128, 44–53. 
    https://doi.org/10.1016/j.neuroimage.2015.12.035
    """

    def __init__(self) -> NoReturn:
        pass

    def fit(self, X: np.ndarray, Y: np.ndarray) -> object:
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Stimulus Featues Matrix, where n_samples is the number of stimulus samples and
            n_features is the number of features.
        Y : np.ndarray of shape (n_samples, n_features)
            Neural Activity Matrix, where n_samples is the number of stimulus samples and
            n_features is the number of features (voxels).
        
        Returns
        -------
        self
            Fitted estimator.
        """
        X, Y = check_X_y(X, Y, multi_output=True)
        self.X_train_: np.ndarray = X
        self.Y_train_: np.ndarray = Y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using representational similarity encoding model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Stimulus Featues Matrix, where n_samples is the number of stimulus samples and
            n_features is the number of features.
        
        Returns
        -------
        Y : np.ndarray of shape (n_samples, n_features)
            Neural Activity Matrix, where n_samples is the number of stimulus samples and
            n_features is the number of features (voxels).
        """
        check_is_fitted(self, ['X_train_', 'Y_train_'])
        X = check_array(X)
        n_train_ = np.shape(self.X_train_)[0]
        
        C_ = np.corrcoef(np.vstack( ( self.X_train_, X ) ))[n_train_:, :n_train_]
        C_normed_ = np.dot(np.diag(1/np.sum(np.abs(C_),axis=1)), C_)
        return np.dot(C_normed_, self.Y_train_)


import torch
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from helper_functions import reshape_ratings_dataframe, imputate_data_with_0

class my_SVD1(TruncatedSVD):

    def __init__(self, n_components=5, random_state=42):
        """
        Initialize the SVD1 model.

        :param n_components: Number of singular values/components to keep (rank r).
        :param random_state: Random seed.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.W = None
        self.H = None

    def fit(self, Z):
        """
        Fit the TruncatedSVD model to the input matrix Z, and compute W and H.

        :param Z: np.ndarray, shape (n_users, n_items). Fully populated (no missing entries) rating matrix.
        """
        super().fit(Z)
        # singular_values_ and components_ set by super().fit
        sigma = self.singular_values_
        Sigma = np.diag(sigma)
        VT = self.components_
        # transform(Z) yields U_r * Sigma_r
        URSigma = super().transform(Z)
        # Compute W and H
        self.W = URSigma / sigma
        self.H = Sigma.dot(VT)
        return self

    def predict(self, user_index, item_index):
        """
        Predict the rating for a given user and item.
        Returns rounded to the nearest 0.5.

        :param user_index: np.ndarray
        :param item_index: np.ndarray
        """
        if self.W is None or self.H is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        val = self.W[user_index].dot(self.H[:, item_index])
        return np.round(val * 2) / 2


if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings.csv")
    Z2 = reshape_ratings_dataframe(ratings)
    Z2 = imputate_data_with_0(Z2)
    print(Z2)

    model = my_SVD1(n_components=234, random_state=42)
    model.fit(Z2)

    prediction = model.predict([1,2,3], [1,2,3])
    print(prediction)
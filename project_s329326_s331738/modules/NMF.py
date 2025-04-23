from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
from helper_functions import reshape_ratings_dataframe, imputate_data_with_0

class my_NMF(NMF):
    def __init__(self, n_components=10, init='random', max_iter=100, random_state=42):
        """
                Initialize the NMF model.

                :param n_components: Number of singular values/components to keep (rank r).
                :param random_state: Random seed.
        """
        super().__init__(n_components=n_components, init=init, max_iter=max_iter, random_state=random_state)
        self.W = None
        self.H = None

    def fit(self, Z):
        """
        Fit the NMF model to the input matrix Z, and compute W and H.

        :param Z: np.ndarray, shape (n_users, n_items). Fully populated (no missing entries) rating matrix.
        """
        super().fit(Z)
        # Compute W and H
        self.W = self.fit_transform(Z)
        self.H = self.components_
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
        return pd.DataFrame(val * 2).round().clip(0, 10)/2


if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings.csv")
    Z2 = reshape_ratings_dataframe(ratings)
    Z2 = imputate_data_with_0(Z2)
    #print(Z2)

    model = my_NMF(n_components=23, max_iter=500, random_state=42)
    model.fit(Z2)

    prediction = model.predict([1, 2, 3, 609, 0], [0, 1, 2, 3])
    print(prediction)
    #print(Z2)
    print(ratings)
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd


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
        self.recovered_Z = None

    def fit(self, Z):
        """
        Fit the TruncatedSVD model to the input matrix Z, and compute W and H.

        :param Z: np.ndarray, shape (n_users, n_items). Fully populated (no missing entries) rating matrix.
        """
        super().fit(Z)
        # Compute W and H
        self.W = self.fit_transform(Z)
        self.H = self.components_
        return self

    def get_recovered_Z(self):
        """
        Recovering Z function with proper rounding
        :return:
        """
        if self.W is None or self.H is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        val = self.W.dot(self.H)
        self.recovered_Z = np.clip(np.round(val*2)/2, 0, 5)

        return pd.DataFrame(self.recovered_Z)

    def predict(self, user_index):
        """
        Predict the rating for a given user and item.
        Returns rounded to the nearest 0.5.

        :param user_index: np.ndarray
        :param item_index: np.ndarray
        """
        if self.W is None or self.H is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        if self.recovered_Z is None:
            raise ValueError("Z is not recovered yet. Call get_recovered_Z() first.")
        return self.recovered_Z[tuple(zip(*user_index))]


if __name__ == "__main__":
    from helper_functions import reshape_ratings_dataframe, imputate_data_with_0
    from project_s329326_s331738.modules.build_train_matrix import build_train_set, build_test_set
    from helper_functions import rmse

    ratings = pd.read_csv("../data/ratings.csv")
    Z2_nt = reshape_ratings_dataframe(ratings)
    Z2 = imputate_data_with_0(Z2_nt)

    idx_train, rt_train = build_train_set(Z2_nt, 60000)
    idx_test, rt_test = build_test_set(Z2_nt, idx_train)

    model = my_SVD1(n_components=250, random_state=42)
    model.fit(Z2)
    model.get_recovered_Z()

    prediction = model.predict(idx_test)
    print(prediction)

    print(rmse(prediction, rt_test))

    for i in [10, 50, 100, 150, 200, 250, 300, 350, 400]:
        ml = my_SVD1(n_components=i, random_state=42)
        ml.fit(Z2)
        ml.get_recovered_Z()
        prediction = ml.predict(idx_test)
        print(rmse(prediction, rt_test))

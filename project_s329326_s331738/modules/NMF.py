from sklearn.decomposition import NMF
import numpy as np
import pandas as pd


class my_NMF(NMF):
    def __init__(self, n_components=10, init='random', max_iter=300, random_state=42):
        """
                Initialize the NMF model.

                :param n_components: Number of singular values/components to keep (rank r).
                :param random_state: Random seed.
        """
        super().__init__(n_components=n_components, init=init, max_iter=max_iter, random_state=random_state)
        self.W = None
        self.H = None
        self.recovered_Z = None

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
    from helper_functions import reshape_ratings_dataframe, imputate_data_with_0, map_ids
    from project_s329326_s331738.modules.build_train_matrix import build_train_set, build_test_set, convert_train_set_to_good_shape
    from helper_functions import rmse

    ratings = pd.read_csv("../data/ratings.csv")

    rt_train = build_train_set(ratings, 0.6)
    rt_test = build_test_set(ratings, rt_train)

    Z2_nt, usermap, moviemap = convert_train_set_to_good_shape(rt_train, rt_test)

    Z2_nt_imp = imputate_data_with_0(Z2_nt)
    idx_test = map_ids(rt_test, usermap, moviemap)

    model = my_NMF(n_components=1, max_iter=20, random_state=42)
    model.fit(Z2_nt_imp)
    model.get_recovered_Z()
    prediction = model.predict(idx_test)
    print(prediction)

    print(rmse(prediction, rt_test['rating']))

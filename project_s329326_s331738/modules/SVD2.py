from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd


class my_SVD2(TruncatedSVD):

    def __init__(self, n_components=5, n_epochs=100, random_state=42):
        """
        Initialize the SVD1 model.

        :param n_components: Number of singular values/components to keep (rank r).
        :param random_state: Random seed.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.n_epochs = n_epochs
        self.W_r = None
        self.H_r = None
        self.recovered_Z = None
        self.loss_list = []

    def fit(self, Z, id_train_set, verbose=False):
        """
        Fit the TruncatedSVD model to the input matrix Z, and compute W and H.

        :param Z: np.ndarray, shape (n_users, n_items). Fully populated (no missing entries) rating matrix.
        """
        train_rows, train_cols = zip(*id_train_set)
        Z_train = np.array(Z)[train_rows, train_cols]
        Z_previous = np.copy(np.array(Z))

        for epoch in range(self.n_epochs):
            # SVD on previous step Z_previous (from previous step)
            W_r = super().fit_transform(Z_previous)
            H_r = self.components_
            Z_previous = W_r @ H_r

            # Update matrix with known data
            Z_next = np.copy(Z_previous)
            Z_next[train_rows, train_cols] = Z_train

            # Comparing with previous step only on train set
            loss = np.sum((Z_next[train_rows, train_cols] - Z_previous[train_rows, train_cols]) ** 2)

            # Update next step
            Z_previous = Z_next

            if epoch > 1 and abs(loss - self.loss_list[-1]) < 0.001:
                print(f"Number of performed epochs due to small error changes: {epoch}.")
                break

            if epoch % 10 == 0:
                self.loss_list.append(loss)
                if verbose:
                    print(f"Epoch {epoch}: loss = {loss:.4f}")

        self.W_r = W_r
        self.H_r = H_r

        return self

    def get_recovered_Z(self):

        Z_recovered = (2 * np.matmul(self.W_r, self.H_r)).round().clip(0.0, 10.0)
        self.recovered_Z = Z_recovered / 2

        return pd.DataFrame(self.recovered_Z)

    def predict(self, user_index, movie_index):

        ids = zip(user_index, movie_index)
        return self.recovered_Z[tuple(zip(*ids))]


if __name__ == "__main__":
    from helper_functions import reshape_ratings_dataframe, imputate_data_with_0, map_ids
    from project_s329326_s331738.modules.build_train_matrix import build_train_set, build_test_set

    ratings = pd.read_csv("../data/ratings.csv")
    Z2_full, usermap, moviemap = reshape_ratings_dataframe("../data/ratings.csv")
    Z2_imputed = imputate_data_with_0(Z2_full)

    Z2_train = build_train_set(ratings, 0.6)
    id_train = map_ids(Z2_train, usermap, moviemap)

    model = my_SVD2(n_components=10, n_epochs=100, random_state=42)
    model.fit(Z2_imputed, id_train)
    print(model.get_recovered_Z())
    print(model.predict([1, 3, 5], [3, 4, 1]))

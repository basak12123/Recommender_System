from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from helper_functions import reshape_ratings_dataframe, imputate_data_with_0
from tools.build_train_matrix import build_train_set, build_test_set


class my_SVD2(TruncatedSVD):

    def __init__(self, n_components=5, n_epochs=200, random_state=42):
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

    def fit(self, Z, id_train_set, verbose=True):
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
                print(loss)
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

    def compute_RMSE_on_test(self,  id_test_set, ratings_for_test_set):

        test_users_id, test_movies_id = tuple(zip(*id_test_set))
        predictions = self.predict(test_users_id, test_movies_id)
        print(predictions)

        return np.sqrt(np.mean((predictions - ratings_for_test_set) ** 2))


if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings.csv")
    Z2 = reshape_ratings_dataframe(ratings)

    id_train, Z2_train = build_train_set(Z2, 0.6)
    id_test, Z2_test = build_test_set(Z2, id_train)

    Z2 = imputate_data_with_0(Z2)
    print(Z2)

    model = my_SVD2(n_components=250, n_epochs=500, random_state=42)
    model.fit(Z2, id_train)
    print(model.get_recovered_Z())
    print(model.predict([1, 3, 5], [3, 4, 1]))
    print(model.compute_RMSE_on_test(id_test, Z2_test))

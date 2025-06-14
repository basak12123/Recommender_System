from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd


class my_SVD2(TruncatedSVD):

    def __init__(self, n_components=5, n_epochs=100):
        """
        Initialize the SVD1 model.

        :param n_components: Number of singular values/components to keep (rank r).
        :param n_epochs: Number of iterations of algorithm
        """
        super().__init__(n_components=n_components)
        self.n_epochs = n_epochs
        self.W_r = None
        self.H_r = None
        self.recovered_Z = None

    def fit(self, Z, id_train_set, verbose=False):
        """
        Fit the TruncatedSVD model to the input matrix Z, and compute W and H.
        """
        train_rows, train_cols = zip(*id_train_set)
        Z_np = np.array(Z)
        Z_train = Z_np[train_rows, train_cols]
        Z_previous = Z_np.copy()

        for epoch in range(self.n_epochs):
            # SVD on previous step Z_previous (from previous step)
            W_r = self.fit_transform(Z_previous)
            H_r = self.components_
            Z_previous = W_r @ H_r

            # Update matrix with known data
            Z_next = np.copy(Z_previous)
            Z_next[train_rows, train_cols] = Z_train

            # Comparing with previous step only on train set
            loss = np.sum((Z_next[train_rows, train_cols] - Z_previous[train_rows, train_cols]) ** 2)

            # Update next step
            Z_previous = Z_next

            if abs(loss) < 0.01:
                print(f"Number of performed epochs due to small error changes: {epoch}.")
                break

            if epoch % 10 == 0:
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
    from helper_functions import imputate_data_with_mean, map_ids
    from project_s329326_s331738.modules.build_train_matrix import build_train_set, build_test_set, convert_train_set_to_good_shape

    ratings = pd.read_csv("../data/ratings.csv")

    rt_train = build_train_set(ratings, 0.6)
    rt_test = build_test_set(ratings, rt_train)

    Z2_nt, usermap, moviemap = convert_train_set_to_good_shape(rt_train, rt_test)

    idx_test = map_ids(rt_test, usermap, moviemap)
    id_train = map_ids(rt_train, usermap, moviemap)

    # print(np.array(Z2_nt)[tuple(zip(*idx_test))])
    Z2_nt_imp = imputate_data_with_mean(Z2_nt)

    id_test_user, id_test_movie = tuple(zip(*idx_test))

    model = my_SVD2(n_components=4, n_epochs=1000)
    model.fit(Z2_nt_imp, id_train, verbose=True)
    print(model.get_recovered_Z())
    print(model.predict(id_test_user, id_test_movie))

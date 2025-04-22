from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from helper_functions import reshape_ratings_dataframe, imputate_data_with_0
# from tools.build_train_matrix import build_train_set


class my_SVD2(TruncatedSVD):

    def __init__(self, n_components=5, n_epochs=50, random_state=42):
        """
        Initialize the SVD1 model.

        :param n_components: Number of singular values/components to keep (rank r).
        :param random_state: Random seed.
        """
        super().__init__(n_components=n_components, random_state=random_state)
        self.n_epochs = n_epochs
        self.W_r = None
        self.H_r = None
        self.loss_list = []

    def fit(self, Z, train_df, verbose=True):
        """
        Fit the TruncatedSVD model to the input matrix Z, and compute W and H.

        :param Z: np.ndarray, shape (n_users, n_items). Fully populated (no missing entries) rating matrix.
        """
        train_array = np.array(train_df)
        Z_previous = np.copy(np.array(Z))
        Z_next = np.zeros((Z_previous.shape))

        for epoch in range(self.n_epochs):
            # SVD on previous step Z_previous (from previous step)
            W_r = super().fit_transform(Z_previous)
            H_r = self.components_
            Z_previous = W_r @ H_r

            # Update matrix with known data
            Z_next = np.copy(Z_previous)
            Z_next[train_df.index - 1, :] = train_array

            # Loss function - MSE
            loss = np.mean(np.square(Z_next - Z_previous))

            # update next step
            Z_previous = Z_next

            if epoch % 10 == 0:
                self.loss_list.append(loss)
                if verbose:
                    print(f"Epoch {epoch}: loss = {loss:.4f}")

        self.W_r = W_r
        self.H_r = H_r

        return self

    def get_recovered_Z(self):

        Z_recovered = np.matmul(self.W_r, self.H_r)
        Z_recovered_df = (2 * pd.DataFrame(Z_recovered)).round().clip(0.0, 5.0)

        return Z_recovered_df / 2


if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings.csv")
    Z2 = reshape_ratings_dataframe(ratings)
    Z2 = imputate_data_with_0(Z2)
    print(Z2)

    model = my_SVD2(n_components=10, random_state=42)
    train_set_df = pd.DataFrame(Z2).sample(n=200, random_state=42)
    model.fit(Z2, train_set_df)
    print(model.get_recovered_Z())

    # prediction = model.predict([1, 2, 3, 609, 0], [0, 1, 2, 3])
    # print(prediction)

import torch
import pandas as pd
import numpy as np
from pandas.core.algorithms import unique1d

from helper_functions import reshape_ratings_dataframe


class my_SGD:
    """
        SGD (Stochastic Gradient Descent) model

    """

    def __init__(self, lr=0.06, lmb=0, n_epochs=500, optimizer_name="Adam", device=None):
        self.lr = lr
        self.lmb = lmb
        self.n_epochs = n_epochs
        self.optimizer_name = optimizer_name
        self.W_r = None
        self.H_r = None
        self.device = device if device is not None else torch.device("cpu")
        self.loss_list = []

    def fit(self, Z, r, verbose=True):
        self.loss_list = []

        Z_tensor = torch.tensor(np.array(Z))
        W_r = torch.randn((Z_tensor.shape[0], r), requires_grad=True, dtype=torch.float, device=self.device)
        H_r = torch.randn((r, Z_tensor.shape[1]), requires_grad=True, dtype=torch.float, device=self.device)

        # subset of rows and columns dataframe with not nulls
        notnull_row_idx, notnull_col_idx = np.where(Z.notnull())

        unique_rows = torch.unique(torch.tensor(notnull_row_idx)).detach()
        unique_cols = torch.unique(torch.tensor(notnull_col_idx)).detach()
        print(unique_rows)

        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([W_r, H_r], lr=self.lr)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([W_r, H_r], lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

        for epoch in range(self.n_epochs):
            actual = Z_tensor[unique_rows, unique_cols]
            print(actual)
            predicted = torch.matmul(W_r[unique_rows, :], H_r[:, unique_cols])
            print(predicted.shape)

            MSE = torch.mean(torch.pow(actual - predicted, 2))

            reg_W = torch.sum(W_r[unique_rows] ** 2)
            reg_H = torch.sum(H_r[:, unique_cols] ** 2)

            regularization = self.lmb * (reg_W + reg_H)

            loss = MSE + regularization

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                self.loss_list.append(loss.item())
                if verbose:
                    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        self.W_r = W_r
        self.H_r = H_r

    def get_recovered_Z(self):
        return torch.matmul(self.W_r, self.H_r)


if __name__ == "__main__":
    # print(os.getcwd())
    Z = pd.read_csv("../data/ratings.csv")
    Z2 = reshape_ratings_dataframe(Z)

    model = my_SGD(lmb=0.3)
    model.fit(Z2, r=1)

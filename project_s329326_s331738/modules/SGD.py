import torch
import pandas as pd
import numpy as np
from helper_functions import reshape_ratings_dataframe, get_right_scale_rate


class my_SGD:
    """
    SGD (Stochastic Gradient Descent) model

    """

    def __init__(self, lr=0.06, lmb=0, n_epochs=500, optimizer_name="Adam", device=None):
        """
        Initializing of SGD model where chosen optimizer minimizes function:
        $\sum_{i,j: z[i, j] \neq NaN} (z[i, j] - W_i^T * H_j) ** 2 + \lmb * (||w_i||^2 + ||h_j||^2)$

        :param lr: learning rate
        :param lmb: ridge penalty rate, if lmb=0 then sum of squared errors is minimize
        :param n_epochs: number of repeats
        :param optimizer_name:
        :param device:
        """

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

        # select indexes where data is nan
        notnulls = ~Z_tensor.isnan()
        notnull_row_idx, notnull_col_idx = torch.where(notnulls)

        # select unique indexes where data is nan for slicing W_r, H_r
        unique_rows = torch.unique(notnull_row_idx)
        unique_cols = torch.unique(notnull_col_idx)

        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([W_r, H_r], lr=self.lr)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([W_r, H_r], lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

        for epoch in range(self.n_epochs):
            error = Z_tensor - torch.matmul(W_r, H_r)

            squared_error = (error[notnulls]) ** 2
            reg_W = torch.sum(W_r[unique_rows] ** 2)
            reg_H = torch.sum(H_r[:, unique_cols] ** 2)
            regularization = self.lmb * (reg_W + reg_H)

            loss = torch.sum(squared_error) + regularization

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
        """
        Recovering Z function with proper rounding
        :return:
        """
        Z_recovered_tensor = torch.matmul(self.W_r, self.H_r)
        Z_recovered_array = Z_recovered_tensor.detach().numpy()
        Z_recovered_df = (2 * get_right_scale_rate(pd.DataFrame(Z_recovered_array))).round()

        return Z_recovered_df / 2

    def predict(self, user_index, item_index):
        """
        To do
        :param user_index:
        :param item_index:
        :return:
        """


if __name__ == "__main__":
    # print(os.getcwd()) # show where you are to better write file track

    # Example of usage
    ratings = pd.read_csv("../data/ratings.csv")
    Z2 = reshape_ratings_dataframe(ratings)

    model = my_SGD(lmb=0.3, n_epochs=100)
    model.fit(Z2, r=200)
    print(model.get_recovered_Z())

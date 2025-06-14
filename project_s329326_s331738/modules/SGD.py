import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class my_SGD:
    """
    SGD (Stochastic Gradient Descent) model
    """

    def __init__(self, lr=0.001, lmb=0, n_components=5, n_epochs=100, batch_size=1024, optimizer_name="Adam",
                 device=None):

        self.lr = lr
        self.lmb = lmb
        self.r = n_components
        self.n_epochs = n_epochs
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.W_r = None
        self.H_r = None
        self.recovered_Z = None
        self.device = device if device is not None else torch.device("cpu")
        self.loss_list = []

    def fit(self, Z, id_train_set, verbose=False):
        self.loss_list = []

        Z_tensor = torch.tensor(np.array(Z), dtype=torch.float, device=self.device)
        W_r = torch.rand((Z_tensor.shape[0], self.r), requires_grad=True, dtype=torch.float, device=self.device)
        H_r = torch.rand((self.r, Z_tensor.shape[1]), requires_grad=True, dtype=torch.float, device=self.device)

        # select unique indexes where data is nan for slicing W_r, H_r
        train_rows, train_cols = zip(*id_train_set)
        train_rows_tensor = torch.tensor(train_rows, device=self.device)
        train_cols_tensor = torch.tensor(train_cols, device=self.device)

        Z_train = Z_tensor[train_rows_tensor, train_cols_tensor].to(self.device)

        # Dataset + DataLoader
        train_dataset = TensorDataset(train_rows_tensor, train_cols_tensor, Z_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([W_r, H_r], lr=self.lr)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([W_r, H_r], lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0

            for rows_batch, cols_batch, ratings_batch in train_loader:
                predictions = torch.sum(W_r[rows_batch] * H_r[:, cols_batch].T, dim=1)
                error = predictions - ratings_batch

                reg_W = torch.sum(W_r[rows_batch] ** 2)
                reg_H = torch.sum(H_r[:, cols_batch] ** 2)
                regularization = reg_W + reg_H

                loss = torch.sum((error ** 2)) + self.lmb * regularization

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            if epoch % 10 == 0:
                self.loss_list.append(epoch_loss)
                if verbose:
                    print(f"Epoch {epoch}: loss = {epoch_loss:.4f}")

            # if epoch_loss > self.loss_list[-1]:
            #     print(f"Number of performed epochs due to raise of error: {epoch}.")
            #     break

        self.W_r = W_r
        self.H_r = H_r

        return self

    def get_recovered_Z(self):
        with torch.no_grad():
            Z_recovered_tensor = torch.matmul(self.W_r, self.H_r)
        Z_recovered_array = Z_recovered_tensor.detach().numpy()
        Z_recovered = (2 * Z_recovered_array).round().clip(0.0, 10.0) / 2

        self.recovered_Z = Z_recovered

        return pd.DataFrame(self.recovered_Z)

    def predict(self, user_index, item_index):
        ids = zip(user_index, item_index)
        return self.recovered_Z[tuple(zip(*ids))]


if __name__ == "__main__":
    from helper_functions import reshape_ratings_dataframe, map_ids
    from project_s329326_s331738.modules.build_train_matrix import build_train_set

    # Example of usage
    ratings_file = "../data/ratings.csv"
    ratings = pd.read_csv(ratings_file)
    Z2, usermap, moviemap = reshape_ratings_dataframe(ratings)


    Z2_train = build_train_set(ratings, 0.8)
    id_train = map_ids(Z2_train, usermap, moviemap)

    model = my_SGD(lmb=0.0, lr=0.001, n_components=5, n_epochs=500, optimizer_name="Adam")
    # optimazer SGD good if lr smaller
    model.fit(Z2, id_train, verbose=True)
    mapped_u, mapped_m = zip(*id_train)
    model.get_recovered_Z()
    print(model.predict(mapped_u, mapped_m))

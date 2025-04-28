import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def initial_imputation(M):
    M_imputed = M.copy()
    user_means = np.nanmean(M, axis=1)
    inds = np.where(np.isnan(M_imputed))
    M_imputed[inds] = np.take(user_means, inds[0])  # podstaw średnią użytkownika
    return M_imputed


def pca_imputer(M, n_components=2, n_iter=10):
    # Wstępna imputacja
    M_imputed = initial_imputation(M)

    for iteration in range(n_iter):
        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42)
        M_projected = pca.fit_transform(M_imputed)
        M_reconstructed = pca.inverse_transform(M_projected)
        print(pca.singular_values_)

        # Uzupełnij tylko brakujące miejsca
        missing_mask = np.isnan(M)
        M_imputed[missing_mask] = M_reconstructed[missing_mask]
    return M_imputed


if __name__ == "__main__":
    from helper_functions import reshape_ratings_dataframe
    ratings = pd.read_csv("../data/ratings.csv")

    Z, _, _ = reshape_ratings_dataframe(ratings)
    Z = np.array(Z)

    imputed = pca_imputer(Z)
    #print(imputed)

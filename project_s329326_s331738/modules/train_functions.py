import pandas as pd
import numpy as np
from .NMF import my_NMF
from .SGD import my_SGD
from .SVD1 import my_SVD1
from .SVD2 import my_SVD2
from .helper_functions import reshape_ratings_dataframe, imputate_data_with_0, map_ids
from .build_train_matrix import build_train_set


def train_model(train_file, model_type, n_components=5):
    Z, usermap, moviemap = reshape_ratings_dataframe(train_file)
    Z_imputed = imputate_data_with_0(Z)

    if model_type == "NMF":
        model = my_NMF(n_components=n_components, init='random')
        model.fit(Z_imputed)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

    if model_type == "SVD1":
        model = my_SVD1(n_components=n_components)
        model.fit(Z_imputed)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

    if model_type == "SVD2":
        Z_train = build_train_set(pd.read_csv(train_file), 0.8)
        id_train = map_ids(Z_train, usermap, moviemap)

        model = my_SVD2(n_components=n_components)
        model.fit(Z, id_train_set=id_train)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

    if model_type == "SGD":
        Z_train = build_train_set(pd.read_csv(train_file), 0.8)
        id_train = map_ids(Z_train, usermap, moviemap)

        model = my_SGD(n_components=n_components)
        model.fit(Z, id_train_set=id_train)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

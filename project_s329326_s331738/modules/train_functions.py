import pandas as pd
from .NMF import my_NMF
from .SGD import my_SGD
from .SVD1 import my_SVD1
from .SVD2 import my_SVD2
from .helper_functions import imputate_data_with_0, map_ids
from .build_train_matrix import reshape_ratings_dataframe


def train_model(train_file, model_type, n_components=5):
    ratings = pd.read_csv(train_file)

    Z_train, usermap, moviemap = reshape_ratings_dataframe(ratings)

    Z_imputed = imputate_data_with_0(Z_train)
    idx_train = map_ids(Z_train, usermap, moviemap)

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
        model = my_SVD2(n_components=n_components)
        model.fit(Z_imputed, id_train_set=idx_train)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

    if model_type == "SGD":
        model = my_SGD(n_components=n_components)
        model.fit(Z_train, id_train_set=idx_train)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

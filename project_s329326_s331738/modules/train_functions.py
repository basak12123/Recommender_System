import pandas as pd
import numpy as np
from .NMF import my_NMF
from .SGD import my_SGD
from .SVD1 import my_SVD1
from .SVD2 import my_SVD2
from .helper_functions import imputate_data_with_0, map_ids
from .build_train_matrix import build_train_set, build_test_set, convert_train_set_to_good_shape


def train_model(train_file, model_type, n_components=5):
    ratings = pd.read_csv(train_file)

    rt_train = build_train_set(ratings, 0.8)
    rt_test = build_test_set(ratings, rt_train)

    rt_train_gd, usermap, moviemap = convert_train_set_to_good_shape(rt_train, rt_test)

    Z_imputed = imputate_data_with_0(rt_train_gd)
    idx_train = map_ids(rt_train, usermap, moviemap)

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
        model.fit(rt_train_gd, id_train_set=idx_train)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

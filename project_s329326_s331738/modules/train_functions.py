import pandas as pd
import numpy as np
from .NMF import my_NMF
from .SGD import my_SGD
from .SVD1 import my_SVD1
from .SVD2 import my_SVD2
from .helper_functions import reshape_ratings_dataframe, imputate_data_with_0

def train_model(train_file, model_type, idx = None, n_components=5):
    Z = reshape_ratings_dataframe(train_file)
    Z = imputate_data_with_0(Z)

    if model_type == "NMF":
        model = my_NMF(n_components=n_components, init='random')
        model.fit(Z)
        model.get_recovered_Z()
        return model.recovered_Z

    if model_type == "SVD1":
        model = my_SVD1(n_components=n_components)
        model.fit(Z)
        model.get_recovered_Z()
        return model.recovered_Z

    if model_type == "SVD2":
        model = my_SVD2(n_components=n_components)
        model.fit(Z, id_train_set=idx)
        model.get_recovered_Z()
        return model.recovered_Z

    if model_type == "SGD":
        model = my_SGD(n_components=n_components)
        model.fit(Z, id_train_set=idx)
        model.get_recovered_Z()
        return model.recovered_Z


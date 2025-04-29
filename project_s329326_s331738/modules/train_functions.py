import pandas as pd
from .NMF import my_NMF
from .SGD import my_SGD
from .SVD1 import my_SVD1
from .SVD2 import my_SVD2
from .helper_functions import imputate_data_with_PCA, map_ids
from .build_train_matrix import reshape_ratings_dataframe, build_train_set


def train_model(train_file, model_type):
    NMF_components = 50
    SVD1_components = 30
    SVD2_components = 2
    SGD_components = 1
    ratings = pd.read_csv(train_file)

    r_train = build_train_set(ratings, 1)
    Z_train, usermap, moviemap = reshape_ratings_dataframe(ratings)

    idx_train = map_ids(r_train, usermap, moviemap)
    Z_imputed = imputate_data_with_PCA(Z_train)


    if model_type == "NMF":
        model = my_NMF(n_components=NMF_components, init='random')
        model.fit(Z_imputed)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

    if model_type == "SVD1":
        model = my_SVD1(n_components=SVD1_components)
        model.fit(Z_imputed)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

    if model_type == "SVD2":
        model = my_SVD2(n_components=SVD2_components)
        model.fit(Z_imputed, id_train_set=idx_train)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

    if model_type == "SGD":
        model = my_SGD(n_components=SGD_components)
        model.fit(Z_train, id_train_set=idx_train)
        model.get_recovered_Z()
        return model.recovered_Z, usermap, moviemap

    if model_type == "ALL":
        model1 = my_NMF(n_components=NMF_components, init='random')
        model2 = my_SVD1(n_components=SVD1_components)
        model3 = my_SVD2(n_components=SVD2_components)
        model4 = my_SGD(n_components=SGD_components)

        model1.fit(Z_imputed)
        model2.fit(Z_imputed)
        model3.fit(Z_imputed, id_train_set=idx_train)
        model4.fit(Z_train, id_train_set=idx_train)

        model1.get_recovered_Z()
        model2.get_recovered_Z()
        model3.get_recovered_Z()
        model4.get_recovered_Z()
        return ({"NMF": model1.recovered_Z, "SVD1": model2.recovered_Z, "SVD2": model3.recovered_Z, "SGD": model4.recovered_Z},
                usermap, moviemap)

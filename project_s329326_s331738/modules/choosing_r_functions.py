from SGD import my_SGD
from SVD2 import my_SVD2
from SVD1 import my_SVD1
from NMF import my_NMF
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from project_s329326_s331738.modules.build_train_matrix import (get_id_of_full_data, split_train_test,
                                                                convert_train_set_to_good_shape, build_test_set)
from helper_functions import (reshape_ratings_dataframe, imputate_data_with_0, imputate_data_with_mean, map_ids,
                              imputate_data_with_mean_of_user, imputate_data_with_KNN, imputate_data_with_PCA)

# DATA
ratings = pd.read_csv("../data/ratings.csv")
Z_full, usermap, moviemap = reshape_ratings_dataframe(ratings)


Z_imputed0 = imputate_data_with_0(Z_full)

ratings = pd.read_csv("../data/ratings.csv")

# ALL FULL DATA
id_user_movies = get_id_of_full_data(ratings)

# PARAM GRID
SGD_param_grid = {
    'n_components': [5, 10, 50, 100, 150, 200],
    'lmb': [0.0]
}

param_grid = {
     'n_components': [5, 10, 50, 100, 150, 200],
}


def GridSearchCV(ratings, grid_param, num_of_kfolds, type_of_model="SGD", imputing_style="fill_with_0"):

    kf = split_train_test(ratings, num_of_kfolds)

    ParamGridObj = ParameterGrid(grid_param)

    if type_of_model.lower() == "sgd":
        Stats_AvgRMSE = np.zeros((len(ParamGridObj), 3))
        Stats_FoldsRMSE = np.zeros((len(ParamGridObj), num_of_kfolds + 2))
    else:
        Stats_AvgRMSE = np.zeros((len(ParamGridObj), 2))
        Stats_FoldsRMSE = np.zeros((len(ParamGridObj), num_of_kfolds + 1))
    i = 0

    for params in ParamGridObj:
        print(f"Trying parameters: {params}")

        fold_scores = []

        for j in range(num_of_kfolds):

            test_set = kf[j]
            train_set = build_test_set(ratings, test_set)

            Z_train_gd, usermap, moviemap = convert_train_set_to_good_shape(train_set, test_set)

            if imputing_style.lower() == "fill_with_0":
                Z_train_gd_imputeted = imputate_data_with_0(Z_train_gd)

            if imputing_style.lower() == "fill_with_mean":
                Z_train_gd_imputeted = imputate_data_with_mean(Z_train_gd)

            if imputing_style.lower() == "fill_with_mean_by_users":
                Z_train_gd_imputeted = imputate_data_with_mean_of_user(Z_train_gd)

            if imputing_style.lower() == "fill_with_knn":
                Z_train_gd_imputeted = imputate_data_with_KNN(Z_train_gd)

            if imputing_style.lower() == "fill_with_pca":
                Z_train_gd_imputeted = imputate_data_with_PCA(Z_train_gd)

            idx_test = map_ids(test_set, usermap, moviemap)
            idx_train = map_ids(train_set, usermap, moviemap)

            idx_test_user, idx_test_movie = tuple(zip(*idx_test))

            if type_of_model.lower() == "sgd":
                model = my_SGD(
                    lr=0.001,
                    lmb=params['lmb'],
                    n_components=params['n_components'],
                    optimizer_name="Adam",
                    n_epochs=200
                )
                model.fit(Z_train_gd, idx_train, verbose=False)
                model.get_recovered_Z()

                preds = model.predict(list(idx_test_user), list(idx_test_movie))

            elif type_of_model.lower() == "svd1":
                model = my_SVD1(n_components=params['n_components'])
                model.fit(Z_train_gd_imputeted)
                model.get_recovered_Z()

                preds = model.predict(idx_test)

            elif type_of_model.lower() == "svd2":
                model = my_SVD2(n_components=params['n_components'])
                model.fit(Z_train_gd_imputeted, idx_train, verbose=False)
                model.get_recovered_Z()

                preds = model.predict(list(idx_test_user), list(idx_test_movie))

            elif type_of_model.lower() == "nmf":
                model = my_NMF(n_components=params['n_components'])
                model.fit(Z_train_gd_imputeted)
                model.get_recovered_Z()

                preds = model.predict(idx_test)

            rmse = np.sqrt(mean_squared_error(test_set['rating'], preds))

            fold_scores.append(rmse)

        mean_rmse_of_fold = np.mean(fold_scores)

        if type_of_model.lower() == "sgd":
            Stats_AvgRMSE[i, ] = [params['lmb'], params['n_components'], mean_rmse_of_fold]
            Stats_FoldsRMSE[i, ] = [params['lmb'], params['n_components'], *fold_scores]

            df_Stats_AvgRMSE = pd.DataFrame(Stats_AvgRMSE, columns=('lmb', 'r_component', 'mean_rmse'))
            df_Stats_FoldsRMSE = pd.DataFrame(Stats_FoldsRMSE,
                                              columns=('lmb', 'r_component',
                                                       *[f'rmse_fold{i}' for i in range(1, num_of_kfolds + 1)]))
        else:
            Stats_AvgRMSE[i,] = [params['n_components'], mean_rmse_of_fold]
            Stats_FoldsRMSE[i,] = [params['n_components'], *fold_scores]

            df_Stats_AvgRMSE = pd.DataFrame(Stats_AvgRMSE, columns=('r_component', 'mean_rmse'))
            df_Stats_FoldsRMSE = pd.DataFrame(Stats_FoldsRMSE,
                                              columns=('r_component',
                                                       *[f'rmse_fold{i}' for i in range(1, num_of_kfolds + 1)]))

        i += 1

    return df_Stats_AvgRMSE, df_Stats_FoldsRMSE


# SGD :

# # with zeros:
# Stats_AvgRMSE_SGD, Stats_FoldsRMSE_SGD = GridSearchCV(ratings, SGD_param_grid, num_of_kfolds=5, type_of_model="sgd", imputing_style="fill_with_0")
# print(Stats_AvgRMSE_SGD)
# print(Stats_FoldsRMSE_SGD)
#
# Stats_AvgRMSE_SGD.to_csv("../data/grid_search_AvgRMSE_SGD_2.csv", index=False)
# Stats_FoldsRMSE_SGD.to_csv("../data/grid_search_FoldsRMSE_SGD_2.csv", index=False)

# with mean:

Stats_AvgRMSE_SGD, Stats_FoldsRMSE_SGD = GridSearchCV(ratings, SGD_param_grid, num_of_kfolds=5, type_of_model="sgd", imputing_style="fill_with_means")
print(Stats_AvgRMSE_SGD)
print(Stats_FoldsRMSE_SGD)

# Stats_AvgRMSE_SGD.to_csv("../data/grid_search_AvgRMSE_SGD_mean.csv", index=False)
# Stats_FoldsRMSE_SGD.to_csv("../data/grid_search_FoldsRMSE_SGD_mean.csv", index=False)

# # SVD2 :
#
# Stats_AvgRMSE_SVD2, Stats_FoldsRMSE_SVD2 = GridSearchCV(Z_imputed0, kf, param_grid, num_of_kfolds=5, type_of_model="svd2")
# print(Stats_AvgRMSE_SVD2)
# print(Stats_FoldsRMSE_SVD2)
#
# Stats_AvgRMSE_SVD2.to_csv("../data/grid_search_AvgRMSE_SVD2.csv", index=False)
# Stats_FoldsRMSE_SVD2.to_csv("../data/grid_search_FoldsRMSE_SVD2.csv", index=False)

# SVD1 :

# with zeros:

# Stats_AvgRMSE_SVD1, Stats_FoldsRMSE_SVD1 = GridSearchCV(ratings, param_grid, num_of_kfolds=5, type_of_model="svd1", imputing_style="fill_with_0")
# print(Stats_AvgRMSE_SVD1)
# print(Stats_AvgRMSE_SVD1)

# Stats_AvgRMSE_SVD1.to_csv("../data/grid_search_AvgRMSE_SVD1.csv", index=False)
# Stats_FoldsRMSE_SVD1.to_csv("../data/grid_search_FoldsRMSE_SVD1.csv", index=False)

# with means:

# Stats_AvgRMSE_SVD1, Stats_FoldsRMSE_SVD1 = GridSearchCV(ratings, param_grid, num_of_kfolds=5, type_of_model="svd1", imputing_style="fill_with_mean")
# print(Stats_AvgRMSE_SVD1)
# print(Stats_FoldsRMSE_SVD1)
#
# Stats_AvgRMSE_SVD1.to_csv("../data/grid_search_AvgRMSE_SVD1_mean.csv", index=False)
# Stats_FoldsRMSE_SVD1.to_csv("../data/grid_search_FoldsRMSE_SVD1_mean.csv", index=False)

# with means by users:

#Stats_AvgRMSE_SVD1, Stats_FoldsRMSE_SVD1 = GridSearchCV(ratings, param_grid, num_of_kfolds=5, type_of_model="svd1", imputing_style="fill_with_pca")
#print(Stats_AvgRMSE_SVD1)
#print(Stats_FoldsRMSE_SVD1)

#Stats_AvgRMSE_SVD1.to_csv("../data/grid_search_AvgRMSE_SVD1_pca.csv", index=False)
#Stats_FoldsRMSE_SVD1.to_csv("../data/grid_search_FoldsRMSE_SVD1_pca.csv", index=False)

# NMF:

# # with zero
# Stats_AvgRMSE_NMF, Stats_FoldsRMSE_NMF = GridSearchCV(ratings, param_grid, num_of_kfolds=5, type_of_model="nmf", imputing_style="fill_with_0")
# print(Stats_AvgRMSE_NMF)
# print(Stats_AvgRMSE_NMF)
#
# Stats_AvgRMSE_NMF.to_csv("../data/grid_search_AvgRMSE_NMF.csv", index=False)
# Stats_FoldsRMSE_NMF.to_csv("../data/grid_search_FoldsRMSE_NMF.csv", index=False)

# with mean
# Stats_AvgRMSE_NMF, Stats_FoldsRMSE_NMF = GridSearchCV(ratings, param_grid, num_of_kfolds=5, type_of_model="nmf", imputing_style="fill_with_mean")
# print(Stats_AvgRMSE_NMF)
# print(Stats_FoldsRMSE_NMF)
#
# Stats_AvgRMSE_NMF.to_csv("../data/grid_search_AvgRMSE_NMF_mean.csv", index=False)
# Stats_FoldsRMSE_NMF.to_csv("../data/grid_search_FoldsRMSE_NMF_mean.csv", index=False)

# with mean user:
# Stats_AvgRMSE_NMF, Stats_FoldsRMSE_NMF = GridSearchCV(ratings, param_grid, num_of_kfolds=5, type_of_model="nmf", imputing_style="fill_with_mean_by_users")
# print(Stats_AvgRMSE_NMF)
# print(Stats_FoldsRMSE_NMF)
#
# Stats_AvgRMSE_NMF.to_csv("../data/grid_search_AvgRMSE_NMF_mean_user.csv", index=False)
# Stats_FoldsRMSE_NMF.to_csv("../data/grid_search_FoldsRMSE_NMF_mean_user.csv", index=False)

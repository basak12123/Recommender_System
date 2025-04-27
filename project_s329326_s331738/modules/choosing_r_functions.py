from SGD import my_SGD
from SVD2 import my_SVD2
from SVD1 import my_SVD1
from NMF import my_NMF
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from project_s329326_s331738.modules.build_train_matrix import get_id_of_full_data
from helper_functions import reshape_ratings_dataframe, imputate_data_with_0, map_ids, reshape_ratings

# DATA
ratings = pd.read_csv("../data/ratings.csv")
Z_full, usermap, moviemap = reshape_ratings_dataframe("../data/ratings.csv")

Z_imputed0 = imputate_data_with_0(Z_full)


# ALL FULL DATA
id_user_movies = get_id_of_full_data(ratings)

# PARAM GRID
SGD_param_grid = {
    'n_components': [5, 10, 50, 100, 200],
    'lmb': [0.0]
}

param_grid = {
     'n_components': [5, 10, 50, 100, 200, 250, 300, 400, 500, 600],
}

num_of_kfolds = 5
kf = KFold(n_splits=num_of_kfolds, shuffle=True, random_state=42)


def GridSearchCV(Z, kf, grid_param, num_of_kfolds, type_of_model="SGD"):

    ParamGridObj = ParameterGrid(grid_param)
    n, d = Z.shape
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
        id_array = np.array(map_ids(id_user_movies, usermap, moviemap))
        Z_np = np.array(Z)

        for train_index, test_index in kf.split(id_array):
            id_train_fold = id_array[train_index.astype(int)].tolist()
            id_test_fold = id_array[test_index.astype(int)].tolist()

            id_test_user, id_test_movie = tuple(zip(*id_test_fold))
            id_train_user, id_train_movie = tuple(zip(*id_train_fold))

            ratings_test_fold = Z_np[id_test_user, id_test_movie]
            ratings_train_fold, u_mp, id_mp = reshape_ratings(pd.DataFrame(Z_np[id_train_user, id_train_movie]))
            ratings_train_fold = imputate_data_with_0(ratings_train_fold)

            if type_of_model.lower() == "sgd":
                model = my_SGD(
                    lmb=params['lmb'],
                    n_components=params['n_components'],
                    optimizer_name="Adam",
                    n_epochs=100
                )
                model.fit(Z, id_train_fold, verbose=False)
                model.get_recovered_Z()

                preds = model.predict(list(id_test_user), list(id_test_movie))

            elif type_of_model.lower() == "svd1":
                model = my_SVD1(n_components=params['n_components'])
                model.fit(ratings_train_fold)
                model.get_recovered_Z()

                preds = model.predict(id_test_fold)

            elif type_of_model.lower() == "svd2":
                model = my_SVD2(n_components=params['n_components'])
                model.fit(Z, id_train_fold, verbose=False)
                model.get_recovered_Z()

                preds = model.predict(list(id_test_user), list(id_test_movie))

            elif type_of_model.lower() == "nmf":
                model = my_NMF(n_components=params['n_components'])
                model.fit(Z)
                model.get_recovered_Z()

                preds = model.predict(id_test_fold)

            rmse = np.sqrt(mean_squared_error(ratings_test_fold.flatten(), preds))

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

# Stats_AvgRMSE_SGD, Stats_FoldsRMSE_SGD = GridSearchCV(Z_full, kf, SGD_param_grid, num_of_kfolds=5, type_of_model="sgd")
# print(Stats_AvgRMSE_SGD)
# print(Stats_FoldsRMSE_SGD)
#
# Stats_AvgRMSE_SGD.to_csv("../data/grid_search_AvgRMSE_SGD.csv", index=False)
# Stats_FoldsRMSE_SGD.to_csv("../data/grid_search_FoldsRMSE_SGD.csv", index=False)
#
# # SVD2 :
#
# Stats_AvgRMSE_SVD2, Stats_FoldsRMSE_SVD2 = GridSearchCV(Z_imputed0, kf, param_grid, num_of_kfolds=5, type_of_model="svd2")
# print(Stats_AvgRMSE_SVD2)
# print(Stats_FoldsRMSE_SVD2)
#
# Stats_AvgRMSE_SVD2.to_csv("../data/grid_search_AvgRMSE_SVD2.csv", index=False)
# Stats_FoldsRMSE_SVD2.to_csv("../data/grid_search_FoldsRMSE_SVD2.csv", index=False)

# SVD1 :

Stats_AvgRMSE_SVD1, Stats_FoldsRMSE_SVD1 = GridSearchCV(Z_imputed0, kf, param_grid, num_of_kfolds=5, type_of_model="svd1")
print(Stats_AvgRMSE_SVD1)
print(Stats_AvgRMSE_SVD1)

Stats_AvgRMSE_SVD1.to_csv("../data/grid_search_AvgRMSE_SVD1.csv", index=False)
Stats_FoldsRMSE_SVD1.to_csv("../data/grid_search_FoldsRMSE_SVD1.csv", index=False)

# NMF:

# Stats_AvgRMSE_NMF, Stats_FoldsRMSE_NMF = GridSearchCV(Z_imputed0, kf, param_grid, num_of_kfolds=5, type_of_model="nmf")
# print(Stats_AvgRMSE_NMF)
# print(Stats_AvgRMSE_NMF)
#
# Stats_AvgRMSE_NMF.to_csv("../data/grid_search_AvgRMSE_NMF.csv", index=False)
# Stats_FoldsRMSE_NMF.to_csv("../data/grid_search_FoldsRMSE_NMF.csv", index=False)

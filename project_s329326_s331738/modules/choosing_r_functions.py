from SGD import my_SGD
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from operator import itemgetter
from project_s329326_s331738.modules.build_train_matrix import get_id_of_full_data
from helper_functions import reshape_ratings_dataframe, imputate_data_with_0

# DATA
Z_full, usermap, moviemap = reshape_ratings_dataframe("../data/ratings.csv")

Z_imputed0 = imputate_data_with_0(Z_full)


# ALL FULL DATA
notnull_row_idx, notnull_col_idx = get_id_of_full_data(Z_full)
id_not_nulls = [i for i in zip(notnull_row_idx, notnull_col_idx)][1:20]

# PARAM GRID
SGD_param_grid = {
    'n_components': [4, 6, 8],
    'lmb': [0.0, 0.001, 0.01]
}

SVD2_param_grid = {
     'n_components': [4, 6, 8],
}

num_of_kfolds = 5
kf = KFold(n_splits=num_of_kfolds, shuffle=True, random_state=42)


def GridSearchCV(Z, kf, grid_param, num_of_kfolds):

    ParamGridObj = ParameterGrid(grid_param)
    Stats_AvgRMSE = np.zeros((len(ParamGridObj), 3))
    Stats_FoldsRMSE = np.zeros((len(ParamGridObj), num_of_kfolds + 2))
    i = 0

    for params in ParamGridObj:
        print(f"Trying parameters: {params}")

        fold_scores = []
        id_array = np.array(id_not_nulls)
        Z_np = np.array(Z)

        for train_index, test_index in kf.split(id_array):
            id_train_fold = id_array[train_index.astype(int)].tolist()
            id_test_fold = id_array[test_index.astype(int)].tolist()

            id_test_user, id_test_movie = tuple(zip(*id_test_fold))

            user_idx = itemgetter(*id_test_user)(usermap)
            movie_idx = itemgetter(*id_test_movie)(moviemap)

            ratings_test_fold = Z_np[user_idx, movie_idx]

            model = my_SGD(
                lmb=params['lmb'],
                n_components=params['n_components'],
                optimizer_name="SGD"
            )

            model.fit(Z, id_train_fold, verbose=False)
            model.get_recovered_Z()

            user_idx = itemgetter(*id_test_user)(usermap)
            movie_idx = itemgetter(*id_test_movie)(moviemap)

            preds = model.predict(list(user_idx), list(movie_idx))
            rmse = np.sqrt(mean_squared_error(ratings_test_fold, preds))

            fold_scores.append(rmse)

        mean_rmse_of_fold = np.mean(fold_scores)

        Stats_AvgRMSE[i, ] = [params['lmb'], params['r_components'], mean_rmse_of_fold]
        Stats_FoldsRMSE[i, ] = [params['lmb'], params['r_components'], *fold_scores]
        i += 1

    df_Stats_AvgRMSE = pd.DataFrame(Stats_AvgRMSE, columns=('lmb', 'r_component', 'mean_rmse'))
    df_Stats_FoldsRMSE = pd.DataFrame(Stats_FoldsRMSE, columns=('lmb', 'r_component', *[f'rmse_fold{i}' for i in range(1, num_of_kfolds + 1)]))

    return df_Stats_AvgRMSE, df_Stats_FoldsRMSE


model = my_SGD()
grid_seach = GridSearchCV(Z_full, kf, SGD_param_grid, num_of_kfolds=num_of_kfolds)

print(grid_seach)


# Stats_AvgRMSE, Stats_FoldsRMSE = GridSearchCV_SGD(Z_full, SGD_param_grid, num_of_kfolds=5)
# print(Stats_AvgRMSE)
# print(Stats_FoldsRMSE)
#
# Stats_AvgRMSE.to_csv("../data/grid_search_SGD_AvgRMSE.csv", index=False)
# Stats_FoldsRMSE.to_csv("../data/grid_search_SGD_FoldsRMSE.csv", index=False)


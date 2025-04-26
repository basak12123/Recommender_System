from SGD import my_SGD
from sklearn.model_selection import KFold, ParameterGrid
import numpy as np
import pandas as pd
from project_s329326_s331738.modules.build_train_matrix import get_id_of_full_data
from helper_functions import reshape_ratings_dataframe, imputate_data_with_0

# DATA
ratings = pd.read_csv("../data/ratings.csv")

Z_full = reshape_ratings_dataframe(ratings)

Z_imputed0 = imputate_data_with_0(Z_full)

# ALL FULL DATA
notnull_row_idx, notnull_col_idx = get_id_of_full_data(Z_full)
id_not_nulls = [i for i in zip(notnull_row_idx, notnull_col_idx)][1:20]

# PARAM GRID
SGD_param_grid = {
    'r_components': [4, 6, 8],
    'lmb': [0.0, 0.001, 0.01]
}

SVD2_param_grid = {
     'n_components': [4, 6, 8],
}

num_of_kfolds = 5
kf = KFold(n_splits=num_of_kfolds, shuffle=True, random_state=42)


def GridSearchCV_SGD(Z, kf, grid_param, num_of_kfolds):

    # best_score = float('inf')
    # best_params = None
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

            ratings_test_fold = Z_np[tuple(zip(*id_test_fold))]

            model = my_SGD(
                lmb=params['lmb'],
                r_components=params['r_components'],
                optimizer_name="SGD"
            )

            model.fit(Z2, id_train_fold, verbose=False)
            model.get_recovered_Z()

            fold_scores.append(model.compute_RMSE_on_test(id_test_fold, ratings_test_fold))

        neg_mean_rmse = - np.mean(fold_scores)

        Stats_AvgRMSE[i, ] = [params['lmb'], params['r_components'], neg_mean_rmse]
        Stats_FoldsRMSE[i, ] = [params['lmb'], params['r_components'], *fold_scores]
        i += 1

    df_Stats_AvgRMSE = pd.DataFrame(Stats_AvgRMSE, columns=('lmb', 'r_component', 'neg_mean_rmse'))
    df_Stats_FoldsRMSE = pd.DataFrame(Stats_FoldsRMSE, columns=('lmb', 'r_component', *[f'rmse_fold{i}' for i in range(1, num_of_kfolds + 1)]))

    return df_Stats_AvgRMSE, df_Stats_FoldsRMSE


# model = my_SVD2()
# grid_seach = GridSearchCV(model, SVD2_param_grid, cv=kf, scoring='neg_root_mean_squared_error')
#
# print(grid_seach)


# Stats_AvgRMSE, Stats_FoldsRMSE = GridSearchCV_SGD(Z_full, SGD_param_grid, num_of_kfolds=5)
# print(Stats_AvgRMSE)
# print(Stats_FoldsRMSE)
#
# Stats_AvgRMSE.to_csv("../data/grid_search_SGD_AvgRMSE.csv", index=False)
# Stats_FoldsRMSE.to_csv("../data/grid_search_SGD_FoldsRMSE.csv", index=False)


import csv

from SGD import my_SGD
from sklearn.model_selection import KFold, ParameterGrid
import numpy as np
import pandas as pd
from tools.build_train_matrix import get_id_of_full_data
from helper_functions import reshape_ratings_dataframe

# DATA
ratings = pd.read_csv("../data/ratings.csv")
Z2 = reshape_ratings_dataframe(ratings)

# ALL FULL DATA
notnull_row_idx, notnull_col_idx = get_id_of_full_data(Z2)
id_not_nulls = [i for i in zip(notnull_row_idx, notnull_col_idx)][1:20]

# PARAM GRID
SGD_param_grid = {
    'r_components': [4, 6, 8],
    'lmb': [0.0, 0.001, 0.01]
}


def GridSearchCV_SGD(Z, grid_param, num_of_kfolds):
    kf = KFold(n_splits=num_of_kfolds, shuffle=True, random_state=42)

    best_score = float('inf')
    best_params = None
    Stats = {}

    for params in ParameterGrid(grid_param):
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

        mean_rmse = np.mean(fold_scores)
        print(f"Mean RMSE across folds: {mean_rmse}")

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

        Stats[*params.values()] = {"fold_scores": fold_scores, "mean_rmse": mean_rmse}

    Stats['best_score'] = best_score
    Stats['best_params'] = best_params
    return Stats


# grid_search = GridSearchCV_SGD(Z2, SGD_param_grid, num_of_kfolds=5)
# # print(grid_search[(0.0, 4)])
#
# csv_filename = "../data/grid_search_SVD.csv"
#
# # Writing single dictionary to CSV
# with open(csv_filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(grid_search.keys())  # Write header
#     writer.writerow(grid_search.values())  # Write values


with open("../data/grid_search_SVD.csv", 'r') as f:
    dict_reader = csv.DictReader(f)
    list_of_dict = list(dict_reader)
    g = list_of_dict[0]

print(g['(0.0, 4)'])

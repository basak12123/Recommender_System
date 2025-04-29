from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from helper_functions import reshape_ratings_dataframe, imputate_data_with_0, map_ids
from project_s329326_s331738.modules.build_train_matrix import build_train_set, build_test_set, convert_train_set_to_good_shape
import pandas as pd
import numpy as np

ratings = pd.read_csv("../data/ratings.csv")

user_ids = ratings['userId'].unique()
item_ids = ratings['movieId'].unique()
user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {m: j for j, m in enumerate(item_ids)}

n_users = len(user_ids)
n_items = len(item_ids)

M = np.full((n_users, n_items), np.nan)

for _, row in ratings.iterrows():
    i = user_map[row.userId]
    j = item_map[row.movieId]
    M[i, j] = row.rating

imputer = KNNImputer(
    n_neighbors=5,
    weights='distance',    # cięższe znaczenie najbliższych
    metric='nan_euclidean' # domyślna; liczy odległości ignorując nan
)

# Dopasuj i przekształć macierz
M_imputed = imputer.fit_transform(M)
print(np.round(M_imputed * 2)/2)

def round_half(x):
    return np.round(x * 2) / 2

def evaluate_knn_imputer(M, k, Z, mask_test):
    imputer = KNNImputer(n_neighbors=k, weights='distance', keep_empty_features=True)
    M_imputed = imputer.fit_transform(np.array(M))
    M_imputed = round_half(M_imputed)

    # Liczymy RMSE tylko na ukrytych miejscach
    y_true = np.array(Z)
    y_true = y_true[tuple(zip(*mask_test))]
    y_pred = M_imputed[tuple(zip(*mask_test))]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

rt_train = build_train_set(ratings, 0.8)
rt_test = build_test_set(ratings, rt_train)
Z, _, _ = reshape_ratings_dataframe(ratings)
Z2_nt, usermap, moviemap = convert_train_set_to_good_shape(rt_train, rt_test)

Z2_nt_imp = imputate_data_with_0(Z2_nt)
idx_test = map_ids(rt_test, usermap, moviemap)
idx_train = map_ids(rt_train, usermap, moviemap)

k_list = [i for i in range(1, 16)]
results = {}

for k in k_list:
    rmse = evaluate_knn_imputer(Z2_nt, k, Z, idx_test)
    results[k] = rmse
    print(f"k={k}, RMSE={rmse:.4f}")

best_k = min(results, key=results.get)
print(f"Najlepsze k: {best_k}")

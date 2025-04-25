import random

import pandas as pd
import numpy as np
import os
from project_s329326_s331738.modules.helper_functions import reshape_ratings_dataframe


def get_id_of_full_data(df):
    array_sample = np.array(df)
    notnulls = ~np.isnan(array_sample)
    return np.where(notnulls)


def build_train_set(df, size_of_train_test):
    notnull_row_idx, notnull_col_idx = get_id_of_full_data(df)

    id_not_nulls = [i for i in zip(notnull_row_idx, notnull_col_idx)]
    id_trains = random.choices(id_not_nulls, k=size_of_train_test)

    return id_trains, np.array(df)[tuple(zip(*id_trains))]


def build_test_set(df, id_train):
    notnull_row_idx, notnull_col_idx = get_id_of_full_data(df)

    id_not_nulls = [i for i in zip(notnull_row_idx, notnull_col_idx)]
    id_test = list(set(id_not_nulls) - set(id_train))

    return id_test, np.array(df)[tuple(zip(*id_test))]


if __name__ == "__main__":
    # print(os.getcwd())

    ratings = pd.read_csv("../project_s329326_s331738/data/ratings.csv")
    Z2 = reshape_ratings_dataframe(ratings)

    id_train, Z2_train_ratings = build_train_set(Z2, 60000)
    id_test, Z2_test_ratings = build_test_set(Z2, id_train)

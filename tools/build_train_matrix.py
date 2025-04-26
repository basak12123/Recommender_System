import random

import pandas as pd
import numpy as np
import os
from project_s329326_s331738.modules.helper_functions import reshape_ratings_dataframe


def get_id_of_full_data(df):
    """
    Get ids of those pairs (user_id, movie_id) from reshaped data frame which
    are not censored data.
    :param df: DataFrame object which was transformed by reshape_ratings_dataframe function
    :return: tuple of arrays with user_id and movie_id coordinates where the data is full
    """
    array_sample = np.array(df)
    notnulls = ~np.isnan(array_sample)
    return np.where(notnulls)


def build_train_set(df, perc_of_data_to_train):
    """
    Build train set based on NOT IMPUTED reshaped data frame.
    :param df: DataFrame object which was transformed by reshape_ratings_dataframe function (NOT IMPUTED)
    :param perc_of_data_to_train: fraction of observations in train set
    :return: tuples with pairs (user_id, movie_id) and ratings correspond to those pairs
    """
    notnull_row_idx, notnull_col_idx = get_id_of_full_data(df)
    size_of_train_test = int(np.floor(perc_of_data_to_train * len(notnull_row_idx)))

    id_not_nulls = [i for i in zip(notnull_row_idx, notnull_col_idx)]
    id_trains = random.choices(id_not_nulls, k=size_of_train_test)

    return id_trains, np.array(df)[tuple(zip(*id_trains))]


def build_test_set(df, id_train):
    """
    Build test set. Use this function after using build_train_set.
    :param df: DataFrame object which was transformed by reshape_ratings_dataframe function (NOT IMPUTED)
    :param id_train: Pairs (user_id, movie_id) which are in train set (first component of tuple from
    function build_train_set)
    :return: tuples with pairs (user_id, movie_id) and ratings correspond to those pairs
    """
    notnull_row_idx, notnull_col_idx = get_id_of_full_data(df)

    id_not_nulls = [i for i in zip(notnull_row_idx, notnull_col_idx)]
    id_test = list(set(id_not_nulls) - set(id_train))

    return id_test, np.array(df)[tuple(zip(*id_test))]


if __name__ == "__main__":
    # print(os.getcwd())

    # Example of usage
    ratings = pd.read_csv("../project_s329326_s331738/data/ratings.csv")
    Z2 = reshape_ratings_dataframe(ratings)

    id_train, Z2_train_ratings = build_train_set(Z2, 0.6)
    id_test, Z2_test_ratings = build_test_set(Z2, id_train)
    print(len(id_train))

import random

import pandas as pd
import numpy as np
import os
from .helper_functions import reshape_ratings_dataframe


def get_id_of_full_data(rating_df):
    """
    Get ids of those pairs (user_id, movie_id) from reshaped data frame which
    are not censored data.
    :param df: DataFrame object which was transformed by reshape_ratings_dataframe function
    :return: tuple of arrays with user_id and movie_id coordinates where the data is full
    """
    id_user_movie = rating_df[['userId', 'movieId']]
    return id_user_movie


def build_train_set(df, perc_of_data_to_train):
    """
    Build train set based on NOT IMPUTED reshaped data frame.
    :param df: DataFrame object which was transformed by reshape_ratings_dataframe function (NOT IMPUTED)
    :param perc_of_data_to_train: fraction of observations in train set
    :return: tuples with pairs (user_id, movie_id) and ratings correspond to those pairs
    """
    id_user_movie = get_id_of_full_data(df)
    size_of_train_test = int(np.floor(perc_of_data_to_train * id_user_movie.shape[0]))

    train_df = df.sample(n=size_of_train_test)

    return train_df[['userId', 'movieId', 'rating']]


def build_test_set(df, train_df):
    """
    Build test set. Use this function after using build_train_set.
    :param df: DataFrame object which was transformed by reshape_ratings_dataframe function (NOT IMPUTED)
    :param train_df: Pairs (user_id, movie_id) which are in train set (first component of tuple from
    function build_train_set)
    :return: tuples with pairs (user_id, movie_id) and ratings correspond to those pairs
    """
    df_test = df.merge(train_df[['userId', 'movieId']], on=['userId', 'movieId'], how='left', indicator=True)
    df_test = df_test[df_test['_merge'] == 'left_only']
    df_test = df_test[['userId', 'movieId', 'rating']]

    return df_test


def split_train_test(df, num_of_splits):
    df_copy = df.copy()
    size_of_test = df.shape[0] / num_of_splits

    sets = []

    for i in range(num_of_splits - 1):
        frac_of_test = size_of_test / df_copy.shape[0]
        tr_set = build_train_set(df_copy, frac_of_test)
        sets.append(tr_set)
        df_copy = build_test_set(df_copy, tr_set)

    sets.append(df_copy)
    return sets

def convert_train_set_to_good_shape(train_df, test_df):
    test_df_unvisible = test_df.copy()
    test_df_unvisible['rating'] = np.NaN

    train_set_good_shape = pd.concat([train_df, test_df_unvisible], ignore_index=True, sort=False)
    Z_train_pivot, usermap_train, moviemap_train = reshape_ratings_dataframe(train_set_good_shape)

    return Z_train_pivot, usermap_train, moviemap_train


if __name__ == "__main__":
    # print(os.getcwd())

    # Example of usage
    ratings = pd.read_csv("../data/ratings.csv")
    #Z2 = reshape_ratings_dataframe(ratings)

    Z_train = build_train_set(ratings, 0.6)
    print(build_test_set(ratings, Z_train))

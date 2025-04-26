import pandas as pd
import numpy as np


def reshape_ratings_dataframe(ratings_df):
    """
    Function which reshape the ratings.csv dataframe with 100836 rows and 4 columns
    on the dataframe with 610 rows (userId) x 9724 columns (movieId).

    The i-th and j-th element of this dataframe is rating
    from user with id equals i about movie with id equals j.

    :param ratings_df: DataFrame object from ratings.csv file
    :return: DataFrame object with Nans for movies which are not rated by specific user
    """

    reshape_rating_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')
    return reshape_rating_df


def imputate_data_with_0(df):
    """
    Imputation missing data by replace them with 0.

    :param df: DataFrame object which was transformed by reshape_ratings_dataframe function
    :return: DataFrame object without missing data which were replaced by 0
    """
    return df.fillna(0)


# not rounds properly function; round to 1 number after coma.
def imputate_data_with_mean(df):
    """
    To think how impute by mean: impute missing values mean of rating each user
    or mean of rating each movie - now is by each movie

    :param df: DataFrame object which was transformed by reshape_ratings_dataframe function
    :return: DataFrame object without missing data which were replaced by mean of each movie
    """
    return (2 * df.where(pd.notna(df), df.mean(), axis="columns")).round() / 2


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# check
if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings.csv")
    Z2 = reshape_ratings_dataframe(ratings)

    Z2_0 = imputate_data_with_0(Z2)
    Z2_mean = imputate_data_with_mean(Z2)


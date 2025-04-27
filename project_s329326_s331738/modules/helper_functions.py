import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def reshape_ratings_dataframe(ratings_df):
    """
    Function which reshape the ratings.csv dataframe with 100836 rows and 4 columns
    on the dataframe with 610 rows (userId) x 9724 columns (movieId).

    The i-th and j-th element of this dataframe is rating
    from user with id equals i about movie with id equals j.

    :param ratings_df: DataFrame object from ratings.csv file
    :return: DataFrame object with Nans for movies which are not rated by specific user
    """
    # Extract unique users and movies
    unique_users = ratings_df["userId"].unique()
    unique_movies = ratings_df["movieId"].unique()

    # Create mappings: userId -> row index, movieId -> column index
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    reshape_rating_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')

    return reshape_rating_df, user_map, movie_map


def map_ids(df, user_map, movie_map):
    ids = []
    for row in df.itertuples():
        u = row.userId
        m = row.movieId

        i = user_map[u]
        j = movie_map[m]
        ids.append((i, j))

    return ids



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
    df_mean = (2 * df.stack().mean()).round() / 2
    return df.fillna(df_mean)


def imputate_data_with_mean_of_user(df):
    """
    Impute missing values by using the mean rating of each user (row).
    If a user has no ratings, use the global mean of all ratings.

    :param df: DataFrame object which was transformed by reshape_ratings_dataframe function
    :return: DataFrame object without missing data, with missing ratings replaced by the mean of each user
    """
    df_users_mean = (2 * df.mean(axis=1)).round() / 2
    global_mean = (2 * df.stack().mean()).round() / 2
    df_users_mean = df_users_mean.fillna(global_mean)

    return df.T.fillna(df_users_mean).T

def imputate_data_with_KNN(df):
    m = np.array(df)
    imputer = KNNImputer(n_neighbors=12, weights='distance', metric='nan_euclidean', keep_empty_features=True)
    M_imputed = imputer.fit_transform(m)
    return pd.DataFrame(np.round(M_imputed*2)/2)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# check
if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings.csv")
    Z2, usermap, moviemap = reshape_ratings_dataframe(ratings)

    Z2_0 = imputate_data_with_0(Z2)
    Z2_mean = imputate_data_with_mean(Z2)
    Z2_mean_user = imputate_data_with_mean_of_user(Z2)
    Z2_KNN = imputate_data_with_KNN(Z2)
    print(Z2_KNN)

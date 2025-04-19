import pandas as pd


def reshape_ratings_dataframe(ratings_df):
    """
        Function which reshape the ratings.csv dataframe with 100836 rows and 4 columns
        on the dataframe with 610 rows (userId) x 9724 columns (movieId).

        The i-th and j-th element of this dataframe is rating
        from user with id equals i about movie with id equals j.

        This function returns dataframe with Nans for movies which are not rated by specific user.
    """

    reshape_rating_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')
    return reshape_rating_df







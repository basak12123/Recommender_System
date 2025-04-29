import pandas as pd
import numpy as np


def predict_data(test_file, model_data, model_type):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.
    Missing userId/movieId combos produce a default rating (e.g., 0 or average).

    Returns a list of dicts with keys: 'model_type', 'userId', 'movieId', 'rating'.
    """
    df = pd.read_csv(test_file)

    Z_approx = model_data[0]
    user_map = model_data[1]
    movie_map = model_data[2]

    predictions = np.zeros((df.shape[0], 3))
    id_matrix = 0
    for row in df.itertuples():
        u = row.userId
        m = row.movieId
        if u in user_map and m in movie_map:
            i = user_map[u]
            j = movie_map[m]
            rating = Z_approx[i, j]
        else:
            # If user or movie not seen in training, default to 0 (or any strategy)
            rating = 0

        predictions[id_matrix, ] = [int(u), int(m), abs(rating)]
        id_matrix += 1

    convert_dict = {'model_type': str, 'userId': int, 'movieId': int, 'rating': float}
    predictions_df = pd.DataFrame(predictions,  columns=['userId', 'movieId', 'rating'])
    predictions_df.insert(0, 'model_type', model_type)
    predictions_df = predictions_df.astype(convert_dict)

    return predictions_df

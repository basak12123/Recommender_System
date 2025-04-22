import pandas as pd
import os
from project_s329326_s331738.modules.helper_functions import reshape_ratings_dataframe


# def build_train_set(df, n_rows):
    # df_sample = df.sample(n=n_rows, random_state=42)

    # # Create a DataFrame for sample_test.csv (only userId,movieId)
    # df_test = df_sample[["userId", "movieId"]].copy()
    #
    # # Create a DataFrame for sample_test_with_ratings.csv (userId,movieId,rating)
    # df_test_with_ratings = df_sample[["userId", "movieId", "rating"]].copy()
    #
    # # Save the output files
    # df_test.to_csv(args.output_test, index=False)
    # df_test_with_ratings.to_csv(args.output_test_with_ratings, index=False)

    # return df_sample


# print(os.getcwd())
# print()
#
# ratings = pd.read_csv("../project_s329326_s331738/data/ratings.csv")
# Z2 = reshape_ratings_dataframe(ratings)
#
# k = build_train_set(Z2, 10).index
# print(list(filter(lambda x: x not in k, Z2.index)))

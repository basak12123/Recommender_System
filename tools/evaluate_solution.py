import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import argparse


def compute_RMSE_model(test_file_with_ratings, pred_file_of_test_set):

    df_true = pd.read_csv(test_file_with_ratings)
    df_pred = pd.read_csv(pred_file_of_test_set)

    if "rating" not in df_pred.columns or "rating" not in df_true.columns:
        print("Error: Both CSV files must contain a 'rating' column.")
        return

    for model in pd.unique(df_pred["model_type"]):
        df_model = df_pred.loc[df_pred["model_type"] == model]
        rmse = np.sqrt(mean_squared_error(df_true['rating'], df_model['rating']))
        print(f"RMSE for {model}: {rmse:.4f}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute RMSE between predictions and true ratings.")
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Path to the CSV file with predictions (must contain a column 'rating').")
    parser.add_argument("--true_file", type=str, required=True,
                        help="Path to the CSV file with true ratings (must contain a column 'rating').")
    return parser.parse_args()


def main():
    args = parse_arguments()

    compute_RMSE_model(args.true_file, args.pred_file)


if __name__ == "__main__":
    main()

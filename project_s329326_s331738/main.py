import argparse
import os
import pickle
from modules.NMF import my_NMF
from modules.helper_functions import reshape_ratings_dataframe, imputate_data_with_0
from modules.train_functions import train_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple Recommender")
    parser.add_argument("--train", type=str, default="no",
                        help="Train mode: 'yes' to train NMF model, 'no' otherwise.")
    parser.add_argument("--predict", type=str, default="no",
                        help="Predict mode: 'yes' to predict ratings, 'no' otherwise.")
    parser.add_argument("--train_file", type=str, default="data/ratings.csv",
                        help="CSV file with training data (userId,movieId,rating).")
    parser.add_argument("--test_file", type=str, default="data/test_file.csv",
                        help="CSV file with (userId,movieId) for predictions.")
    parser.add_argument("--model_path", type=str, default="models_trained/nmf_model.pkl",
                        help="Path to save/load the trained NMF model.")
    parser.add_argument("--output_file", type=str, default="predictions/preds.csv",
                        help="Where to save predictions.")
    parser.add_argument("--alg", type=str, default="NMF",
                        help="Algorithm to use")
    return parser.parse_args()

def main():
    args = parse_arguments()
    train_mode = (args.train.lower() == "yes")
    predict_mode = (args.predict.lower() == "yes")
    model_type = args.alg.upper()

    if not model_type in ['NMF', "SVD1", "SVD2", "SGD", "ALL"]:
        print("Choose right model")
        return

    if train_mode:
        print(f"Training mode activated ({model_type}).")
        Z_approx = train_model(args.train_file, model_type)
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        with open(args.model_path, "wb") as f:
            pickle.dump(Z_approx, f)
        print(f"Model saved to {args.model_path}")

    if predict_mode:
        print(f"Predicting mode activated ({model_type}).")
        if not os.path.exists(args.model_path):
            print("Model file does not exist. Please run training first")
            return
        with open(args.model_path, "rb") as f:
            model_data = pickle.load(f)

if __name__ == "__main__":
    main()

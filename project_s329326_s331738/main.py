import argparse
import os
import pickle
import pandas as pd
from modules.train_functions import train_model
from modules.predict_functions import predict_data


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

    if train_mode:
        if not model_type in ['NMF', "SVD1", "SVD2", "SGD", "ALL"]:
            print("Choose right model")
            return

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

        if model_type == "ALL":
            types_of_models = ["NMF", "SVD1", "SVD2", "SGD"]
            results = []
            for name, model in model_data[0].items():
                results.append(predict_data(args.test_file, (model, model_data[1], model_data[2]), name))

            res = pd.concat(results)
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            res.to_csv(args.output_file, index=False)
            return

        predictions = predict_data(args.test_file, model_data, model_type)

        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        predictions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()

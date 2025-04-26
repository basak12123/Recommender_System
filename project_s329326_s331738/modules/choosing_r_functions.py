from SGD import mySGD_sklearnCompatible
from sklearn.model_selection import GridSearchCV
import pandas as pd
from tools.build_train_matrix import build_train_set, build_test_set
from helper_functions import reshape_ratings_dataframe

# DATA
ratings = pd.read_csv("../data/ratings.csv")
Z2 = reshape_ratings_dataframe(ratings)

id_train, Z2_train = build_train_set(Z2, 60000)
id_test, Z2_test = build_test_set(Z2, id_train)


SGDmodel = mySGD_sklearnCompatible()

SGD_param_grid = param_grid = {
    'r_components': [4, 6, 8],
    'lmb': [0.0, 0.001, 0.01]
}

from utils import *
from mlflow_utils import get_mlflow_experiment
from tqdm import tqdm
from joblib import Parallel, delayed
import lightgbm as lgb
import numpy as np

def accuracy(val_ratings_matrix, preds):
    mask = val_ratings_matrix != 0
    pred_ratings = np.clip(np.round(preds * 2) / 2, 0.5, 5.0)
    correct_preds = (pred_ratings == val_ratings_matrix) & mask
    return np.sum(correct_preds.astype(float)) / np.sum(mask.astype(float))

def rmse(val_ratings_matrix, preds):
    mask = val_ratings_matrix != 0
    predictions = np.clip(preds, 0.5, 5.0)
    E = (val_ratings_matrix - predictions) ** 2
    return np.sqrt(np.sum(E[mask]) / np.sum(mask.astype(float)))

# Function to process each user
def process_user(i, features, ratings, n_estimators, max_depth, num_leaves, learning_rate, lgb):
    ratings_user = ratings[i]
    X_train = features[ratings_user != 0]
    y_train = ratings_user[ratings_user != 0]
    if len(y_train) == 0:
        return np.zeros(ratings.shape[1])

    # Initialize LightGBM regressor with appropriate parameters
    
    
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)
    indices = (ratings_user == 0)
    preds_user = np.zeros(ratings.shape[1])
    if np.any(indices):
        X_test = features[indices]
        preds_user[indices] = model.predict(X_test)
    return preds_user


experiment = get_mlflow_experiment(experiment_name = "iasd_ds_lab_p1_features")
print("Name: {}".format(experiment.name))

if __name__ == "__main__":
        # Load data
        features = np.load("data_transformed/movie_embeddings.npy")
        ratings = np.load("data_raw/ratings_train.npy")
        ratings_test = np.load("data_raw/ratings_test.npy")
        ratings = np.nan_to_num(ratings, nan=0)
        ratings_test = np.nan_to_num(ratings_test, nan=0)

        num_leaves_values = [120]
        max_depth_values = [3]
        lr_values = [0.1]

        for lr in lr_values:
             for md in max_depth_values:
                for nl in num_leaves_values:

                    run_name = f"testing_lgb_features"
                    description = "Original train/val datasets."
                    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:
                        mlflow.set_tag("description", description)

                        # Initialize predictions array
                        preds = np.zeros(ratings.shape)

                        # Define hyper-parameters
                        params = {"n_estimators" : 100,
                                "learning_rate" : lr,
                                "max_depth" : md,
                                "num_leaves" : nl}
                        
                        mlflow.log_params(params)
                        
                        n_estimators = params["n_estimators"]
                        max_depth = params["max_depth"]
                        num_leaves = params["num_leaves"]
                        learning_rate = params['learning_rate']

                        # Parallel processing over users
                        preds_list = Parallel(n_jobs=-1)(
                            delayed(process_user)(i, features, ratings, n_estimators, max_depth, num_leaves, learning_rate, lgb) for i in range(ratings.shape[0])
                        )
                        preds = np.array(preds_list)

                        rms = rmse(ratings_test, preds)
                        acc = accuracy(ratings_test, preds)

                        mlflow.log_metric("accuracy", acc)
                        mlflow.log_metric("rmse", rms)

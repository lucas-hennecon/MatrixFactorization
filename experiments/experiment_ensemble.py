from utils import *
from mlflow_utils import get_mlflow_experiment
from tqdm import tqdm
from joblib import Parallel, delayed
import lightgbm as lgb
# torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = get_mlflow_experiment(experiment_name = "iasd_ds_lab_p1_ensemble")
print("Name: {}".format(experiment.name))

if __name__ == "__main__":

    device = torch.device("cpu")
    print(f"Using device: {device}")

    n_samples_per_run = 1
    for _ in tqdm(range(n_samples_per_run), desc="sample"):   
        run_name = f"best_weights_two_models"
        description = "Original train/val datasets."

        #content_based_weight_values = [0.35 + i*0.01 for i in range(11)]
        content_based_weight_values = [0.4]

        for weight in content_based_weight_values:
            with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:

                mlflow.set_tag("description", description)

                params = {"cb_weight" : weight}

                mlflow.log_params(params)

                ratings_matrix = torch.tensor(np.load("data_raw/ratings_train.npy")).to(device)
                
                # CF part

                n_steps = 57
                initial_learning_rate = 0.5*10**-3
                decay_rate = 0.5*10**-3
                latent_dimension = 101
                regul_user = 13
                regul_item = 1
                pref_threshold = 0.48
                init_type = 'randomA'
                coeff_randomA = 1000
                rounding_preference = "no"
                ratings_matrix = torch.nan_to_num(ratings_matrix, nan=0.0)
                cf_preds = model_inference(n_steps, initial_learning_rate, ratings_matrix, latent_dimension, regul_user, regul_item, decay_rate, device, init_type = init_type, coeff_randomA = coeff_randomA, rounding_preference = rounding_preference, pref_threshold = pref_threshold)

                # CB part
                features = np.load("data_transformed/movie_embeddings.npy")
                ratings = np.load("data_raw/ratings_train.npy")
                ratings_val = np.load("data_raw/ratings_test.npy")
                ratings = np.nan_to_num(ratings, nan=0)
                ratings_val = np.nan_to_num(ratings_val, nan=0)

                n_estimators = 100
                max_depth = 3
                learning_rate = 0.1
                num_leaves = 120

                preds_list = Parallel(n_jobs=-1)(
                            delayed(process_user)(i, features, ratings, n_estimators, max_depth, num_leaves, learning_rate, lgb) for i in range(ratings.shape[0])
                        )
                cb_preds = np.array(preds_list)

                preds = weight*cb_preds + (1-weight)*cf_preds


                pref_based = True
                if pref_based == True :
                    # "preference_based" rounding with threshold 0.48
                    pref_threshold = 0.48

                    half_ratings_mask = np.where((ratings % 1) != 0, 1, 0)
                    n_half_ratings_per_user = np.sum(half_ratings_mask, axis=1)
                    ratings_mask = np.where(ratings != 0, 1, 0)
                    n_ratings_per_user = np.sum(ratings_mask, axis=1)
                    semi_ratings_proportion = n_half_ratings_per_user / n_ratings_per_user

                    # Create a copy of best_preds to avoid modifying it in place
                    rounded_preds = preds.copy()

                    for i in range(ratings.shape[0]):
                        if semi_ratings_proportion[i] >= pref_threshold:
                            rounded_preds[i, :] = np.round(preds[i, :] * 2) / 2
                        else:
                            rounded_preds[i, :] = np.round(preds[i, :])


                mlflow.log_metric("accuracy", accuracy_npy(ratings_val, rounded_preds))
                mlflow.log_metric("rmse", rmse_npy(ratings_val, rounded_preds))

                # mlflow.log_metric("cf_accuracy", accuracy_npy(ratings_val, cf_preds))
                # mlflow.log_metric("cf_rmse", rmse_npy(ratings_val, cf_preds))   

                # mlflow.log_metric("cb_accuracy", accuracy_npy(ratings_val, cb_preds))
                # mlflow.log_metric("cb_rmse", rmse_npy(ratings_val, cb_preds))
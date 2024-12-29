from utils import *
from mlflow_utils import get_mlflow_experiment
from tqdm import tqdm
# torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = get_mlflow_experiment(experiment_name = "iasd_ds_lab_p1_preferences")
print("Name: {}".format(experiment.name))

if __name__ == "__main__":

    device = torch.device("cpu")
    print(f"Using device: {device}")


    k_values = [101] # Add other values if needed
    regul_user_values = [13] # Add other values if needed
    regul_item_values = [1] # Add other values if needed
    distrib_values = ['randomA'] # Add other values if needed

    pref_threshold_values = [0.49] # Add other values if needed

    for thresh in tqdm(pref_threshold_values):

        for dist in distrib_values:
                
            for k in k_values:
                
                for mu in regul_item_values:

                    for lbda in regul_user_values:
                        
                        n_samples_per_run = 5
                        for _ in tqdm(range(n_samples_per_run), desc="sample"):   
                            # run_name = "best_pref_threshold"
                            run_name = f"best_thresh_0.35_0.55"
                            description = "Original train/val datasets."
                            with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:

                                mlflow.set_tag("description", description)

                                ratings_matrix = torch.tensor(np.load("data_raw/ratings_train.npy")).to(device)
                                val_ratings_matrix = torch.tensor(np.load("data_raw/ratings_test.npy")).to(device)

                                params = {"n_steps" : 60,
                                        "initial_learning_rate" : 10**-3,
                                        "decay_rate" : 10**-3,
                                        "latent_dimension" : k,
                                        "regul_user" : lbda,
                                        "regul_item" : mu,
                                        "rounding_preference" : "preference_based",
                                        "init_type" : dist,
                                        "coeff_randomA": 1000,
                                        "coeff_randomC": 50,
                                        "pref_threshold" : thresh}
                                
                                mlflow.log_params(params)
                                
                                n_steps = params["n_steps"]
                                initial_learning_rate = params["initial_learning_rate"]
                                decay_rate = params["decay_rate"]
                                latent_dimension = params["latent_dimension"]
                                regul_user = params["regul_user"]
                                regul_item = params["regul_item"]
                                rounding_preference = params["rounding_preference"]
                                init_type = params["init_type"]
                                coeff_randomC = params["coeff_randomC"]
                                coeff_randomA = params["coeff_randomA"]
                                pref_threshold = params["pref_threshold"]


                                ratings_matrix = torch.nan_to_num(ratings_matrix, nan=0.0)
                                val_ratings_matrix = torch.nan_to_num(val_ratings_matrix, nan=0.0)
                                preds = model(n_steps, initial_learning_rate, ratings_matrix, latent_dimension, regul_user, regul_item, decay_rate,val_ratings_matrix, device, init_type, coeff_randomC, coeff_randomA, visuals = False, rounding_preference=rounding_preference, pref_threshold=pref_threshold)
                                rms = rmse(val_ratings_matrix, preds)
                                rms_train = rmse(ratings_matrix, preds)
                                acc = accuracy(val_ratings_matrix, preds)


                                mlflow.log_metric("accuracy", acc)
                                mlflow.log_metric("rmse", rms)
                                mlflow.log_metric("rmse_train", rms_train)

                                torch.cuda.empty_cache()
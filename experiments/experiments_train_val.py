from utils import *
from mlflow_utils import get_mlflow_experiment
# torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = get_mlflow_experiment(experiment_name = "iasd_ds_lab_p1")
print("Name: {}".format(experiment.name))

if __name__ == "__main__":

    with mlflow.start_run(run_name="testing", experiment_id=experiment.experiment_id) as run:

        device = torch.device("cpu")
        print(f"Using device: {device}")

        ratings_matrix = torch.tensor(np.load("data_raw/ratings_train.npy")).to(device)
        val_ratings_matrix = torch.tensor(np.load("data_raw/ratings_test.npy")).to(device)

        n_steps = 100
        initial_learning_rate = 10**-3
        decay_rate = 10**-3
        latent_dimension = 10
        regul_user = 4.2
        regul_item = 4.2

        params = {"n_steps" : 100,
                "initial_learning_rate" : 10**-3,
                "decay_rate" : 10**-3,
                "latent_dimension" : 10,
                "regul_user" : 4.2,
                "regul_item" : 4.2}
        
        n_steps = params["n_steps"]
        initial_learning_rate = params["initial_learning_rate"]
        decay_rate = params["decay_rate"]
        latent_dimension = params["latent_dimension"]
        regul_user = params["regul_user"]
        regul_item = params["regul_item"]

        mlflow.log_params(params)

        ratings_matrix = torch.nan_to_num(ratings_matrix, nan=0.0)
        val_ratings_matrix = torch.nan_to_num(val_ratings_matrix, nan=0.0)
        preds = gradient_descent(n_steps, initial_learning_rate, ratings_matrix, latent_dimension, regul_user, regul_item, decay_rate,val_ratings_matrix, device)
        rms = rmse(val_ratings_matrix, preds)
        acc = accuracy(val_ratings_matrix, preds)

        

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("rmse", rms)

        # torch.cuda.empty_cache()
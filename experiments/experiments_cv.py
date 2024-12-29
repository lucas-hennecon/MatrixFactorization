from utils import *
from mlflow_utils import get_mlflow_experiment
import matplotlib.pyplot as plt
# torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = get_mlflow_experiment(experiment_name = "iasd_ds_lab_p1")
print("Name: {}".format(experiment.name))

if __name__ == "__main__":

    run_name = "cross_val_01"
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:

        device = torch.device("cpu")
        print(f"Using device: {device}")

        params = {"n_steps" : 45,
                "initial_learning_rate" : 0.5*10**-3,
                "decay_rate" : 0.5*10**-3,
                "latent_dimension" : 10,
                "regul_user" : 0,
                "regul_item" : 0}
        
        n_steps = params["n_steps"]
        initial_learning_rate = params["initial_learning_rate"]
        decay_rate = params["decay_rate"]
        latent_dimension = params["latent_dimension"]
        regul_user = params["regul_user"]
        regul_item = params["regul_item"]

        mlflow.log_params(params)

        train_files = ["data_transformed/mat_train0.npy",
                    "data_transformed/mat_train1.npy",
                    "data_transformed/mat_train2.npy",
                    "data_transformed/mat_train3.npy",
                    "data_transformed/mat_train4.npy"]

        val_files = ["data_transformed/mat_val0.npy",
                    "data_transformed/mat_val1.npy",
                    "data_transformed/mat_val2.npy",
                    "data_transformed/mat_val3.npy",
                    "data_transformed/mat_val4.npy"]

        rm = torch.zeros(5)
        ac = torch.zeros(5)

        for i in range(5):

            ratings_matrix = torch.tensor(np.load(train_files[i])).to(device)
            val_ratings_matrix = torch.tensor(np.load(val_files[i])).to(device)

            ratings_matrix = torch.nan_to_num(ratings_matrix, nan=0.0)
            val_ratings_matrix = torch.nan_to_num(val_ratings_matrix, nan=0.0)
            preds = model(n_steps, initial_learning_rate, ratings_matrix, latent_dimension, regul_user, regul_item, decay_rate,val_ratings_matrix, device)
            rm[i] = rmse(val_ratings_matrix, preds)
            ac[i] = accuracy(val_ratings_matrix, preds)

        rms = rm.mean()
        acc = ac.mean()

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("rmse", rms)

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.bar(range(5), rm.detach().numpy())
        plt.title("RMSE on each fold")
        
        plt.subplot(1, 2, 2)
        plt.bar(range(5), ac.detach().numpy())
        plt.title("Accuracy on each fold")

        plt.subplots_adjust(wspace=0.4)

        mlflow.log_figure(fig, f"./cross_validation_performance_run{run_name}.png")


        # torch.cuda.empty_cache()
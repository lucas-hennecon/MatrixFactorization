from utils import *
from mlflow_utils import get_mlflow_experiment
import matplotlib.pyplot as plt
# torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = get_mlflow_experiment(experiment_name = "iasd_ds_lab_p1")
print("Name: {}".format(experiment.name))

if __name__ == "__main__":

    run_name = "best_lambda_k_fixed"
    description = "Varying lambda (same reg for U and V) for K fixed."
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:

        mlflow.set_tag("description", description)

        device = torch.device("cpu")
        print(f"Using device: {device}")

        lbda_values = [0.1 + i*0.1 for i in range(10)]
        n_values = len(lbda_values)
        k_acc = torch.zeros(n_values)
        k_rms = torch.zeros(n_values)

        for idx, lbda in enumerate(lbda_values):

            params = {"n_steps" : 45,
                    "initial_learning_rate" : 0.5*10**-3,
                    "decay_rate" : 0.5*10**-3,
                    "latent_dimension" : 10,
                    "regul_user" : lbda,
                    "regul_item" : lbda}
            
            n_steps = params["n_steps"]
            initial_learning_rate = params["initial_learning_rate"]
            decay_rate = params["decay_rate"]
            latent_dimension = params["latent_dimension"]
            regul_user = params["regul_user"]
            regul_item = params["regul_item"]

            # mlflow.log_params(params)

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

            k_rms[idx] = rm.mean()
            k_acc[idx] = ac.mean()


        fig = plt.figure()
        plt.bar(lbda_values, k_rms.detach().numpy(), label=[f"lambda = {lbda}" for lbda in lbda_values])
        plt.title("RMSE for different regularisators (lambda = mu), for k = 10")
        plt.legend(title="Lambda values")
        mlflow.log_figure(fig, f"./best_lbda_values_fixed_k_rmse{run_name}.png")
        
        fig = plt.figure()
        plt.bar(lbda_values, k_acc.detach().numpy(), label=[f"lambda = {lbda}" for lbda in lbda_values])
        plt.title("Accuracy for different regularisators (lambda = mu), for k = 10")
        plt.legend(title="Lambda values")
        mlflow.log_figure(fig, f"./best_lbda_values_fixed_k_accuracy{run_name}.png")
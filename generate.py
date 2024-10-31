import numpy as np
import torch
import os
from tqdm import tqdm, trange
import argparse
import lightgbm as lgb
from joblib import Parallel, delayed


def randomC(rating_matrix, latent_dimension, q, device):
    n_users, n_items = rating_matrix.shape
    user_embeddings = torch.zeros(n_users, latent_dimension, device=device)
    norms = torch.norm(rating_matrix, p=2, dim=0)
    sorted_indices = torch.argsort(norms, descending=True)

    for i in range(latent_dimension):
        top_indices = sorted_indices[:10*q]
        random_indices = top_indices[torch.randperm(top_indices.size(0))[:q]]
        selected_columns = rating_matrix[:, random_indices]
        mean_per_row = selected_columns.mean(dim=1)
        user_embeddings[:,i] = mean_per_row
    return user_embeddings

def scale(rating_matrix):
    rescaled = rating_matrix.clone()
    mask = rating_matrix != 0
    rescaled[mask] = (4 * rating_matrix[mask] - 11)
    return rescaled

def unscale(rating_matrix):
    rescaled = rating_matrix.clone()
    mask = rating_matrix != 0
    rescaled[mask] = (rating_matrix[mask] + 11) / 4
    return rescaled

def randomA(rating_matrix, latent_dimension, q, device):
    n_users, n_items = rating_matrix.shape
    user_embeddings = torch.zeros(n_users, latent_dimension, device=device)

    for i in range(latent_dimension):
        random_indices = torch.randperm(n_items)[:q]
        selected_columns = rating_matrix[:, random_indices]
        mean_per_row = selected_columns.mean(dim=1)
        user_embeddings[:,i] = mean_per_row
    
    return user_embeddings

def model_inference(n_steps, initial_learning_rate, ratings_matrix, latent_dimension, regul_user, regul_item, decay_rate, device, init_type = "normal + latent_dimension", coeff_randomC ="100", coeff_randomA = "100", rounding_preference = "no", pref_threshold = 0.49):

    ratings_matrix = scale(ratings_matrix)
    n_users, n_items = ratings_matrix.shape

    #Defines the distribution of the initialization matrices:
    if init_type == 'normal/(n + latent_dimension)':
        user_embeddings = (
            torch.randn(n_users, latent_dimension, device=device) *
            torch.sqrt(torch.tensor(2 / (n_users + latent_dimension), device=device))
        ).requires_grad_(True)

        item_embeddings = (
            torch.randn(n_items, latent_dimension, device=device) *
            torch.sqrt(torch.tensor(2 / (n_items + latent_dimension), device=device))
        ).requires_grad_(True)

    elif init_type == 'uniform':
        user_embeddings = (
            torch.rand(n_users, latent_dimension, device=device)/100
        ).requires_grad_(True)

        item_embeddings = (
            torch.rand(n_items, latent_dimension, device=device)/100
        ).requires_grad_(True)

    elif init_type == 'normal':
        user_embeddings = (
            torch.randn(n_users, latent_dimension, device = device)
        ).requires_grad_(True)

        item_embeddings = (
            torch.randn(n_items, latent_dimension, device = device)
        ).requires_grad_(True)
    
    elif init_type == 'ones/100':
        user_embeddings = (
            torch.ones(n_users, latent_dimension, device = device)/100
        ).requires_grad_(True)

        item_embeddings = (
            torch.ones(n_items, latent_dimension, device = device)/100
        ).requires_grad_(True)

    elif init_type == 'randomA':
        user_embeddings = randomA(ratings_matrix, latent_dimension, coeff_randomA, device).requires_grad_(True)
        item_embeddings = (
            torch.ones(n_items, latent_dimension, device = device)*2 / (n_items + latent_dimension)
        ).requires_grad_(True)

    elif init_type == 'randomC':
        user_embeddings = randomC(ratings_matrix, latent_dimension, coeff_randomC, device).requires_grad_(True)
        item_embeddings = (
            torch.ones(n_items, latent_dimension, device = device)*2 / (n_items + latent_dimension)
        ).requires_grad_(True)


    for t in range(n_steps):
        learning_rate = initial_learning_rate / (1 + decay_rate * t)

        mask = ratings_matrix != 0
        preds = user_embeddings @ item_embeddings.T
        squared_diff = (ratings_matrix - preds) ** 2
        cost = torch.sum(squared_diff[mask]) + regul_user * torch.norm(user_embeddings)**2 + regul_item * torch.norm(item_embeddings)**2

        cost.backward()
        grad_user, grad_item = user_embeddings.grad, item_embeddings.grad

        with torch.no_grad():
            user_embeddings -= learning_rate * grad_user
            item_embeddings -= learning_rate * grad_item

        user_embeddings.grad.zero_()
        item_embeddings.grad.zero_()

    preds = user_embeddings @ item_embeddings.T
    preds = torch.clamp(preds, -9, 9)
    preds = unscale(preds)

    ratings_matrix = unscale(ratings_matrix)

    if rounding_preference == "no":
        preds = preds.detach().cpu().numpy()
        return preds

    elif rounding_preference == "half_ratings":
        rounded_preds = torch.round(preds * 2) / 2
        rounded_preds = rounded_preds.detach().cpu().numpy()
        return rounded_preds

    elif rounding_preference == "preference_based":
        half_ratings_mask = torch.where((ratings_matrix % 1) != 0, 1, 0)
        n_half_ratings_per_user = torch.sum(half_ratings_mask, dim=1)
        ratings_mask = torch.where(ratings_matrix != 0, 1, 0)
        n_ratings_per_user = torch.sum(ratings_mask, dim=1)
        semi_ratings_proportion = n_half_ratings_per_user / n_ratings_per_user

        # Create a copy of preds to avoid modifying it in place
        rounded_preds = preds.clone()

        for i in range(ratings_matrix.shape[0]):
            if semi_ratings_proportion[i] >= (pref_threshold):
                rounded_preds[i, :] = torch.round(preds[i, :] * 2) / 2
            else:
                rounded_preds[i, :] = torch.round(preds[i, :])

        rounded_preds = rounded_preds.detach().cpu().numpy()
        return rounded_preds
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Any method you want
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ratings_matrix = torch.tensor(table).to(device)

    weight = 0.4

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
    features = np.load("movie_embeddings.npy")
    ratings = table.copy()
    ratings = np.nan_to_num(ratings, nan=0)

    n_estimators = 100
    max_depth = 3
    learning_rate = 0.1
    num_leaves = 120

    preds_list = Parallel(n_jobs=-1)(
                delayed(process_user)(i, features, ratings, n_estimators, max_depth, num_leaves, learning_rate, lgb) for i in range(ratings.shape[0])
            )
    cb_preds = np.array(preds_list)

    # Combine the two methods

    table = weight*cb_preds + (1-weight)*cf_preds


    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        

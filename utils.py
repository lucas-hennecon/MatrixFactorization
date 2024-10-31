import torch
import numpy as np
import mlflow
import matplotlib.pyplot as plt

def accuracy(val_ratings_matrix, preds):
    mask = val_ratings_matrix != 0
    pred_ratings = torch.clamp(torch.round(preds * 2) / 2, 0.5, 5.0)
    correct_preds = (pred_ratings == val_ratings_matrix) & mask
    return torch.sum(correct_preds.float()) / torch.sum(mask.float())

def rmse(val_ratings_matrix, preds):
    mask = val_ratings_matrix != 0
    predictions = torch.clamp(preds, 0.5, 5.0)
    E = (val_ratings_matrix - predictions) ** 2
    return torch.sqrt(torch.sum(E[mask]) / torch.sum(mask.float()))

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


def model(n_steps, initial_learning_rate, ratings_matrix, latent_dimension, regul_user, regul_item, decay_rate,val_ratings_matrix, device, init_type = "normal + latent_dimension", coeff_randomC ="100", coeff_randomA = "100", visuals = False, rounding_preference = "no", pref_threshold = 0.25):

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



    best_user_embeddings = user_embeddings.clone()
    best_item_embeddings = item_embeddings.clone()

    best_val_error = torch.tensor(float('inf'), device=device)
    errors = torch.zeros(n_steps, device=device)
    val_errors = torch.zeros(n_steps, device=device)

    for t in range(n_steps):
        learning_rate = initial_learning_rate / (1 + decay_rate * t)

        mask = ratings_matrix != 0
        preds = user_embeddings @ item_embeddings.T
        squared_diff = (ratings_matrix - preds) ** 2
        cost = torch.sum(squared_diff[mask]) + regul_user * torch.norm(user_embeddings)**2 + regul_item * torch.norm(item_embeddings)**2

        errors[t] = cost.item()

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

        current_error = rmse(val_ratings_matrix, preds)
        val_errors[t] = current_error.clone()

        if current_error < best_val_error:
            best_val_error = current_error
            best_accuracy = accuracy(val_ratings_matrix, preds)
            best_user_embeddings = user_embeddings.clone()
            best_item_embeddings = item_embeddings.clone()

    best_preds = best_user_embeddings @ best_item_embeddings.T
    best_preds = torch.clamp(best_preds, -9, 9)
    best_preds = unscale(best_preds)
    ratings_matrix = unscale(ratings_matrix)
    

    if visuals == True :

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

        # Left plot for training error
        axs[0].plot(range(n_steps), errors.detach().cpu().numpy(), label="training error")
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Training Error")
        axs[0].set_title("Evolution of Training Error")

        # Right plot for validation error
        axs[1].plot(range(n_steps), val_errors.detach().cpu().numpy(), label="validation loss")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("Validation Error")
        axs[1].set_title("Evolution of Validation Error")

        plt.tight_layout()

        mlflow.log_figure(fig, "training_validation_error.png")

    if rounding_preference == "no":
        return best_preds

    elif rounding_preference == "half_ratings":
        return torch.round(best_preds * 2) / 2

    elif rounding_preference == "preference_based":
        half_ratings_mask = torch.where((ratings_matrix % 1) != 0, 1, 0)
        n_half_ratings_per_user = torch.sum(half_ratings_mask, dim=1)
        ratings_mask = torch.where(ratings_matrix != 0, 1, 0)
        n_ratings_per_user = torch.sum(ratings_mask, dim=1)
        semi_ratings_proportion = n_half_ratings_per_user / n_ratings_per_user

        # Create a copy of best_preds to avoid modifying it in place
        rounded_preds = best_preds.clone()

        for i in range(ratings_matrix.shape[0]):
            if semi_ratings_proportion[i] >= (pref_threshold):
                rounded_preds[i, :] = torch.round(best_preds[i, :] * 2) / 2
            else:
                rounded_preds[i, :] = torch.round(best_preds[i, :])

        return rounded_preds
    
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
    
def accuracy_npy(val_ratings_matrix, preds):
    mask = val_ratings_matrix != 0
    pred_ratings = np.clip(np.round(preds * 2) / 2, 0.5, 5.0)
    correct_preds = (pred_ratings == val_ratings_matrix) & mask
    return np.sum(correct_preds.astype(float)) / np.sum(mask.astype(float))

def rmse_npy(val_ratings_matrix, preds):
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
# Collaborative Filtering - Matrix Factorization

This repository contains the work done for the **Data Science Lab Project 1 (IASD)**, in collaboration with **Alexandre Olech** and **Matthieu Neau**, focusing on **Collaborative Filtering** techniques for recommendation systems. For a detailed presentation of the project and results, refer to the `report.pdf` file.

## 📜 Summary
This project focuses on **matrix factorization** as a key technique for **collaborative filtering**, complemented by **content-based approaches** to build a robust recommendation system.
- A custom implementation of **Unconstrained Matrix Factorization** in **PyTorch**.
- Experiments with initialization strategies, latent dimensions, and regularization coefficients.
- Integration of a **content-based model** using one-hot-encoded movie features and **gradient boosting**.
- Development of a **hybrid model** combining collaborative filtering and content-based methods for enhanced performance.

## 📂 Repository Files
- **`utils.py`**: Implementation of **Matrix Factorization**, various initialization techniques, and evaluation metrics.
- **`generate.py`**: Contains code for inference.
- **`report.pdf`**: Detailed documentation of the project, experiments, and results.
- **`/experiments` Folder** includes:
  - **`experiment_ensemble.py`**: Implementation and evaluation of the hybrid ensemble model.
  - **`experiments_grid_search_train_val.py`**: Scripts for grid search on train-validation splits.
  - **`experiments_lgb.py`**: Experiments using **LightGBM** for content-based modeling with movie features.
  - Additional scripts for cross-validation and content-based approaches.
- **`mlflow_utils.py`**: Utilities for logging experiments and results using MLflow.
- **`/data_raw` Folder**: Contains raw data.



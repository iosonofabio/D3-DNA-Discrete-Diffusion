""" 

Use Tree Structured Parzen Estimator to optimize the hyperparameters of the model.

"""

# import optuna

# def objective(trial):
#     x = trial.suggest_float("x", -10, 10)
#     y = trial.suggest_float("y", -10, 10)
#     return x**2 + y**2

# study = optuna.create_study(direction="minimize")
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np

X_train, X_test, y_train, y_test = joblib.load('data/split_data.pkl')

models = {
    'linear': LinearRegression(),
    'ridge': Ridge(),
    'gradient_boosting': GradientBoostingRegressor()
}

def train_and_save_models(models, X_train, y_train, label):
    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train[label])
        joblib.dump(model, f'models/{label}_{model_name}_model.pkl')
        trained_models[model_name] = model
    return trained_models

acousticness_models = train_and_save_models(models, X_train, y_train, 'acousticness')
danceability_models = train_and_save_models(models, X_train, y_train, 'danceability')
energy_models = train_and_save_models(models, X_train, y_train, 'energy')

def evaluate_models(models, X_test, y_test, label):
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(root_mean_squared_error(y_test[label], y_pred))
        print(f"{label.capitalize()} {model_name.capitalize()} RMSE: {rmse:.4f}")

evaluate_models(acousticness_models, X_test, y_test, 'acousticness')
evaluate_models(danceability_models, X_test, y_test, 'danceability')
evaluate_models(energy_models, X_test, y_test, 'energy')

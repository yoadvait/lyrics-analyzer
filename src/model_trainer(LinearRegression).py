import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np

X_train, X_test, y_train, y_test = joblib.load('data/split_data.pkl')

acousticness_model = LinearRegression()
danceability_model = LinearRegression()
energy_model = LinearRegression()

def train_and_save_model(model, X_train, y_train, label, model_name):
    model.fit(X_train, y_train[label])
    joblib.dump(model, f'models/{model_name}.pkl')
    return model

acousticness_model = train_and_save_model(acousticness_model, X_train, y_train, 'acousticness', 'acousticness_model')
danceability_model = train_and_save_model(danceability_model, X_train, y_train, 'danceability', 'danceability_model')
energy_model = train_and_save_model(energy_model, X_train, y_train, 'energy', 'energy_model')

def evaluate_model(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(root_mean_squared_error(y_test[label], y_pred))
    print(f"{label.capitalize()} RMSE: {rmse:.4f}")

evaluate_model(acousticness_model, X_test, y_test, 'acousticness')
evaluate_model(danceability_model, X_test, y_test, 'danceability')
evaluate_model(energy_model, X_test, y_test, 'energy')
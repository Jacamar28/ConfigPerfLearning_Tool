import pandas as pd
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np

def load_data(datasets_location, csv_file):
    # Load data from CSV file
    return pd.read_csv(os.path.join(datasets_location, csv_file))

def preprocess_data(data, add_poly=False):
    # Separate features (X) and target (Y)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Polynomial feature expansion
    if add_poly:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_scaled = poly.fit_transform(X_scaled)

    return X_scaled, y

def tune_hyperparameters(X, y, random_state):
    # Find best parameters for GBR using grid search
    
    param_grid = { # Init dictionary of hyperparameters to test
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    model = GradientBoostingRegressor(random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    print("Best parameters found:", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_model(model, X, y, cv_folds=5): # Number of folds set to 5
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    mape_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_percentage_error', cv=kf)
    mae_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf))

    # Returns the mean and standard deviation for each error metric
    print("MAPE: {:.2f} ± {:.2f}".format(np.mean(mape_scores), np.std(mape_scores)))
    print("MAE: {:.2f} ± {:.2f}".format(np.mean(mae_scores), np.std(mae_scores)))
    print("RMSE: {:.2f} ± {:.2f}".format(np.mean(rmse_scores), np.std(rmse_scores)))
    
    return mape_scores, mae_scores, rmse_scores
    
def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    base_random_seed = 1

    for current_system in systems:
        datasets_location = f'datasets/{current_system}' # Modify this to specify the location of the datasets
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}')
            data = load_data(datasets_location, csv_file)
            X, y = preprocess_data(data, add_poly=True) 
            
            # Tune the hyperparameters
            best_model = tune_hyperparameters(X, y, random_state=base_random_seed)
            
            # Evaluate the model using cross-validation
            evaluate_model(best_model, X, y)
            
if __name__ == "__main__":
    main()
    

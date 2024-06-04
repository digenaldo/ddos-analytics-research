# hyperparameter_tuning.py in abd

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from data_processing import load_data
from model_training import preprocess_data

def perform_hyperparameter_tuning(X, y, model, param_grid, cv=5):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_score_

def main_hyperparameter_tuning():
    # Load and preprocess data
    train_data, test_data = load_data()
    X_train, y_train = preprocess_data(train_data)
    
    # Define models and parameter grids
    param_grids = {
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        }
    }

    # Perform hyperparameter tuning
    results = {}
    for model_name, config in param_grids.items():
        best_params, best_score = perform_hyperparameter_tuning(X_train, y_train, config['model'], config['params'])
        results[model_name] = (best_params, best_score)
        print(f"{model_name}: Best Params = {best_params}, Best Score = {best_score:.2f}")

    return results
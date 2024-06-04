# cross_validation.py in abd

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from data_processing import load_data
from model_training import preprocess_data

def perform_cross_validation(X, y, model, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

def main_cross_validation():
    # Load and preprocess data
    train_data, test_data = load_data()
    X_train, y_train = preprocess_data(train_data)
    
    # Define models
    models = {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier()
    }

    # Perform cross-validation
    results = {}
    for model_name, model in models.items():
        mean_accuracy, std_accuracy = perform_cross_validation(X_train, y_train, model)
        results[model_name] = (mean_accuracy, std_accuracy)
        print(f"{model_name}: Mean Accuracy = {mean_accuracy:.2f}, Std = {std_accuracy:.2f}")
    
    return results
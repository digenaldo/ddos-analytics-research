from sklearn.metrics import accuracy_score

def evaluate_models(models, X_test, y_test):
    accuracies = []
    for model in models:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies

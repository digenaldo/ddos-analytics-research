import time
from Preprocessing import preprocess_data
from Models import train_models
from Evaluation import evaluate_models
from Analysis import analyze_results

# Load datasets
train_file = 'train_mosaic.csv.zip'
test_file = 'test_mosaic.csv.zip'

# Preprocessing
X_train, X_test, y_train, y_test = preprocess_data(train_file, test_file)

# Train models
models = train_models(X_train, y_train)

# Evaluate models
accuracies = evaluate_models(models, X_test, y_test)

# Analyze results
results = [{"algoritmo": "Naive Bayes", "acuracia": accuracies[0]},
           {"algoritmo": "Decision Tree", "acuracia": accuracies[1]},
           {"algoritmo": "Logistic Regression", "acuracia": accuracies[2]},
           {"algoritmo": "Random Forest", "acuracia": accuracies[3]}]

analyze_results(results)

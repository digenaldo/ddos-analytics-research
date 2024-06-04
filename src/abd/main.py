# main.py
from spark_setup import initialize_spark_session
from data_preprocessing import preprocess_data
from model_training import train_models
from evaluation import evaluate_models
from cross_validation import main_cross_validation
from hyperparameter_tuning import main_hyperparameter_tuning

def main():
    # Initialize the Spark session
    spark = initialize_spark_session()
    
    # Data preprocessing
    train_data, test_data = preprocess_data(spark)
    
    # Train models
    trained_models = train_models(train_data)
    
    # Run Cross Validation
    print("Running Cross Validation...")
    cv_results = main_cross_validation()
    print(cv_results)
    
    # Run Hyperparameter Tuning
    print("Running Hyperparameter Tuning...")
    ht_results = main_hyperparameter_tuning()
    print(ht_results)
    
    # Evaluate models
    evaluate_models(trained_models, test_data)

if __name__ == "__main__":
    main()
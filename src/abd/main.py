# main.py
from spark_setup import initialize_spark_session
from data_preprocessing import preprocess_data
from model_training import train_models
from evaluation import evaluate_models

def main():
    # Inicializar a sessão do Spark
    spark = initialize_spark_session()
    
    # Pré-processamento dos dados
    train_data, test_data = preprocess_data(spark)
    
    # Treinar modelos
    trained_models = train_models(train_data)
    
    # Avaliar modelos
    evaluate_models(trained_models, test_data)

if __name__ == "__main__":
    main()

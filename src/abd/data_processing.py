# data_preprocessing.py
from pyspark.sql.functions import when, col

def preprocess_data(spark):
    """Realiza o pré-processamento dos dados."""
    train_data = spark.read.csv("train_mosaic.csv", header=True, inferSchema=True)
    test_data = spark.read.csv("test_mosaic.csv", header=True, inferSchema=True)

    # Substitui valores negativos
    train_data = replace_values_less_than_zero(train_data)
    test_data = replace_values_less_than_zero(test_data)

    # Filtra valores inválidos
    train_data_filtered = filter_invalid_values(train_data)
    test_data_filtered = filter_invalid_values(test_data)

    return train_data_filtered, test_data_filtered

def replace_values_less_than_zero(df, replacement_value=0.0):
    """Substitui todos os valores menores que zero por um valor de substituição em colunas específicas de um DataFrame."""
    # Colunas a serem substituídas
    feature_cols_replaces = [...]  # lista de colunas específicas
    for col_name in feature_cols_replaces:
        df = df.withColumn(
            col_name,
            when(col(col_name) < 0, replacement_value).otherwise(col(col_name))
        )
    return df

def filter_invalid_values(df):
    """Filtra linhas com valores inválidos (negativos, NaN, infinito, string vazia) em colunas específicas."""
    # Colunas a serem filtradas
    feature_cols = [...]  # lista de colunas específicas
    invalid_conditions = [
        col(col_name).isin(float("inf"), float("-inf"), "") | col(col_name).isNull() | (col(col_name) < 0)
        for col_name in feature_cols
    ]
    combined_condition = invalid_conditions[0]
    for condition in invalid_conditions[1:]:
        combined_condition = combined_condition | condition
    return df.filter(~combined_condition)

# spark_setup.py
from pyspark.sql import SparkSession

def initialize_spark_session():
    """Inicializa e retorna a sessão do Spark."""
    return SparkSession.builder \
        .appName("DDoS Attack Data Analysis") \
        .getOrCreate()

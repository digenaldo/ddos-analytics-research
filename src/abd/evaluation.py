# evaluation.py
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def evaluate_models(trained_models, test_data):
    """Avalia os modelos treinados."""
    for clf_name, model in trained_models.items():
        # Fazer previsões
        predictions = model.transform(test_data)

        # Avaliar o modelo
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)

        # Imprimir resultados
        print(f"Classifier: {clf_name}")
        print(f"Accuracy: {accuracy}")
        print("\n")

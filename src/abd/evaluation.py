# evaluation.py
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def evaluate_models(trained_models, test_data):
    """Evaluates the trained models."""
    for clf_name, model in trained_models.items():
        # Make predictions
        predictions = model.transform(test_data)

        # Evaluate the model
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)

        # Print results
        print(f"Classifier: {clf_name}")
        print(f"Accuracy: {accuracy}")
        print("\n")
# model_training.py
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier

def train_models(train_data):
    """Treina modelos de classificação."""
    # Identificar as colunas de recurso
    feature_cols = [...]  # lista de colunas de recurso específicas

    # Criar um VectorAssembler para combinar recursos
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Combinar recursos para dados de treinamento
    train_features = vector_assembler.transform(train_data)

    # Selecionar apenas as colunas relevantes
    selected_cols = feature_cols + ["label", "features"]
    train_features_selected = train_features.select(selected_cols)

    # Criar os modelos
    classifiers = {
        "Naive Bayes": NaiveBayes(featuresCol="features", labelCol="label", smoothing=1e-9),
        "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5, maxBins=32),
        "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=10),
        "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10, maxDepth=5)
    }

    # Treinar os modelos
    trained_models = {}
    for clf_name, clf_model in classifiers.items():
        model = clf_model.fit(train_features_selected)
        trained_models[clf_name] = model

    return trained_models

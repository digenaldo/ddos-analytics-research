# -*- coding: utf-8 -*-
"""DDoS-Attack-Analytics-Digenaldo-ML.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11IQO2O9TTIL4jBe53JrvzLdOjVZoQPuZ
"""

from mpl_toolkits.mplot3d import Axes3D #For Basic ploting
from sklearn.preprocessing import StandardScaler #Preprocessing
from sklearn import preprocessing    # Preprocessing
from sklearn.naive_bayes import GaussianNB #import gaussian naive bayes model
from sklearn.tree import DecisionTreeClassifier #import Decision tree classifier
from sklearn import metrics  #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import time

nRowsRead = 1000 # specify No. of row. 'None' for whole data
# test_mosaic.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('test_mosaic.csv.zip', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'test_mosaic.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

df1.head(6)

nRowsRead = 1000 # specify No. of rows. 'None' for whole file
# train_mosaic.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('train_mosaic.csv.zip', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'train_mosaic.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')

df2.head(5)

nRowsRead = None # specify No. of row. 'None' for whole data
train_data = pd.read_csv('train_mosaic.csv.zip', delimiter=',', nrows = nRowsRead)
train_data.dataframeName = 'train_mosaic.csv'
nRow, nCol = train_data.shape
print(f'There are {nRow} rows and {nCol} columns')

train_data.head()

nRowsRead = None # specify No. of row. 'None' for whole data
test_data = pd.read_csv('test_mosaic.csv.zip', delimiter=',', nrows = nRowsRead)
test_data.dataframeName = 'test_mosaic.csv'
nRow, nCol = test_data.shape
print(f'There are {nRow} rows and {nCol} columns')

test_data.head()

train_data['Label'].unique()
test_data['Label'].unique()

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
train_data['Label'] = label_encoder.fit_transform(train_data['Label'])
test_data['Label'] = label_encoder.fit_transform(test_data['Label'])

train_data.head()

test_data.head()

X_train = train_data.drop('Label',axis=1)
X_test = test_data.drop('Label',axis=1)
y_train = train_data['Label']
y_test = test_data['Label']

X_train.head()

y_train.head()

X_test.head()

y_test.head()

# create gaussian naive bayes classifier
gnb = GaussianNB(var_smoothing=1e-9)
#Train the model using the training sets
gnb.fit(X_train,y_train)
#Predict the response for test dataset
gnb_pred = gnb.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy : ",metrics.accuracy_score(y_test,gnb_pred))

# Classification Report for Gaussian Naive Bayes
print("\nClassification Report for Naive Bayes:")
print(classification_report(y_test, gnb_pred))

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
dt_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, dt_pred))

# Classification Report
print("\nClassification Report for Decision Tree:")
print(classification_report(y_test, dt_pred))

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
dt_pred1 = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, dt_pred1))

# Classification Report
print("\nClassification Report for Decision Tree:")
print(classification_report(y_test, dt_pred1))

import warnings
warnings.filterwarnings("ignore")

# Logistic Regression classifier with 'lbfgs' solver and 'auto' multi_class
#log_reg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_accuracy = metrics.accuracy_score(y_test, log_reg_pred)
print("Logistic Regression Accuracy:", log_reg_accuracy)

# Classification Report for Logistic Regression
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, log_reg_pred))

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
rf_accuracy = metrics.accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

# Classification Report for Random Forest
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, rf_pred))

import warnings
warnings.filterwarnings("ignore")

# Inicializar listas para armazenar os tempos de execução
train_times = []

# Treinar e testar Naive Bayes
start_time = time.time()
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
train_times.append(time.time() - start_time)
print("Tempo de execução do Naive Bayes:", train_times[-1], "segundos")

# Treinar e testar Decision Tree
start_time = time.time()
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
dt_pred = clf.predict(X_test)
train_times.append(time.time() - start_time)
print("Tempo de execução da Decision Tree:", train_times[-1], "segundos")

# Treinar e testar Logistic Regression
start_time = time.time()
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
train_times.append(time.time() - start_time)
print("Tempo de execução da Logistic Regression:", train_times[-1], "segundos")

# Treinar e testar Random Forest
start_time = time.time()
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
train_times.append(time.time() - start_time)
print("Tempo de execução do Random Forest:", train_times[-1], "segundos")

# Armazenar os nomes dos classificadores
classifiers = ['Naive Bayes', 'Decision Tree', 'Logistic Regression', 'Random Forest']

import warnings
warnings.filterwarnings("ignore")

# Inicializar listas para armazenar as acurácias
accuracies = []

# Gaussian Naive Bayes
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
gnb_accuracy = accuracy_score(y_test, gnb_pred)
accuracies.append(gnb_accuracy)
print("Acurácia do Naive Bayes:", gnb_accuracy)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
accuracies.append(dt_accuracy)
print("Acurácia da Decision Tree:", dt_accuracy)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
accuracies.append(log_reg_accuracy)
print("Acurácia da Logistic Regression:", log_reg_accuracy)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
accuracies.append(rf_accuracy)
print("Acurácia do Random Forest:", rf_accuracy)

# Armazenar os nomes dos classificadores
classifiers = ['Naive Bayes', 'Decision Tree', 'Logistic Regression', 'Random Forest']

results = []

# Função para adicionar os resultados à lista
def add_results(algorithm, environment, times, accuracies, iteration_times, iteration_accuracies, run):
    results.append({
        "rodada": run,
        "algoritmo": algorithm,
        "ambiente": environment,
        "tempo": times[-1],
        "acuracia": accuracies[-1],
        "iteracoes_tempo": iteration_times,
        "iteracoes_acuracia": iteration_accuracies
})

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# Dividir os dados em conjunto de treinamento e teste
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar listas para armazenar os tempos de execução e as acurácias
train_times = {'Naive Bayes': [], 'Decision Tree': [], 'Logistic Regression': [], 'Random Forest': []}
accuracies = {'Naive Bayes': [], 'Decision Tree': [], 'Logistic Regression': [], 'Random Forest': []}
classifiers = ['Naive Bayes', 'Decision Tree', 'Logistic Regression', 'Random Forest']

# Loop para treinar e testar os modelos dez vezes
for i in range(30):
    print(f"----------------------------- Iteração {i+1} -----------------------------")

    # Naive Bayes
    start_time = time.time()
    gnb = GaussianNB(var_smoothing=1e-9)
    gnb.fit(X_train, y_train)
    gnb_pred = gnb.predict(X_test)
    nb_train_time = time.time() - start_time
    nb_accuracy = accuracy_score(y_test, gnb_pred)
    print("Tempo de execução do Naive Bayes:", nb_train_time, "segundos")
    print("Acurácia do Naive Bayes:", nb_accuracy)

    nb_times = []
    nb_accuracies = []
    for j in range(30):
        start_time = time.time()
        gnb.fit(X_train, y_train)
        gnb_pred = gnb.predict(X_test)
        nb_times.append(time.time() - start_time)
        nb_accuracies.append(accuracy_score(y_test, gnb_pred))
        print(f"Execução para 30x {j+1}:")
        print("    Tempo de execução do Naive Bayes:", nb_times[-1], "segundos")
        print("    Acurácia do Naive Bayes:", nb_accuracies[-1])

    add_results("nb", "ml", [nb_train_time], [nb_accuracy], nb_times, nb_accuracies, i+1)

    # Decision Tree
    start_time = time.time()
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_train_time = time.time() - start_time
    dt_accuracy = accuracy_score(y_test, dt_pred)
    print("Tempo de execução do Decision Tree:", dt_train_time, "segundos")
    print("Acurácia do Decision Tree:", dt_accuracy)

    dt_times = []
    dt_accuracies = []
    for k in range(30):
        start_time = time.time()
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        dt_times.append(time.time() - start_time)
        dt_accuracies.append(accuracy_score(y_test, dt_pred))
        print(f"Execução para 30x {k+1}:")
        print("    Tempo de execução do Decision Tree:", dt_times[-1], "segundos")
        print("    Acurácia do Decision Tree:", dt_accuracies[-1])

    add_results("dt", "ml", [dt_train_time], [dt_accuracy], dt_times, dt_accuracies, i+1)

    # Logistic Regression
    start_time = time.time()
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg_pred = log_reg.predict(X_test)
    log_reg_train_time = time.time() - start_time
    log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
    print("Tempo de execução do Logistic Regression:", log_reg_train_time, "segundos")
    print("Acurácia do Logistic Regression:", log_reg_accuracy)

    log_reg_times = []
    log_reg_accuracies = []
    for l in range(30):
        start_time = time.time()
        log_reg.fit(X_train, y_train)
        log_reg_pred = log_reg.predict(X_test)
        log_reg_times.append(time.time() - start_time)
        log_reg_accuracies.append(accuracy_score(y_test, log_reg_pred))
        print(f"Execução para 30x {l+1}:")
        print("    Tempo de execução do Logistic Regression:", log_reg_times[-1], "segundos")
        print("    Acurácia do Logistic Regression:", log_reg_accuracies[-1])

    add_results("lr", "ml", [log_reg_train_time], [log_reg_accuracy], log_reg_times, log_reg_accuracies, i+1)

    # Random Forest
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_train_time = time.time() - start_time
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print("Tempo de execução do Random Forest:", rf_train_time, "segundos")
    print("Acurácia do Random Forest:", rf_accuracy)

    rf_times = []
    rf_accuracies = []
    for m in range(30):
        start_time = time.time()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_times.append(time.time() - start_time)
        rf_accuracies.append(accuracy_score(y_test, rf_pred))
        print(f"Execução para 30x {m+1}:")
        print("    Tempo de execução do Random Forest:", rf_times[-1], "segundos")
        print("    Acurácia do Random Forest:", rf_accuracies[-1])

    add_results("rf", "ml", [rf_train_time], [rf_accuracy], rf_times, rf_accuracies, i+1)

print(results)

# Converte a lista de resultados em um DataFrame
df_results = pd.DataFrame(results)
print(df_results)

# Exporta o DataFrame para um arquivo CSV
df_results.to_csv("resultados.csv", index=False)
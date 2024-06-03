import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(train_file, test_file):
    train_data = pd.read_csv(train_file, delimiter=',')
    test_data = pd.read_csv(test_file, delimiter=',')

    label_encoder = LabelEncoder()
    train_data['Label'] = label_encoder.fit_transform(train_data['Label'])
    test_data['Label'] = label_encoder.fit_transform(test_data['Label'])

    X_train = train_data.drop('Label', axis=1)
    X_test = test_data.drop('Label', axis=1)
    y_train = train_data['Label']
    y_test = test_data['Label']

    return X_train, X_test, y_train, y_test

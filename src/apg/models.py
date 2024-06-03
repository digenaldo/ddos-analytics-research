from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    gnb = GaussianNB(var_smoothing=1e-9)
    gnb.fit(X_train, y_train)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    return gnb, clf, log_reg, rf_classifier

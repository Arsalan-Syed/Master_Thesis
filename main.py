import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def _apply_transformation(df):
    normalize_df = df.drop(["transactionid", "commitdate", "fix", "bug"], axis=1)
    normalize_df = np.log(normalize_df + 1)
    normalize_df["fix"] = df["fix"]
    return normalize_df


def preprocess(df):
    X = _apply_transformation(df)
    y = df["bug"]
    X = np.array(X)
    y = np.array(y)
    return X, y


def raw(df):
    y = df["bug"]
    X = df.drop(["bug", "transactionid", "commitdate"], axis=1)
    X = np.array(X)
    y = np.array(y)
    return X, y


def get_base_model(model_type):
    if model_type == "LR":
        return LogisticRegression(random_state=0, solver='lbfgs', max_iter=5000)
    elif model_type == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    elif model_type == "ADA":
        return AdaBoostClassifier()
    elif model_type == "XGB":
        return XGBClassifier(random_state=42)
    else:
        raise Exception("Unrecognized model type: %s" % model_type)


def evaluate(model_type, dataset, mode):
    df = pd.read_csv(dataset)

    if mode == "RAW":
        X, y = raw(df)
    else:
        X, y = preprocess(df)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    kf.get_n_splits(X)

    acc_array = []
    prec_array = []
    rec_array = []
    f1_array = []

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        if mode == "UNDER":
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        elif mode == "SMOTE":
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        # Define our supervised, semi-supervised models
        classifier = get_base_model(model_type)

        # Train all classifiers on the same data
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        acc_array.append(accuracy)
        prec_array.append(precision)
        rec_array.append(recall)
        f1_array.append(f1)

    row = {"Accuracy": np.mean(acc_array),
           "Precision": np.mean(prec_array),
           "Recall": np.mean(rec_array),
           "F1": np.mean(f1_array),
           "Dataset": dataset,
           "Model": model_type,
           "Mode": mode
           }
    print(row)
    print("")
    return row


modes = ["RAW", "UNDER", "SMOTE"]
datasets = ["bugzilla.csv", "columba.csv", "jdt.csv", "mozilla.csv", "platform.csv", "postgres.csv"]
models = ["RF", "XGB"]

def run():
    rows = []

    for mode in modes:

        for dataset in datasets:

            for model_type in models:
                row = evaluate(model_type, "datasets/%s" % dataset, mode)
                rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["Precision", "Recall", "F1", "Dataset", "Mode"]]
    df.to_csv("results.csv")


run()

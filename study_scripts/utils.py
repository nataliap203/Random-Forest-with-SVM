from forest.forest import RandomForest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

def make_raport(means: dict, id3: int, svms: int, output_file: str, param_c: float, f1: float, prec: float, rec: float, train_time: float, pred_time: float):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f'{{"n_models":"{id3 + svms}","num_id3":"{id3}","num_svm":"{svms}","param_c":"{param_c}",'
                f'"mean_accuracy":"{means["mean"]}","std_accuracy":"{means["std"]}","min_accuracy":"{means["min"]}","max_accuracy":"{means["max"]}",'
                f'"recall":"{rec}","f1":"{f1}","precison":"{prec}",'
                f'"training_time_seconds":"{train_time:.4f}","prediction_time_seconds":"{pred_time:.4f}"}}\n')
        f.close()

def study_case(forest: RandomForest | RandomForestClassifier, X: pd.DataFrame, y: pd.Series, iterations: int, labels: list):
    best = 0
    matrix = None
    accuracies = []
    recalls= []
    precisions = []
    f1s = []
    train_times = []
    pred_times = []
    encoder = LabelEncoder()
    encoder.fit_transform(y)

    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + i * 2)

        y_train_enc = pd.Series(encoder.transform(y_train))
        y_test_enc = pd.Series(encoder.transform(y_test))

        start_train_time = time.perf_counter()
        forest.fit(X_train, y_train_enc)
        end_train_time = time.perf_counter()
        train_times.append(end_train_time-start_train_time)

        start_pred_time = time.perf_counter()
        predictions = forest.predict(X_test)
        end_pred_time = time.perf_counter()
        pred_times.append(end_pred_time-start_pred_time)

        pd.Series(encoder.inverse_transform(predictions))

        acc = accuracy_score(y_test_enc, predictions)
        accuracies.append(acc)
        if acc > best:
            best = acc
            matrix = confusion_matrix(pd.Series(encoder.inverse_transform(y_test_enc)), pd.Series(encoder.inverse_transform(predictions)), labels=labels)

        rec = recall_score(y_test_enc, predictions, average='macro')
        recalls.append(rec)

        prec = precision_score(y_test_enc, predictions, average='macro', zero_division=0)
        precisions.append(prec)

        f1 = f1_score(y_test_enc, predictions, average='macro')
        f1s.append(f1)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    max_acc = np.max(accuracies)
    min_acc = np.min(accuracies)

    mean_f1 = np.mean(f1s)
    mean_prec = np.mean(precisions)
    mean_rec = np.mean(recalls)

    train_time = np.mean(train_times)
    pred_time = np.mean(pred_times)

    return (
        matrix,
        {
            'mean': mean_acc,
            'std': std_acc,
            'min': min_acc,
            'max': max_acc
        },
        mean_f1,
        mean_prec,
        mean_rec,
        train_time,
        pred_time
    )

def run_grid(X: pd.DataFrame, y: pd.Series, iterations: int, label_range: list, dataset_name: str, RAPORTS_DIR_NAME: str):
    if os.path.exists(f"{RAPORTS_DIR_NAME}/{dataset_name}.jsonl"):
        os.remove(f"{RAPORTS_DIR_NAME}/{dataset_name}.jsonl")

    MATRIX_DIR_NAME = f"{dataset_name}_matrixes"
    os.makedirs(MATRIX_DIR_NAME, exist_ok=True)
    n_models = [1, 10, 25, 50, 75, 100]
    id3_ratio = [0.0, 0.25, 0.5, 0.75, 1.0]
    c = [1, 5, 10]

    for n in n_models:
        for ratio in id3_ratio:
            for param_c in c:
                forest = RandomForest(num_ID3=int(n*ratio), num_SVM=n-int(n*ratio), svm_regularization=param_c)
                matrix, acc, f1, prec, rec, train_time, pred_time  = study_case(forest, X, y, iterations, label_range)
                make_raport(acc, int(n*ratio), n-int(n*ratio), f"{RAPORTS_DIR_NAME}/{dataset_name}.jsonl", param_c, f1, prec, rec, train_time, pred_time)
                plt.figure(figsize=(6,4))
                sns.heatmap(matrix, annot=True, cmap='Oranges', fmt="d", xticklabels=label_range, yticklabels=label_range)
                plt.title(f"n_models = {n}, num_id3 = {int(n*ratio)}, num_svm = {n-int(n*ratio)}, param_c = {param_c}")
                plt.savefig(f"{MATRIX_DIR_NAME}/{n}_{int(n*ratio)}_{n-int(n*ratio)}_{param_c}.png")
                plt.close()


def forest_comparision(X: pd.DataFrame, y: pd.Series, iterations: int, label_range: list, dataset_name: str, RAPORTS_DIR_NAME: str):
    if os.path.exists(f"{RAPORTS_DIR_NAME}/library_{dataset_name}.jsonl"):
        os.remove(f"{RAPORTS_DIR_NAME}/library_{dataset_name}.jsonl")

    MATRIX_DIR_NAME = f"library_{dataset_name}_matrixes"
    os.makedirs(MATRIX_DIR_NAME, exist_ok=True)
    n_models = [1, 10, 25, 50, 75, 100]

    for n in n_models:
        forest = RandomForestClassifier(n_estimators=n, criterion="entropy", max_depth=10, bootstrap=True, max_features="sqrt", random_state=42)
        matrix, acc, f1, prec, rec, train_time, pred_time  = study_case(forest, X, y, iterations, label_range)
        make_raport(acc, n, 0, f"{RAPORTS_DIR_NAME}/library_{dataset_name}.jsonl", "NaN", f1, prec, rec, train_time, pred_time)
        plt.figure(figsize=(6,4))
        sns.heatmap(matrix, annot=True, cmap='Oranges', fmt="d", xticklabels=label_range, yticklabels=label_range)
        plt.title(f"n_models = {n}, num_id3 = {n}")
        plt.savefig(f"{MATRIX_DIR_NAME}/library_{n}.png")
        plt.close()
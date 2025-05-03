from FOREST.forest import RandomForest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score


def make_raport(true: pd.Series, preds: pd.Series, label_range, means: dict, id3: int, svms: int, output_file: str):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n")
        f.write("\n")
        f.write("#############################################################################\n")
        f.write(f"Test case for {id3} ID3 trees and {svms} SVM classifiers\n\n")

        f.write(f"Mean Accuracy: {means['mean']:.4f}\n")
        f.write(f"Standard Deviation: {means['std']:.4f}\n")
        f.write(f"Best Score: {means['max']:.4f}\n")
        f.write(f"Worst Score: {means['min']:.4f}\n\n")

        accuracy = accuracy_score(true, preds)
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")

        matrix = confusion_matrix(true, preds, labels=label_range)

        header = "|  Predicted \\ Actual  | " + " | ".join([f"{label:^6}" for label in label_range]) + " |\n"
        separator = "-" * (len(header) - 1) + "\n"

        f.write("Confusion Matrix:\n")
        f.write(separator)
        f.write(header)
        f.write(separator)

        for i, row in enumerate(matrix):
            row_str = f"| {label_range[i]:^20} | " + " | ".join([f"{val:^6}" for val in row]) + " |\n"
            f.write(row_str)
            f.write(separator)

        recall = recall_score(true, preds, average='weighted')
        f.write(f"\nRecall score: {recall:.4f}\n")

        precision = precision_score(true, preds, average='weighted')
        f.write(f"Precision score: {precision:.4f}\n")

        f1 = f1_score(true, preds, average='weighted')
        f.write(f"F1 score: {f1:.4f}\n")
        f.close


def study_case(n_ID3: int, n_SVM: int, X, y, iterations: int = 25):
    overall_preds = pd.Series(dtype=int)
    overall_true = pd.Series(dtype=int)
    accuracies = []

    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + i * 2)
        encoder = LabelEncoder()

        y_train_enc = pd.Series(encoder.fit_transform(y_train))
        y_test_enc = pd.Series(encoder.transform(y_test))

        forest = RandomForest(num_ID3=n_ID3, num_SVM=n_SVM)
        forest.fit(X_train, y_train_enc)
        predictions = forest.predict(X_test)

        acc = accuracy_score(y_test_enc, predictions)
        accuracies.append(acc)

        overall_preds = pd.concat([overall_preds, pd.Series(predictions)])
        overall_true = pd.concat([overall_true, y_test_enc])

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    max_acc = np.max(accuracies)
    min_acc = np.min(accuracies)


    return (
        pd.Series(encoder.inverse_transform(overall_preds)),
        pd.Series(encoder.inverse_transform(overall_true)),
        {
            'mean': mean_acc,
            'std': std_acc,
            'min': min_acc,
            'max': max_acc
        }
    )

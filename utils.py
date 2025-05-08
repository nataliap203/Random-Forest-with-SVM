from FOREST.forest import RandomForest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score


def make_raport(means: dict, id3: int, svms: int, output_file: str, param_c: float, f1:float, prec:float, rec:float):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f'{{"m":"{id3 + svms}","ISC":"{id3},{svms},{param_c}","mean":"{means["mean"]}","std":"{means["std"]}","min":"{means["min"]}","max":"{means["max"]}","recall":"{rec}","f1":"{f1}","precison":"{prec}"}}\n')
        f.close


def study_case(n_ID3: int, n_SVM: int, X, y, iterations: int, param_c:float, labels):
    best = 0
    matrix = None
    accuracies = []
    recalls= []
    precisions = []
    f1s = []
    encoder = LabelEncoder()
    encoder.fit_transform(y)


    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + i * 2)

        y_train_enc = pd.Series(encoder.transform(y_train))
        y_test_enc = pd.Series(encoder.transform(y_test))

        forest = RandomForest(num_ID3=n_ID3, num_SVM=n_SVM, svm_regularization=param_c)
        forest.fit(X_train, y_train_enc)
        predictions = forest.predict(X_test)

        pd.Series(encoder.inverse_transform(predictions))

        acc = accuracy_score(y_test_enc, predictions)
        accuracies.append(acc)
        if acc > best:
            best = acc
            matrix = confusion_matrix(pd.Series(encoder.inverse_transform(y_test_enc)), pd.Series(encoder.inverse_transform(predictions)), labels=labels)

        rec = recall_score(y_test_enc, predictions, average='macro')
        recalls.append(rec)

        prec = precision_score(y_test_enc, predictions, average='macro')
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
        mean_rec
    )

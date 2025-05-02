from FOREST.forest import RandomForest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import warnings



def main(number_ID3, number_SVM, iterations):

    # -- MUSHROOMS --


    print("\n")
    print("#### MUSHROOMS ####")
    print("\n")
    mushrooms_df = pd.read_csv("DATA/mushrooms.csv")
    mushrooms_df = mushrooms_df.iloc[:, ::-1]
    target_column_name = mushrooms_df.columns[-1]
    feature_column_names = list(mushrooms_df.columns[:-1])
    X = mushrooms_df[feature_column_names]
    y = mushrooms_df[target_column_name]
    categorical_columns = list(X.select_dtypes(include=("object","category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    pred_labels, true_labels = study_case(number_ID3, number_SVM, X, y, iterations)
    label_range = ["p", "e"]

    make_raport(true_labels, pred_labels, label_range)
    

    # -- WINE --

    print("\n")
    print("#### WINE ####")
    print("\n")
    wine_df = pd.read_csv("DATA/WineQT.csv")
    wine_df = wine_df.iloc[:, :-1]
    target_column_name = wine_df.columns[-1]
    feature_column_names = list(wine_df.columns[:-1])
    X = wine_df[feature_column_names]
    y = wine_df[target_column_name]
    categorical_columns = list(X.select_dtypes(include=("object","category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    pred_labels, true_labels = study_case(number_ID3, number_SVM, X, y, iterations)
    label_range = list(range(1, 11))

    make_raport(true_labels, pred_labels, label_range)


    # -- CANCER --

    print("\n")
    print("#### CANCER ####")
    print("\n")
    cancer_df = pd.read_csv("DATA/data.csv")
    cancer_df = cancer_df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    cancer_df = cancer_df.iloc[:, ::-1]
    target_column_name = cancer_df.columns[-1]
    feature_column_names = list(cancer_df.columns[:-1])
    X = cancer_df[feature_column_names]
    y = cancer_df[target_column_name]

    pred_labels, true_labels = study_case(number_ID3, number_SVM, X, y, iterations)
    label_range = ["M", "B"]

    make_raport(true_labels, pred_labels, label_range)



def make_raport(true: pd.Series, preds: pd.Series, label_range):
    accuracy = accuracy_score(true, preds)
    print(f"Test Accuracy: {accuracy:.2f}")

    matrix = confusion_matrix(true, preds, labels=label_range)

    header = "|  Predicted \\ Actual  | " + " | ".join([f"{label:^6}" for label in label_range]) + " |"
    separator = "-" * (len(header) +1)


    print("Confusion Matrix:")
    print(separator)
    print(header)
    print(separator)

    for i, row in enumerate(matrix):
        row_str = f"| {label_range[i]:^20} | " + " | ".join([f"{val:^6}" for val in row]) + " |"
        print(row_str)
        print(separator)

    recall = recall_score(true, preds, average='weighted')
    print(f"Recall score for mushrooms: {recall}")


    precision = precision_score(true, preds, average='weighted')
    print(f"Precision score for mushrooms: {precision}")

    f1 = f1_score(true, preds, average='weighted')
    print(f"F1 score for mushrooms: {f1}")
    


def study_case(n_ID3: int, n_SVM: int, X, y, iterations):
    overall_preds = pd.Series()
    overall_true = pd.Series()

    for i in range(iterations):

        X_mushroom_train_df, X_mushroom_test_df, y_mushroom_train_df, y_mushroom_test_df = train_test_split(X, y, test_size=0.2, random_state=42 + iterations*2)
        encoder = LabelEncoder()

        y_mushroom_train_df = pd.Series(encoder.fit_transform(y_mushroom_train_df))
        y_mushroom_test_df = pd.Series(encoder.transform(y_mushroom_test_df))

        forest = RandomForest(num_ID3=n_ID3, num_SVM=n_SVM)
        forest.fit(X_mushroom_train_df, y_mushroom_train_df)

        predictions = forest.predict(X_mushroom_test_df)
        overall_preds = pd.concat([overall_preds, pd.Series(predictions)])
        overall_true = pd.concat([overall_true, y_mushroom_test_df])
    return pd.Series(encoder.inverse_transform(overall_preds)), pd.Series(encoder.inverse_transform(overall_true))



if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    """
    main function:
    args:
        - Number of ID3 classifiers, specifies how many ID3 decision trees will be included
        - Number of SVM classifiers, specifies how many SVM will be included
        - Number of iterations, determines how many times the experiment should be repeated.
    """
    main(2, 2, 2)

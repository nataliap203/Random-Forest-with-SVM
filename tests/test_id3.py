# Authors: Natalia Pieczko, Antoni Grajek

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from id3.ID3 import ID3


def test_nominal_data_comparision():
    data = pd.read_csv("tests/test_data/test_data_nominal.csv")
    X = data.drop("PlayTennis", axis=1)
    y = data["PlayTennis"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)
    id3_accuracy = accuracy_score(y, id3_pred)

    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, splitter="best")
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)
    sklearn_tree_accuracy = accuracy_score(y, sklearn_tree_pred)

    assert id3_accuracy / sklearn_tree_accuracy >= 0.75


def test_numeric_data_comparision():
    data = pd.read_csv("tests/test_data/test_data_numeric.csv")
    X = data.drop("Target", axis=1)
    y = data["Target"]

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)
    id3_accuracy = accuracy_score(y, id3_pred)

    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, splitter="best")
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)
    sklearn_tree_accuracy = accuracy_score(y, sklearn_tree_pred)

    assert id3_accuracy / sklearn_tree_accuracy >= 0.9


def test_combined_data_comparision():
    data = pd.read_csv("tests/test_data/test_data_combined.csv")
    X = data.drop("Target", axis=1)
    y = data["Target"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)
    id3_accuracy = accuracy_score(y, id3_pred)

    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, splitter="best")
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)
    sklearn_tree_accuracy = accuracy_score(y, sklearn_tree_pred)

    assert id3_accuracy / sklearn_tree_accuracy >= 0.8


def test_multiclass_data_comparision():
    data = pd.read_csv("tests/test_data/test_data_multiclass.csv")
    X = data.drop("Target", axis=1)
    y = data["Target"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)
    id3_accuracy = accuracy_score(y, id3_pred)

    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, splitter="best")
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)
    sklearn_tree_accuracy = accuracy_score(y, sklearn_tree_pred)

    assert id3_accuracy / sklearn_tree_accuracy >= 0.8


def test_mushrooms_data_comparision():
    data = pd.read_csv("data/mushrooms.csv")
    X = data.drop("class", axis=1)
    y = data["class"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)
    id3_accuracy = accuracy_score(y, id3_pred)

    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, splitter="best")
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)
    sklearn_tree_accuracy = accuracy_score(y, sklearn_tree_pred)

    assert id3_accuracy / sklearn_tree_accuracy >= 0.9


def test_cancer_data_comparision():
    data = pd.read_csv("data/cancer.csv")
    X = data[1:]
    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)

    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, splitter="best")
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)

    assert (id3_pred == sklearn_tree_pred).all()


def test_wine_data_comparision():
    data = pd.read_csv("data/wine.csv")
    X = data[:-1]
    X = data.drop("quality", axis=1)
    y = data["quality"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=20)
    id3.fit(X, y)
    id3_pred = id3.predict(X)
    id3_accuracy = accuracy_score(y, id3_pred)

    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=20, splitter="best")
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)
    sklearn_tree_accuracy = accuracy_score(y, sklearn_tree_pred)

    assert id3_accuracy / sklearn_tree_accuracy >= 0.9


def test_crop_data_comparision():
    data = pd.read_csv("data/crop.csv")
    X = data[:-1]
    X = data.drop("label", axis=1)
    y = data["label"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=20)
    id3.fit(X, y)
    id3_pred = id3.predict(X)
    id3_accuracy = accuracy_score(y, id3_pred)

    sklearn_tree = DecisionTreeClassifier(criterion="entropy", max_depth=20, splitter="best")
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)
    sklearn_tree_accuracy = accuracy_score(y, sklearn_tree_pred)

    assert id3_accuracy / sklearn_tree_accuracy >= 0.9

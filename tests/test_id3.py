import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from id3.ID3 import ID3


def test_nominal_data_comparision():
    data = pd.read_csv("tests/data/test_data_nominal.csv")
    X = data.drop("PlayTennis", axis=1)
    y = data["PlayTennis"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)

    sklearn_tree = DecisionTreeClassifier(
        criterion="entropy", max_depth=10, splitter="best"
    )
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)

    assert (id3_pred == sklearn_tree_pred).all()


def test_numeric_data_comparision():
    data = pd.read_csv("tests/data/test_data_numeric.csv")
    X = data.drop("Target", axis=1)
    y = data["Target"]

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)

    sklearn_tree = DecisionTreeClassifier(
        criterion="entropy", max_depth=10, splitter="best"
    )
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)

    assert (id3_pred == sklearn_tree_pred).all()


def test_combined_data_comparision():
    data = pd.read_csv("tests/data/test_data_combined.csv")
    X = data.drop("Target", axis=1)
    y = data["Target"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)

    sklearn_tree = DecisionTreeClassifier(
        criterion="entropy", max_depth=10, splitter="best"
    )
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)

    assert (id3_pred == sklearn_tree_pred).all()


def test_multiclass_data_comparision():
    data = pd.read_csv("tests/data/test_data_multiclass.csv")
    X = data.drop("Target", axis=1)
    y = data["Target"]
    categorical_columns = list(X.select_dtypes(include=("object", "category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    id3 = ID3(max_depth=10)
    id3.fit(X, y)
    id3_pred = id3.predict(X)

    sklearn_tree = DecisionTreeClassifier(
        criterion="entropy", max_depth=10, splitter="best"
    )
    sklearn_tree.fit(X, y)
    sklearn_tree_pred = sklearn_tree.predict(X)

    assert (id3_pred == sklearn_tree_pred).all()

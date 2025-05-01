from FOREST.forest import RandomForest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def main():

    # -- MUSHROOMS --
    mushrooms_df = pd.read_csv("DATA/mushrooms.csv")
    mushrooms_df = mushrooms_df.iloc[:, ::-1]

    target_column_name = mushrooms_df.columns[-1]
    feature_column_names = list(mushrooms_df.columns[:-1])

    X = mushrooms_df[feature_column_names]
    y = mushrooms_df[target_column_name]

    categorical_columns = list(X.select_dtypes(include=("object","category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)

    X_mushroom_train_df, X_mushroom_test_df, y_mushroom_train_df, y_mushroom_test_df = train_test_split(X, y, test_size=0.2, random_state=42)
    encoder = LabelEncoder()

    y_mushroom_train_df = pd.Series(encoder.fit_transform(y_mushroom_train_df))
    y_mushroom_test_df = pd.Series(encoder.transform(y_mushroom_test_df))

    forest = RandomForest(num_ID3=10, num_SVM=10)
    forest.fit(X_mushroom_train_df, y_mushroom_train_df)

    predictions = forest.predict(X_mushroom_test_df)
    accuracy = accuracy_score(y_mushroom_test_df, predictions)
    print(f"Mushroom Test Accuracy: {accuracy:.2f}")

    # -- WINE --
    wine_df = pd.read_csv("DATA/WineQT.csv")
    wine_df = wine_df.iloc[:, :-1]

    wine_train_df, wine_test_df = train_test_split(wine_df, test_size=0.2, random_state=42)
    encoder = LabelEncoder()

    target_column_name = wine_train_df.columns[-1]
    feature_column_names = list(wine_train_df.columns[:-1])

    X_train = wine_train_df[feature_column_names]
    y_train = wine_train_df[target_column_name]
    X_test = wine_test_df[feature_column_names]
    y_test = wine_test_df[target_column_name]

    y_train = pd.Series(encoder.fit_transform(y_train))
    y_test = pd.Series(encoder.transform(y_test))

    forest = RandomForest(num_ID3=10, num_SVM=10)
    forest.fit(X_train, y_train)

    predictions = forest.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Wine Test Accuracy Label: {accuracy:.2f}")


    # -- CANCER --
    cancer_df = pd.read_csv("DATA/data.csv")

    cancer_df = cancer_df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    cancer_df = cancer_df.iloc[:, ::-1]

    cancer_train_df, cancer_test_df = train_test_split(cancer_df, test_size=0.2, random_state=42)
    encoder = LabelEncoder()

    target_column_name = cancer_train_df.columns[-1]
    feature_column_names = list(cancer_train_df.columns[:-1])

    X_train = cancer_train_df[feature_column_names]
    y_train = cancer_train_df[target_column_name]
    X_test = cancer_test_df[feature_column_names]
    y_test = cancer_test_df[target_column_name]

    y_train = pd.Series(encoder.fit_transform(y_train))
    y_test = pd.Series(encoder.transform(y_test))

    y_test = encoder.fit_transform(y_test)

    forest = RandomForest(num_ID3=10, num_SVM=10)
    forest.fit(X_train, y_train)

    predictions = forest.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Cancer Test Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()

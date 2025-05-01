from FOREST.forest import RandomForest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def main():
    mushrooms_df = pd.read_csv("DATA/mushrooms.csv")
    wine_df = pd.read_csv("DATA/WineQT.csv")
    cancer_df = pd.read_csv("DATA/data.csv")


    mushrooms_df =mushrooms_df.iloc[:, ::-1]
    wine_df = wine_df.iloc[:, :-1]
    cancer_df = cancer_df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    cancer_df = cancer_df.iloc[:, ::-1]

    mushroom_train_df, mushroom_test_df = train_test_split(mushrooms_df, test_size=0.2, random_state=42)
    encoder = LabelEncoder()

    X_test = mushroom_test_df.iloc[:, 0:len(mushroom_test_df.columns)-1]
    y_test = mushroom_test_df.iloc[:, len(mushroom_test_df.columns)-1]

    y_test = encoder.fit_transform(y_test)

    forest = RandomForest(num_ID3=10, num_SVMS=10, regularisation=1, kernel='rbf', degree=3, gamma='scale', data=mushroom_train_df)
    forest.train()

    predictions = forest.predict_majority_vote(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Mushroom Test Accuracy: {accuracy:.2f}")


    wine_train_df, wine_test_df = train_test_split(wine_df, test_size=0.2, random_state=42)
    encoder = LabelEncoder()

    X_test = wine_test_df.iloc[:, 0:len(wine_test_df.columns)-1]
    y_test = wine_test_df.iloc[:, len(wine_test_df.columns)-1]

    y_test = encoder.fit_transform(y_test)

    forest = RandomForest(num_ID3=10, num_SVMS=10, regularisation=1, kernel='rbf', degree=3, gamma='scale', data=wine_train_df)
    forest.train()

    predictions = forest.predict_majority_vote(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Wine Test Accuracy: {accuracy:.2f}")



    cancer_train_df, cancer_test_df = train_test_split(cancer_df, test_size=0.2, random_state=42)
    encoder = LabelEncoder()

    X_test = cancer_test_df.iloc[:, 0:len(cancer_test_df.columns)-1]
    y_test = cancer_test_df.iloc[:, len(cancer_test_df.columns)-1]

    y_test = encoder.fit_transform(y_test)

    forest = RandomForest(num_ID3=10, num_SVMS=10, regularisation=1, kernel='rbf', degree=3, gamma='scale', data=cancer_train_df)
    forest.train()

    predictions = forest.predict_majority_vote(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Cancer Test Accuracy: {accuracy:.2f}")

    




if __name__ == "__main__":
    main()
  
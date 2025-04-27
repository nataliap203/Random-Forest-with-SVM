from ID3.build_tree import build_tree
from ID3.predict import predict
from SVM.svm import train_svm, predict_SVM
import pandas as pd
import random
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

class RandomForrest:
    def __init__(self, num_ID3, num_SVMS, regularisation: int, kernel: str, degree: int, gamma:str, data: pd.DataFrame):
        self.num_ID3 = num_ID3
        self.num_SVMS = num_SVMS
        self.regularisation = regularisation
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.data = data
        self.ID3_models = []
        self.SVM_models = []

    def train(self):
        for i in range(self.num_ID3):
            all_features = list(self.data.columns[:-1])
            subset_len = np.floor(np.sqrt(len(all_features)))
            subset_len = max(1, subset_len)
            features_subset = random.sample(all_features, subset_len)
            model = build_tree(self.data, features_subset, self.data.columns[-1])
            self.ID3_models.append(model)

        for i in range(self.num_SVMS):
            model = train_svm(self.data, self.regularisation, self.kernel, self.degree, self.gamma)
            self.SVM_models.append(model)

    def predict(self, X: pd.DataFrame):
        predictions = []
        for model in self.ID3_models:
            pred = predict(model, X)
            predictions.append(pred)

        for model in self.SVM_models:
            pred = predict_SVM(model, X)
            predictions.append(pred)
        return predictions

    def predict_majority_vote(self, X: pd.DataFrame):
        predictions = self.predict(X)
        final_predictions = []
        for i in range(len(X)):
            votes = {}
            for pred in predictions:
                if pred[i] not in votes:
                    votes[pred[i]] = 1
                else:
                    votes[pred[i]] += 1
            final_predictions.append(max(votes, key=votes.get))
        return final_predictions


# mean1 = 55
# std_dev1 = 10
# num_samples = 500

# column1_numbers = np.random.normal(mean1, std_dev1, num_samples)
# column1_numbers = np.clip(column1_numbers, 30, 120)
# column1_numbers = np.round(column1_numbers).astype(int)

# mean2 = 18
# std_dev2 = 3

# column2_numbers = np.random.normal(mean2, std_dev2, num_samples)
# column2_numbers = np.clip(column2_numbers, 12, 26)
# column2_numbers = np.round(column2_numbers).astype(int)

# column3_numbers = np.random.randint(2, size=num_samples)
# column3_numbers[column1_numbers > mean1] =1

# data = {"Miles_Per_week": column1_numbers,
#         "Farthest_run": column2_numbers,
#         "Qualified_Boston_Marathon": column3_numbers}

# df = pd.DataFrame(data)

# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# # Train the model
# forest = RandomForrest(num_ID3=1, num_SVMS=1, regularisation=1, kernel='rbf', degree=3, gamma='scale', data=train_df)
# forest.train()


# X_test = test_df.iloc[:, 0:len(test_df.columns)-1]
# y_test = test_df.iloc[:, len(test_df.columns)-1]
# # Predict using the model
# predictions = forest.predict_majority_vote(X_test)
# # Calculate accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f"Test Accuracy: {accuracy:.2f}")
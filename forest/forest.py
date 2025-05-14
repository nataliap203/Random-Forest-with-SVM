from id3.ID3 import ID3
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from scipy.stats import mode


class RandomForest:
    def __init__(
        self,
        num_ID3: int,
        num_SVM: int,
        id3_max_depth: int = 10,
        svm_regularization: float = 1.0,
        svm_kernel: str = "rbf",
    ):
        self.num_ID3 = num_ID3
        self.num_SVM = num_SVM
        self.svm_regularization = svm_regularization
        self.svm_kernel = svm_kernel
        self.id3_max_depth = id3_max_depth
        self.tree_models = [ID3(max_depth=self.id3_max_depth) for _ in range(num_ID3)]
        self.SVM_models = [
            SVC(C=self.svm_regularization, kernel=self.svm_kernel)
            for _ in range(num_SVM)
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X.reset_index(drop=True, inplace=True)
        np.random.seed(42)

        for ID3_model in self.tree_models:
            bootstrap_ids = np.random.randint(0, X.shape[0], size=X.shape[0])
            X_bootstraped = X.iloc[bootstrap_ids]
            y_bootstraped = y.iloc[bootstrap_ids]

            ID3_model.fit(X_bootstraped, y_bootstraped)

        for SVM_model in self.SVM_models:
            bootstrap_ids = np.random.randint(0, X.shape[0], size=X.shape[0])
            X_bootstraped = X.iloc[bootstrap_ids]
            y_bootstraped = y.iloc[bootstrap_ids]

            SVM_model.fit(X_bootstraped, y_bootstraped)

    def predict(self, X: pd.DataFrame):
        predictions = []

        for ID3_model in self.tree_models:
            prediction = ID3_model.predict(X)
            predictions.append(prediction)

        for SVM_model in self.SVM_models:
            prediction = SVM_model.predict(X)
            predictions.append(prediction)

        all_preds_array = np.array(predictions)
        all_preds_per_sample = all_preds_array.T
        majority_votes, _ = mode(all_preds_per_sample, axis=1, keepdims=False)

        return majority_votes

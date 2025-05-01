import pandas as pd
import numpy as np 
from sklearn.svm import SVC


def train_svm(data: pd.DataFrame, regularisation: int, kernel: str, degree: int, gamma:str):
    X = data.iloc[:, 0:len(data.columns)-1]
    y = data.iloc[:,len(data.columns)-1]
    

    model = SVC(C =regularisation, kernel=kernel,  degree= degree, gamma=gamma)
    model.fit(X,y)

    return model


def predict_SVM(model:SVC, X: pd.DataFrame):
    return model.predict(X)

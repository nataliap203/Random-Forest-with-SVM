
from utils import study_case, make_raport
import pandas as pd

def cancer_grid():
    with open("RAPORTS/cancer.txt", 'w', encoding='utf-8') as f:
        f.write("###### CANCER ######")
    # m_estimators = [10, 25, 50, 75, 100]
    m_estimators = [10]
    id3_ratio = [0.0, 0.25, 0.5, 0.75, 1.0]

    cancer_df = pd.read_csv("DATA/data.csv")
    cancer_df = cancer_df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    cancer_df = cancer_df.iloc[:, ::-1]
    target_column_name = cancer_df.columns[-1]
    feature_column_names = list(cancer_df.columns[:-1])
    X = cancer_df[feature_column_names]
    y = cancer_df[target_column_name]
    label_range = ["M", "B"]

    for m in m_estimators:
        for ratio in id3_ratio:
            pred_labels, true_labels, cases = study_case(int(m*ratio), m-int(m*ratio), X, y, 2)
            make_raport(true_labels, pred_labels, label_range, cases,  int(m*ratio),  m-int(m*ratio), "RAPORTS/cancer.txt")


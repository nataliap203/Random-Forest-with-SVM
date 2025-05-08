import os
from study_scripts.utils import study_case, make_raport
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cancer_grid(RAPORTS_DIR_NAME="RAPORTS"):
    MATRIX_DIR_NAME = "CANCER_MATRIXES"
    os.makedirs(MATRIX_DIR_NAME, exist_ok=True)

    with open(f"{RAPORTS_DIR_NAME}/cancer.jsonl", 'w', encoding='utf-8') as f:
        f.close()
    # m_estimators = [10, 25, 50, 75, 100]
    m_estimators = [10]
    id3_ratio = [0.0, 0.25, 0.5, 0.75, 1.0]
    c = [0.5, 1, 5]

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
            for param_c in c:
                matrix, cases, f1, prec, rec  = study_case(int(m*ratio), m-int(m*ratio), X, y, 5, param_c, label_range)
                make_raport(cases,  int(m*ratio),  m-int(m*ratio), "RAPORTS/cancer.jsonl", param_c, f1, prec, rec)
                plt.figure(figsize=(8,5))
                sns.heatmap(matrix, annot=True, fmt="d", cmap="mako",
                xticklabels=label_range, yticklabels=label_range)
                plt.title(f"m = {m}, num_id3 = {int(m*ratio)}, num_svm = {m-int(m*ratio)}, param_c = {param_c}")
                plt.savefig(f"{MATRIX_DIR_NAME}/{m}_{int(m*ratio)}_{m-int(m*ratio)}_{param_c}.png")



from utils import study_case, make_raport
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def mushrooms_grid():
    with open("RAPORTS/shrooms.jsonl", 'w', encoding='utf-8') as f:
        f.close()
    # m_estimators = [10, 25, 50, 75, 100]
    m_estimators = [10]
    id3_ratio = [0.0, 0.25, 0.5, 0.75, 1.0]
    c = [0.5, 1, 5]

    mushrooms_df = pd.read_csv("DATA/mushrooms.csv")
    mushrooms_df = mushrooms_df.iloc[:, ::-1]
    target_column_name = mushrooms_df.columns[-1]
    feature_column_names = list(mushrooms_df.columns[:-1])
    X = mushrooms_df[feature_column_names]
    y = mushrooms_df[target_column_name]
    categorical_columns = list(X.select_dtypes(include=("object","category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)
    label_range = ["p", "e"]

    for m in m_estimators:
        for ratio in id3_ratio:
            for param_c in c:
                matrix, cases, f1, prec, rec  = study_case(int(m*ratio), m-int(m*ratio), X, y, 5, param_c, label_range)
                make_raport(cases,  int(m*ratio),  m-int(m*ratio), "RAPORTS/shrooms.jsonl", param_c, f1, prec, rec)
                print(type(matrix))
                sns.heatmap(matrix, annot=True, fmt="d", cmap="mako",
                xticklabels=label_range, yticklabels=label_range)
                plt.title(f"m = {m}, num_id3 = {int(m*ratio)}, num_svm = {m-int(m*ratio)}, param_c = {param_c}")
                plt.savefig(f"SHROOM_MATRIXES/{m}_{int(m*ratio)}_{m-int(m*ratio)}_{param_c}.png")



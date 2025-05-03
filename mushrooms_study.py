
from utils import study_case, make_raport
import pandas as pd

def mushrooms_grid():
    with open("RAPORTS/shrooms.txt", 'w', encoding='utf-8') as f:
        f.write("###### MUSHROOMS ######")
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
                pred_labels, true_labels, cases, f1, prec, rec  = study_case(int(m*ratio), m-int(m*ratio), X, y, 5, param_c)
                make_raport(true_labels, pred_labels, label_range, cases,  int(m*ratio),  m-int(m*ratio), "RAPORTS/shrooms.txt", param_c, f1, prec, rec)


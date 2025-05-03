
from utils import study_case, make_raport
import pandas as pd

def wine_grid():
    with open("RAPORTS/wine.txt", 'w', encoding='utf-8') as f:
        f.write("###### WINE ######")
    # m_estimators = [10, 25, 50, 75, 100]
    m_estimators = [10]
    id3_ratio = [0.0, 0.25, 0.5, 0.75, 1.0]
    c = [0.5, 1, 5]

    wine_df = pd.read_csv("DATA/WineQT.csv")
    wine_df = wine_df.iloc[:, :-1]
    target_column_name = wine_df.columns[-1]
    feature_column_names = list(wine_df.columns[:-1])
    X = wine_df[feature_column_names]
    y = wine_df[target_column_name]
    categorical_columns = list(X.select_dtypes(include=("object","category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)
    label_range = list(range(1, 11))

    for m in m_estimators:
        for ratio in id3_ratio:
            for param_c in c:
                pred_labels, true_labels, cases, f1, prec, rec  = study_case(int(m*ratio), m-int(m*ratio), X, y, 5, param_c)
                make_raport(true_labels, pred_labels, label_range, cases,  int(m*ratio),  m-int(m*ratio), "RAPORTS/wine.txt", param_c, f1, prec, rec)

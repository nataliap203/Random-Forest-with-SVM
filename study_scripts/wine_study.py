from study_scripts.utils import run_grid, forest_comparision
import pandas as pd

def wine_study(iterations: int = 25, raports_dir_name: 25 = "RAPORTS"):
    DATASET_NAME = "wine"

    wine_df = pd.read_csv(f"data/{DATASET_NAME}.csv")
    wine_df = wine_df.iloc[:, :-1]
    target_column_name = wine_df.columns[-1]
    feature_column_names = list(wine_df.columns[:-1])
    X = wine_df[feature_column_names]
    y = wine_df[target_column_name]
    categorical_columns = list(X.select_dtypes(include=("object","category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)
    label_range = list(range(1, 11))

    run_grid(X, y, iterations, label_range, DATASET_NAME, raports_dir_name)
    forest_comparision(X, y, iterations, label_range, DATASET_NAME, raports_dir_name)
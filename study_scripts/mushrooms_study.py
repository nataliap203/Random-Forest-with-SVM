from study_scripts.utils import run_grid
import pandas as pd

def mushrooms_study(iterations: int = 25, raports_dir_name: str = "RAPORTS"):
    DATASET_NAME = "mushrooms"

    mushrooms_df = pd.read_csv(F"data/{DATASET_NAME}.csv")
    mushrooms_df = mushrooms_df.iloc[:, ::-1]
    target_column_name = mushrooms_df.columns[-1]
    feature_column_names = list(mushrooms_df.columns[:-1])
    X = mushrooms_df[feature_column_names]
    y = mushrooms_df[target_column_name]
    categorical_columns = list(X.select_dtypes(include=("object","category")).columns)
    X = pd.get_dummies(X, columns=categorical_columns)
    label_range = ["p", "e"]

    run_grid(X, y, iterations, label_range, DATASET_NAME, raports_dir_name)

from study_scripts.utils import run_grid
import pandas as pd

def cancer_study(iterations: int = 25, raports_dir_name: str = "RAPORTS"):
    DATASET_NAME = "cancer"

    cancer_df = pd.read_csv(f"data/{DATASET_NAME}.csv")
    cancer_df = cancer_df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    cancer_df = cancer_df.iloc[:, ::-1]
    target_column_name = cancer_df.columns[-1]
    feature_column_names = list(cancer_df.columns[:-1])
    X = cancer_df[feature_column_names]
    y = cancer_df[target_column_name]
    label_range = ["M", "B"]

    run_grid(X, y, iterations, label_range, DATASET_NAME, raports_dir_name)

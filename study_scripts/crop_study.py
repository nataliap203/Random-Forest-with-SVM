# Authors: Natalia Pieczko, Antoni Grajek

from study_scripts.utils import run_grid, forest_comparision
import pandas as pd


def crop_study(iterations: int = 25, raports_dir_name: 25 = "RAPORTS"):
    DATASET_NAME = "crop"

    crop_df = pd.read_csv(f"data/{DATASET_NAME}.csv")
    target_column_name = crop_df.columns[-1]
    feature_column_names = list(crop_df.columns[:-1])
    X = crop_df[feature_column_names]
    y = crop_df[target_column_name]
    label_range = y.unique()

    run_grid(X, y, iterations, label_range, DATASET_NAME, raports_dir_name)
    forest_comparision(X, y, iterations, label_range, DATASET_NAME, raports_dir_name)

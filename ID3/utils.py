import numpy as np
import pandas as pd
from collections import Counter

def calculate_entropy(y):
    counts = Counter(y)
    total_count = len(y)
    entropy = 0.0

    if total_count <= 1:
        return 0

    for count in counts.values():
        p_i = count / total_count
        if p_i >0:
            entropy -= p_i * np.log2(p_i)

    return entropy


def calculate_info_gain(data: pd.DataFrame, feature_name: str, target_name: str, threshold=None):
    total_entropy = calculate_entropy(data[target_name])

    weighted_entropy_after_split = 0
    total_count = len(data)

    if total_count == 0:
        return 0

    if threshold is None:
        unique_values = data[feature_name].unique()

        for value in unique_values:
            subset = data[data[feature_name] == value]
            subset_count = len(subset)

            if subset_count > 0:
                subset_entropy = calculate_entropy(subset[target_name])
                weight = subset_count / total_count
                weighted_entropy_after_split += weight * subset_entropy
    else:
        subset_left = data[data[feature_name] <= threshold]
        subset_right = data[data[feature_name] > threshold]

        count_left = len(subset_left)
        count_right = len(subset_right)

        if count_left > 0:
            entropy_left = calculate_entropy(subset_left[target_name])
            weight_left = count_left / total_count
            weighted_entropy_after_split += weight_left * entropy_left

        if count_right > 0:
            entropy_right = calculate_entropy(subset_right[target_name])
            weight_right = count_right / total_count
            weighted_entropy_after_split += weight_right * entropy_right

    info_gain = total_entropy - weighted_entropy_after_split
    return info_gain


def find_best_split(data: pd.DataFrame, feature_names: list, target_name: str):
    best_feature = None
    best_threshold = None
    max_info_gain = -1

    for feature in feature_names:
        if pd.api.types.is_numeric_dtype(data[feature]):
            unique_values = sorted(data[feature].unique())
            if len(unique_values) > 1:
                potential_thresholds = [(unique_values[i] + unique_values[i+1]) / 2
                                        for i in range(len(unique_values) - 1)]

                for threshold in potential_thresholds:
                    current_info_gain = calculate_info_gain(data, feature, target_name, threshold=threshold)
                    if current_info_gain > max_info_gain:
                        max_info_gain = current_info_gain
                        best_feature = feature
                        best_threshold = threshold
        else:
            current_info_gain = calculate_info_gain(data, feature, target_name, threshold=None)
            if current_info_gain > max_info_gain:
                max_info_gain = current_info_gain
                best_feature = feature
                best_threshold = None


    return best_feature, best_threshold, max_info_gain

def get_majority_class(y):
    counts = Counter(y)
    return counts.most_common(1)[0][0] if counts else None

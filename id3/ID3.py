# Authors: Natalia Pieczko, Antoni Grajek

import pandas as pd
import numpy as np
from id3.node import Node
from id3.utils import find_best_split, get_majority_class
import random


class ID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self._feature_names = None
        self._target_name = None

    def _build_tree(
        self,
        data: pd.DataFrame,
        feature_names: list,
        target_name: str,
        current_depth: int = 0,
    ):
        y = data[target_name]
        num_unique_classes = y.nunique()

        # all samples have same class
        if num_unique_classes == 1:
            return Node(value=y.iloc[0], is_leaf=True)
        # no atributes to split
        if not feature_names:
            return Node(value=get_majority_class(y), is_leaf=True)
        # has max depth
        if self.max_depth is not None and current_depth >= self.max_depth:
            return Node(value=get_majority_class(y), is_leaf=True)

        feature_names_subset_len = max(1, int(np.floor(np.sqrt(len(feature_names)))))
        features_names_subset = random.sample(feature_names, feature_names_subset_len)

        best_feature, best_threshold, max_info_gain = find_best_split(data, features_names_subset, target_name)

        if max_info_gain <= 0:
            return Node(value=get_majority_class(y), is_leaf=True)

        node = Node(feature=best_feature, threshold=best_threshold)
        node.children = {}

        remaining_features = [f for f in feature_names if f != best_feature]

        if best_threshold is None:
            unique_values = data[best_feature].unique()
            for value in unique_values:
                subset = data[data[best_feature] == value]

                node.children[value] = self._build_tree(subset, remaining_features, target_name, current_depth + 1)
        else:
            subset_left = data[data[best_feature] <= best_threshold]
            subset_right = data[data[best_feature] > best_threshold]

            split_key_left = f"<={best_threshold}"
            split_key_right = f">{best_threshold}"

            node.children[split_key_left] = self._build_tree(subset_left, remaining_features, target_name, current_depth + 1)

            node.children[split_key_right] = self._build_tree(subset_right, remaining_features, target_name, current_depth + 1)

        return node

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self._feature_names = list(X.columns)
        try:
            self._target_name = y.name
            if self._target_name is None:
                self._target_name = "target"
                y.name = self._target_name
        except AttributeError:
            self._target_name = "target"
            y.name = self._target_name

        data = pd.concat([X, y], axis=1)

        self.root = self._build_tree(data, self._feature_names, self._target_name, current_depth=0)
        return self

    def _predict_sample(self, node: Node, sample: pd.Series):
        while not node.is_leaf:
            feature = node.feature
            threshold = node.threshold

            try:
                sample_value = sample[feature]
            except KeyError:
                print(f"Error: No feature '{feature}' in sample. Cannot continue prediction.")
                return None

            if threshold is None:
                if sample_value in node.children:
                    node = node.children[sample_value]
                else:
                    print(f"Error: Unknown value '{sample_value}' for '{feature}'.")
                    return None
            else:
                split_key = f"<={threshold}" if sample_value <= threshold else f">{threshold}"

                if split_key in node.children:
                    node = node.children[split_key]
                else:
                    print(f"Error: No branch for '{split_key}' in node {feature}.")
                    return None

        return node.value

    def predict(self, X: pd.DataFrame):
        if self.root is None:
            raise Exception("ID3 was not train. Use fit before predict method.")

        if not all(f in X.columns for f in self._feature_names):
            missing_cols = [f for f in self._feature_names if f not in X.columns]
            raise ValueError(f"Missing columns in data X: {missing_cols}")

        predictions = []
        for index, row in X.iterrows():
            prediction = self._predict_sample(self.root, row)
            predictions.append(prediction)

        return np.array(predictions)

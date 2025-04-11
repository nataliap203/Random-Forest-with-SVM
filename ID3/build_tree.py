import pandas as pd
from ID3.node import Node
from ID3.utils import get_majority_class, find_best_split


def build_tree(data: pd.DataFrame, feature_names: list, target_name: str, current_depth: int = 0, max_depth=None):
    y = data[target_name]
    num_unique_classes = y.nunique()

    # all samples have same class
    if num_unique_classes == 1:
        leaf_value = y.iloc[0]
        return Node(value=leaf_value, is_leaf=True)

    # no atributes to split
    if not feature_names:
        return Node(value=get_majority_class(y), is_leaf=True)

    # has max depth
    if max_depth is not None and current_depth >= max_depth:
        return Node(value=get_majority_class(y), is_leaf=True)


    best_feature, best_threshold, max_info_gain = find_best_split(data, feature_names, target_name)

    if max_info_gain <= 0:
        return Node(value=get_majority_class(y), is_leaf=True)

    node = Node(feature=best_feature, threshold=best_threshold)
    node.children = {}

    remaining_features = [f for f in feature_names if f != best_feature]

    if best_threshold is None:
        unique_values = data[best_feature].unique()
        for value in unique_values:
            subset = data[data[best_feature] == value]

            if len(subset) > 0:
                node.children[value] = build_tree(subset, remaining_features, target_name, current_depth+1, max_depth)
            else:
                node.children[value] = Node(value=get_majority_class(y), is_leaf=True)

    else:
        subset_left = data[data[best_feature] <= best_threshold]
        subset_right = data[data[best_feature] > best_threshold]

        split_key_left = f"<={best_threshold}"
        split_key_right = f">{best_threshold}"

        if len(subset_left) > 0:
            node.children[split_key_left] = build_tree(subset_left, remaining_features, target_name, current_depth+1, max_depth)
        else:
            node.children[split_key_left] = Node(value=get_majority_class(y), is_leaf=True)

        if len(subset_right) > 0:
            node.children[split_key_right] = build_tree(subset_right, remaining_features, target_name, current_depth+1, max_depth)
        else:
            node.children[split_key_right] = Node(value=get_majority_class(y), is_leaf=True)

    return node





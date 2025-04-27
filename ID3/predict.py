import pandas as pd
from ID3.node import Node

def predict_sample(node: Node, sample: pd.Series):
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
            if sample_value <= threshold:
                split_key = f"<={threshold}"
            else:
                split_key = f">{threshold}"
            if split_key in node.children:
                node = node.children[split_key]
            else:
                print(f"Error: No branch for '{split_key}' in node {feature}.")
                return None

    return node.value

def predict(tree_root: Node, X: pd.DataFrame):
    if tree_root is None:
        raise ValueError("Tree root was not built")

    predictions = []
    for index, row in X.iterrows():
        prediction = predict_sample(tree_root, row)
        predictions.append(prediction)

    return predictions




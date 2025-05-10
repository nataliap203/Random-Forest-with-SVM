class Node:
    def __init__(self, feature=None, threshold=None, children=None, value=None, is_leaf=None):
        self.feature = feature      # attribute used for split in this node
        self.threshold = threshold  # threshold for numeric attributes
        self.children = children
        self.value = value          # if is leaf, value of class
        self.is_leaf = is_leaf

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(Value: {self.value})"
        else:
            spilt_info = f"Feature {self.feature}"
            if self.threshold is not None:
                spilt_info += f", Threshold: {self.threshold}"
            return f"Node({spilt_info})"

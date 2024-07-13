import pandas as pd
import numpy as np

# Sample dataset
data = {
    'CreditScore': [700, 600, 800, 400, 650],
    'Age': [25, 40, 35, 50, 28],
    'Income': [50000, 60000, 80000, 30000, 55000],
    'Debt': [5000, 10000, 2000, 15000, 7000],
    'Creditworthy': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Features and target
X = df.drop('Creditworthy', axis=1).values
y = df['Creditworthy'].values

# Train-test split
split_ratio = 0.8
split_index = int(split_ratio * len(y))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Random Forest
class RandomForest:
    def __init__(self, n_trees, max_depth):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            indices = np.random.choice(len(y), len(y), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = self._build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return np.mean(y)
        feature, threshold = self._best_split(X, y)
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return (feature, threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        best_gain = -1
        split_feature, split_threshold = None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature
                    split_threshold = threshold
        return split_feature, split_threshold

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        n = len(y)
        n_left = len(y[left_indices])
        n_right = len(y[right_indices])
        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return parent_entropy - child_entropy

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def predict(self, X):
        predictions = [self._predict_tree(tree, x) for x in X for tree in self.trees]
        return np.round(np.mean(predictions, axis=0)).astype(int)

    def _predict_tree(self, tree, x):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_tree, right_tree = tree
        if x[feature] <= threshold:
            return self._predict_tree(left_tree, x)
        else:
            return self._predict_tree(right_tree, x)

# Initialize and train model
model = RandomForest(n_trees=10, max_depth=3)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)

print(f'Accuracy: {accuracy * 100}%')

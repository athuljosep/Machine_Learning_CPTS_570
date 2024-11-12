# Import necessary libraries
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

# Load the dataset from UCI repository
bcwd = fetch_ucirepo(id=17)

# Extract feature matrix and target vector
x = bcwd.data.features
y = bcwd.data.targets

# Combine feature and target data
data = pd.concat([x, y], axis=1)

# Map 'M' (Malignant) to 0 and 'B' (Benign) to 1 for binary classification
data['Diagnosis'] = data['Diagnosis'].map({'M': 0, 'B': 1})

data_array = data.values

# indices for training, validation, and test
n = data_array.shape[0]  # Total number of samples
train_end = int(n * 0.7)  # 70% of data for training
valid_end = int(n * 0.8)  # Next 10% for validation

# Split the dataset into training, validation, and test subsets
x_train, y_train = data_array[:train_end, :-1], data_array[:train_end, -1]  # Training set
x_valid, y_valid = data_array[train_end:valid_end, :-1], data_array[train_end:valid_end, -1]  # Validation set
x_test, y_test = data_array[valid_end:, :-1], data_array[valid_end:, -1]  # Test set

# Function to calculate entropy
def entropy(labels):
    # Handle edge case where the input array is empty
    if len(labels) == 0:
        return 0

    # Calculate the proportion of samples belonging to class '1'
    prob_class_1 = np.mean(labels == 1)  # Probability of class '1'
    prob_class_0 = 1 - prob_class_1  # Probability of class '0'

    # Compute entropy if both probabilities are non-zero
    if prob_class_1 > 0 and prob_class_0 > 0:
        return -prob_class_1 * math.log(prob_class_1) - prob_class_0 * math.log(prob_class_0)  # Entropy formula
    return 0

# Function to split the dataset based on a given threshold and feature index
def split(x, y, threshold, index):
    # Create a boolean mask to identify samples for left and right splits
    mask = x[:, index] < threshold

    # Use the mask to efficiently split the feature and target arrays
    left_x, right_x = x[mask], x[~mask]
    left_y, right_y = y[mask], y[~mask]

    left_x = [np.array(row) for row in left_x]
    right_x = [np.array(row) for row in right_x]

    return right_x, right_y, left_x, left_y

# Find the best attribute and threshold for splitting
def best_split(features, labels):
    total_entropy = entropy(labels)
    max_info_gain = np.zeros(features.shape[1])
    best_thresholds = np.zeros(features.shape[1])

    # Loop over each feature/column
    for feature_idx in range(features.shape[1]):
        sorted_feature_values = np.sort(features[:, feature_idx])
        split_entropies = np.zeros(len(sorted_feature_values) - 1)
        thresholds = np.zeros(len(sorted_feature_values) - 1)

        # Loop through possible threshold points in the sorted column
        for split_idx in range(len(sorted_feature_values) - 1):
            threshold_midpoint = sorted_feature_values[split_idx] + \
                       (sorted_feature_values[split_idx + 1] - sorted_feature_values[split_idx]) / 2
            right_features, right_labels, left_features, left_labels = split(
                features, labels, threshold_midpoint, feature_idx)

            # Calculate the weighted entropy of the split
            split_entropies[split_idx] = (
                (entropy(right_labels) * len(right_labels) +
                 entropy(left_labels) * len(left_labels)) /
                (len(right_labels) + len(left_labels))
            )
            thresholds[split_idx] = threshold_midpoint

        # Store the maximum information gain and corresponding threshold for this feature
        max_info_gain[feature_idx] = max(total_entropy - split_entropies)
        best_thresholds[feature_idx] = thresholds[np.argmax(total_entropy - split_entropies)]

    # Select the feature with the highest information gain
    best_feature = np.argmax(max_info_gain)
    best_threshold = best_thresholds[best_feature]

    return best_feature, best_threshold

# Train a decision tree using recursive splitting
def train_decision_tree(features, labels):
    # Find the best feature and threshold to split the data
    best_feature, best_threshold = best_split(features, labels)
    right_features, right_labels, left_features, left_labels = split(features, labels, best_threshold, best_feature)

    # Initialize child nodes based on entropy and label availability
    right_child = [] if len(right_labels) == 0 else [np.mean(right_labels)] if entropy(right_labels) == 0 else None
    left_child = [] if len(left_labels) == 0 else [np.mean(left_labels)] if entropy(left_labels) == 0 else None

    # Recursively build the tree if the child nodes are not pure
    if right_child is None:
        right_child = train_decision_tree(np.array(right_features), right_labels)  # Recurse for right child
    if left_child is None:
        left_child = train_decision_tree(np.array(left_features), left_labels)  # Recurse for left child

    # Return the current node with its feature, threshold, and children
    return best_feature, best_threshold, right_child, left_child

# Predict the class label for a given test sample using the decision tree
def predict(sample, tree):
    feature_index = tree[0]
    split_threshold = tree[1]

    # Traverse the appropriate branch based on the feature value
    if sample[feature_index] >= split_threshold:
        next_branch = tree[2]
    else:
        next_branch = tree[3]

    # If the next branch is a leaf node, return the class label
    if next_branch == [0.0] or next_branch == [1.0]:
        return next_branch

    # Recursively call predict on the next branch
    return predict(sample, next_branch)

# Calculate the accuracy of the decision tree model
def calc_accuracy(features, labels, tree):
    predictions = np.array([])  # Store predicted labels

    # Loop through each sample in the dataset to make predictions
    for i in range(features.shape[0]):
        predicted_label = predict(features[i], tree)  # Predict the label for the current sample
        predictions = np.append(predictions, predicted_label)  # Store the prediction

    # Calculate accuracy by comparing predicted and actual labels
    accuracy = accuracy_score(labels, predictions)
    return accuracy

# Prune the decision tree based on validation data to improve accuracy
def prune(features, labels, tree):
    pruned_tree = []

    # Split the validation data based on the current node's feature and threshold
    right_features, right_labels, left_features, left_labels = split(features, labels, tree[1], tree[0])

    # Determine majority class for right and left splits
    num_class_1_right = np.count_nonzero(right_labels)
    num_class_0_right = right_labels.shape[0] - num_class_1_right
    majority_class_right = np.argmax([num_class_0_right, num_class_1_right])  # Majority class for right split

    num_class_1_left = np.count_nonzero(left_labels)
    num_class_0_left = left_labels.shape[0] - num_class_1_left
    majority_class_left = np.argmax([num_class_0_left, num_class_1_left])  # Majority class for left split

    # Create a temporary tree with the majority class in the right child
    temp_tree = [tree[0], tree[1], majority_class_right, tree[3]]
    original_accuracy = calc_accuracy(features, labels, tree)  # Original tree accuracy
    temp_accuracy = calc_accuracy(features, labels, temp_tree)  # Temporary tree accuracy

    # If the temporary tree improves accuracy, update the pruned tree
    if temp_accuracy > original_accuracy:
        pruned_tree.extend(temp_tree)
    elif tree[2] not in [[1], [0]]:  # Recursively prune the right child
        temp_tree = prune(features, labels, tree[2])

    # Create another temporary tree with the majority class in the left child
    temp_tree = [tree[0], tree[1], tree[2], majority_class_left]
    temp_accuracy = calc_accuracy(features, labels, temp_tree)  # Temporary tree accuracy

    # If this temporary tree improves accuracy, update the pruned tree
    if temp_accuracy > original_accuracy:
        pruned_tree.extend(temp_tree)
    elif tree[2] not in [[1], [0]]:  # Recursively prune the left child
        temp_tree = prune(features, labels, tree[3])

    return pruned_tree

# main function
# Train the decision tree
tree = train_decision_tree(x_train, y_train)

# Prune the tree using validation data
pruned_tree = prune(x_valid, y_valid, tree)

# Calculate and display accuracies before pruning
print(f"Training Accuracy: {calc_accuracy(x_train, y_train, tree):.4f}")
print(f"Validation Accuracy: {calc_accuracy(x_valid, y_valid, tree):.4f}")
print(f"Testing Accuracy: {calc_accuracy(x_test, y_test, tree):.4f}")

# Calculate and display accuracies after pruning
print(f"Validation Accuracy after Pruning: {calc_accuracy(x_valid, y_valid, pruned_tree):.4f}")
print(f"Testing Accuracy after Pruning: {calc_accuracy(x_test, y_test, pruned_tree):.4f}")




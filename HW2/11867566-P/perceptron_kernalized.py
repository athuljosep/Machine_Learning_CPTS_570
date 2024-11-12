# importing libraries
import numpy as np
import pandas as pd
import matplotlib as plt
from datetime import datetime

# Polynomial Kernel Function
def polynomial_kernel(x, y, degree):
    return (1 + np.dot(x, y.T)) ** degree

# Splits data into training and validation sets based on the ratio
def split_data(train_data, split_ratio=0.8):
    train_split = int(split_ratio * len(train_data))
    return train_data[:train_split], train_data[train_split:]

# Extracts features (X) and labels (y) from the dataset
def extract_features_and_labels(data):
    X = data.iloc[:, 1:].values  # All columns except the first
    y = data.iloc[:, 0].values   # First column as labels
    return X, y

# Initializes alpha coefficients and arrays to track errors and accuracy
def initialize_parameters(train_size, num_classes, epochs):
    alpha_coef = np.zeros((train_size, num_classes))
    train_errors = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    return alpha_coef, train_errors, train_accuracy

# Computes class scores using kernel values
def compute_scores(alpha_coef, inputs, target_input, kernel):
    kernel_vals = kernel(inputs, target_input)
    return np.dot(alpha_coef.T, kernel_vals)

def train_perceptron(train_X, train_y, epochs, alpha_coef, kernel):
    train_size = len(train_X)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for i in range(train_size):
            # Get scores for the current training sample
            class_scores = compute_scores(alpha_coef[i], train_X, train_X[i], kernel)
            predicted_label = np.argmax(class_scores)  # Predicted class
            actual_label = train_y[i]  # Actual class

            if predicted_label != actual_label:
                # Update alpha for correct and incorrect classes
                alpha_coef[i, actual_label] += 1
                alpha_coef[i, predicted_label] -= 1
                train_errors[epoch] += 1  # Track error

        # Calculate training accuracy for the epoch
        train_accuracy[epoch] = (1 - train_errors[epoch] / train_size) * 100

    return train_errors, train_accuracy

def evaluate_performance(X, y, alpha_coef, kernel):
    errors = 0
    size = len(X)

    for i in range(size):
        # Compute scores for the current sample
        class_scores = compute_scores(alpha_coef, X, X[i], kernel)
        predicted_label = np.argmax(class_scores)

        if predicted_label != y[i]:
            errors += 1  # Increment error if prediction is wrong

    # Calculate accuracy percentage
    accuracy = (1 - errors / size) * 100
    return errors, accuracy

def kernelized_multi_perceptron(epochs, train_data, test_data, kernel):
    # Split the data into training and validation sets
    train_set, val_set = split_data(train_data)

    # Extract features and labels from all datasets
    train_X, train_y = extract_features_and_labels(train_set)
    val_X, val_y = extract_features_and_labels(val_set)
    test_X, test_y = extract_features_and_labels(test_data)

    # Get the number of unique classes in the dataset
    num_classes = len(np.unique(train_y))

    # Initialize parameters
    alpha_coef, train_errors, train_accuracy = initialize_parameters(
        len(train_X), num_classes, epochs
    )

    # Train the perceptron model
    train_errors, train_accuracy = train_perceptron(
        train_X, train_y, epochs, alpha_coef, kernel
    )

    # Evaluate performance on validation and test data
    val_errors, val_accuracy = evaluate_performance(val_X, val_y, alpha_coef, kernel)
    test_errors, test_accuracy = evaluate_performance(test_X, test_y, alpha_coef, kernel)

    # Return the final results
    return (
        alpha_coef,
        train_errors,
        train_accuracy,
        val_errors,
        val_accuracy,
        test_errors,
        test_accuracy,
    )

# Load the datasets
training_data = pd.read_csv("fashion-mnist_train.csv")
testing_data = pd.read_csv("fashion-mnist_test.csv")

# Train the kernelized Perceptron with a polynomial kernel
degree = 2
alpha, train_errors, train_accuracy, val_errors, val_accuracy, test_errors, test_accuracy = kernelized_multi_perceptron(
    epochs=5,  # Changed from 'periods' to 'epochs' to match variable names in the function
    train_data=training_data,
    test_data=testing_data,
    kernel=lambda x, y: polynomial_kernel(x, y, degree)
)

# Print results
print("Training Accuracy per Epoch:", train_accuracy)
print("Validation Accuracy:", val_accuracy)
print("Testing Accuracy:", test_accuracy)

# plotting figure
plt.figure()

plt.plot(list(range(1,5)), test_errors, marker='o', linestyle='-', linewidth=2, markersize=8)
plt.title('Mistakes vs Iterations', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Mistakes', fontsize=12)

# Adding grid and setting limits to make the plot clearer
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Generate a timestamp to append to the file name for uniqueness
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# Create the full file name with the timestamp
file_full_name = f"kernel_{timestamp}.png"

plt.savefig(file_full_name)

# Display the plot
plt.show()

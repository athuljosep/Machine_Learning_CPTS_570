'''Author: Athul Jose P'''

# Import necessary libraries
import numpy as np  # For numerical operations and array handling
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting and visualization
import csv  # For reading and writing CSV files
from datetime import datetime  # For working with dates and times

# Import the MNIST reader utility from the fashion_mnist package
from fashion_mnist.utils import mnist_reader

# Load training data
# x_train: Array of training images
# y_train: Array of training labels
x_train, y_train = mnist_reader.load_mnist(
    'C:/Users/athul.p/Documents/GitHub/Machine_Learning_CPTS_570/HW1/fashion_mnist/data/fashion',
    kind='train'
)

# Load test data
# x_test: Array of test images
# y_test: Array of test labels
x_test, y_test = mnist_reader.load_mnist(
    'C:/Users/athul.p/Documents/GitHub/Machine_Learning_CPTS_570/HW1/fashion_mnist/data/fashion',
    kind='t10k'
)


def save_file(file_name, header_name, train_accuracy, test_accuracy):
    """
    Saves training and testing accuracy data to a CSV file.

    Parameters:
    - file_name (str): Base name for the CSV file.
    - header_name (list): List of column headers for the CSV.
    - train_accuracy (list): List of training accuracy values.
    - test_accuracy (list): List of testing accuracy values.

    Returns:
    - None
    """
    # Generate a timestamp to append to the file name for uniqueness
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # Create the full file name with the timestamp
    full_file_name = f"{file_name}{timestamp}.csv"

    # Open the CSV file in write mode
    with open(full_file_name, mode='w', newline='') as f:
        writer = csv.writer(f)  # Initialize the CSV writer
        writer.writerow(header_name)  # Write the header row

        # Iterate over training and testing accuracy and write each pair as a row
        for train_acc, test_acc in zip(train_accuracy, test_accuracy):
            writer.writerow([train_acc, test_acc])

    # Function does not return anything
    return None


def save_figure(file_name, x1, x2, x1_label, x2_label, x_label, y_label):
    """
    Plots two datasets and saves the figure as a PNG file.

    Parameters:
    - file_name (str): Base name for the image file.
    - x1 (list or array): Data for the first plot line.
    - x2 (list or array): Data for the second plot line.
    - x1_label (str): Label for the first data line.
    - x2_label (str): Label for the second data line.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.

    Returns:
    - None
    """
    plt.figure()  # Create a new figure
    plt.plot(x1, color='blue', label=x1_label)  # Plot the first dataset in blue
    plt.plot(x2, color='red', label=x2_label)  # Plot the second dataset in red
    plt.xlabel(x_label)  # Set the label for the x-axis
    plt.ylabel(y_label)  # Set the label for the y-axis
    plt.title(f"{y_label} vs {x_label}")  # Set the title of the plot
    plt.legend(loc='upper right')  # Display the legend in the upper right corner

    # Generate a timestamp to append to the file name for uniqueness
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Create the full file name with the timestamp
    file_full_name = f"{file_name}_{timestamp}.png"

    plt.savefig(file_full_name)  # Save the figure as a PNG file
    plt.show()  # Display the plot

    # Function does not return anything
    return None

# Standard Perceptron function
def standard_perceptron(x_train, y_train, x_test, y_test, n_iter, log):
    # Initialize weights to zero
    weight = np.zeros(x_train.shape[1])
    tau = 1  # Learning rate
    train_accuracy = []  # List to store training accuracy
    test_accuracy = []  # List to store testing accuracy
    train_mistakes = []  # List to store training mistakes count
    test_mistakes = []  # List to store testing mistakes count

    # Epoch loop for the specified number of iterations
    for iter in range(n_iter):
        if log:
            print("epoch:" + str(iter))

        # Training phase
        train_count = 0  # Counter for training mistakes
        for data in range(x_train.shape[0]):
            # Predict the label using the current weights
            y_hat = np.sign(np.dot(weight, x_train[data]))
            # Convert the actual label to +1 or -1
            y_actual = 1.0 if (y_train[data] % 2) == 0 else -1.0
            # Update weights if the prediction is incorrect
            if y_hat != y_actual:
                weight += tau * y_actual * x_train[data]
                train_count += 1  # Increment mistake counter
        train_mistakes.append(train_count)  # Store training mistakes
        # Calculate and store training accuracy
        train_accuracy.append((x_train.shape[0] - train_count) * 100 / x_train.shape[0])

        # Testing phase
        test_count = 0  # Counter for testing mistakes
        for data in range(x_test.shape[0]):
            # Predict the label for the test set
            y_hat = np.sign(np.dot(weight, x_test[data]))
            # Convert the actual label to +1 or -1
            y_actual = 1.0 if (y_test[data] % 2) == 0 else -1.0
            # Count mistakes in predictions
            if y_hat != y_actual:
                test_count += 1
        test_mistakes.append(test_count)  # Store testing mistakes
        # Calculate and store testing accuracy
        test_accuracy.append((x_test.shape[0] - test_count) * 100 / x_test.shape[0])

    if log:
        # Saving the results to a CSV file
        save_file("perceptron_", ["Training Accuracy", "Testing Accuracy"], train_accuracy, test_accuracy)
        # Plotting training and testing accuracies
        save_figure("perceptron_accuracy", train_accuracy, test_accuracy, 'Training Data Accuracy', 'Testing Data Accuracy', 'Number of Iterations', 'Accuracy')

    return weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes  # Return the weights and accuracy metrics

# Passive-Aggressive algorithm function
def passive_aggressive(x_train, y_train, x_test, y_test, n_iter, log):
    # Initialize weights to zero
    weight = np.zeros(x_train.shape[1])
    train_accuracy = []  # List to store training accuracy
    test_accuracy = []  # List to store testing accuracy
    train_mistakes = []  # List to store training mistakes count
    test_mistakes = []  # List to store testing mistakes count

    # Epoch loop for the specified number of iterations
    for iter in range(n_iter):
        # print("epoch:" + str(iter))

        # Training phase
        train_count = 0  # Counter for training mistakes
        for data in range(x_train.shape[0]):
            # Predict the label using the current weights
            y_hat = np.sign(np.dot(weight, x_train[data]))
            # Convert the actual label to +1 or -1
            y_actual = 1.0 if (y_train[data] % 2) == 0 else -1.0
            # Update weights if the prediction is incorrect
            if y_hat != y_actual:
                # Calculate the step size based on the passive-aggressive update rule
                tau = (1 - (y_actual * np.dot(weight, x_train[data]))) / (np.square(np.linalg.norm(x_train[data])))
                weight += tau * y_actual * x_train[data]
                train_count += 1  # Increment mistake counter
        train_mistakes.append(train_count)  # Store training mistakes
        # Calculate and store training accuracy
        train_accuracy.append((x_train.shape[0] - train_count) * 100 / x_train.shape[0])

        # Testing phase
        test_count = 0  # Counter for testing mistakes
        for data in range(x_test.shape[0]):
            # Predict the label for the test set
            y_hat = np.sign(np.dot(weight, x_test[data]))
            # Convert the actual label to +1 or -1
            y_actual = 1.0 if (y_test[data] % 2) == 0 else -1.0
            # Count mistakes in predictions
            if y_hat != y_actual:
                test_count += 1
        test_mistakes.append(test_count)  # Store testing mistakes
        # Calculate and store testing accuracy
        test_accuracy.append((x_test.shape[0] - test_count) * 100 / x_test.shape[0])

    if log:
        # Saving the results to a CSV file
        with open("pa_algorithm_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Training Accuracy", "Testing Accuracy"])  # Write header
            for row in zip(train_accuracy, test_accuracy):  # Write each accuracy pair
                writer.writerow(row)

        # Plotting training and testing accuracies
        plt.figure()
        plt.plot(train_accuracy, color='blue', label='Training Data Accuracy')
        plt.plot(test_accuracy, color='red', label='Testing Data Accuracy')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number of Iterations')
        plt.legend(loc='lower right')
        plt.show()

    return weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes  # Return the weights and accuracy metrics

# Average Perceptron function
def average_perceptron(x_train, y_train, x_test, y_test, n_iter, log):
    # Initialize weights to zero
    weight = np.zeros(x_train.shape[1])
    weight_sum = np.zeros(x_train.shape[1])  # Sum of weights for averaging
    weight_count = 0  # Count of updates to compute average
    tau = 1  # Learning rate
    train_accuracy = []  # List to store training accuracy
    test_accuracy = []  # List to store testing accuracy
    train_mistakes = []  # List to store training mistakes count
    test_mistakes = []  # List to store testing mistakes count

    # Epoch loop for the specified number of iterations
    for iter in range(n_iter):
        # print("epoch:" + str(iter))

        # Training phase
        train_count = 0  # Counter for training mistakes
        for data in range(x_train.shape[0]):
            # Predict the label using the current weights
            y_hat = np.sign(np.dot(weight, x_train[data]))
            # Convert the actual label to +1 or -1
            y_actual = 1.0 if (y_train[data] % 2) == 0 else -1.0
            # Update weights if the prediction is incorrect
            if y_hat != y_actual:
                weight += tau * y_actual * x_train[data]
                train_count += 1  # Increment mistake counter
            weight_sum += weight  # Accumulate weights for averaging
            weight_count += 1  # Increment weight count
        train_mistakes.append(train_count)  # Store training mistakes
        # Calculate and store training accuracy
        train_accuracy.append((x_train.shape[0] - train_count) * 100 / x_train.shape[0])

        # Testing phase
        test_count = 0  # Counter for testing mistakes
        weight = weight_sum / weight_count  # Average the weights
        for data in range(x_test.shape[0]):
            # Predict the label for the test set
            y_hat = np.sign(np.dot(weight, x_test[data]))
            # Convert the actual label to +1 or -1
            y_actual = 1.0 if (y_test[data] % 2) == 0 else -1.0
            # Count mistakes in predictions
            if y_hat != y_actual:
                test_count += 1
        test_mistakes.append(test_count)  # Store testing mistakes
        # Calculate and store testing accuracy
        test_accuracy.append((x_test.shape[0] - test_count) * 100 / x_test.shape[0])

    if log:
        # Saving the results to a CSV file
        with open("average_perceptron_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Training Accuracy", "Testing Accuracy"])  # Write header
            for row in zip(train_accuracy, test_accuracy):  # Write each accuracy pair
                writer.writerow(row)

        # Plotting training and testing accuracies
        plt.figure()
        plt.plot(train_accuracy, color='blue', label='Training Data Accuracy')
        plt.plot(test_accuracy, color='red', label='Testing Data Accuracy')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number of Iterations')
        plt.legend(loc='lower right')
        plt.show()

    return weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes  # Return the weights and accuracy metrics

# Function to calculate F based on inputs
def F_calc(x, y, n, d):
    F = np.zeros(n)  # Initialize F with zeros
    F[d*y:d*(y+1)] = x  # Assign values to F based on input x and index y
    return F  # Return the calculated F

# Multiclass Perceptron function
def multiclass_perceptron(x_train, y_train, x_test, y_test, n_iter, log):
    # Determine the number of unique classes and the dimensionality of the input
    k = len(np.unique(y_train))
    d = x_train.shape[1]
    weight_len = k * d  # Total weight length for all classes
    weight = np.zeros(weight_len)  # Initialize weights to zero

    tau = 1  # Learning rate
    train_accuracy = []  # List to store training accuracy
    test_accuracy = []  # List to store testing accuracy
    train_mistakes = []  # List to store training mistakes count
    test_mistakes = []  # List to store testing mistakes count

    # Epoch loop for the specified number of iterations
    for iter in range(n_iter):
        if log:
            print("epoch:" + str(iter))

        # Training phase
        train_count = 0  # Counter for training mistakes
        for data in range(x_train.shape[0]):
            y_hat = []  # List to store predictions for each class
            for y_t in range(k):
                # Calculate feature vector for the current class
                F_y_t = F_calc(x_train[data], y_t, weight_len, d)
                # Append the dot product of weights and feature vector to predictions
                y_hat.append(np.dot(weight, F_y_t))
            y_hat_max = np.argmax(y_hat)  # Get the predicted class with the highest score

            y_actual = int(y_train[data])  # Get the actual class label
            # If the prediction is incorrect, update weights
            if y_hat_max != y_actual:
                F_y_t = F_calc(x_train[data], y_actual, weight_len, d)  # Feature vector for actual class
                F_y_hat = F_calc(x_train[data], y_hat_max, weight_len, d)  # Feature vector for predicted class
                # Update weights based on the prediction error
                weight += tau * (F_y_t - F_y_hat)
                train_count += 1  # Increment mistake counter
        train_mistakes.append(train_count)  # Store training mistakes
        # Calculate and store training accuracy
        train_accuracy.append((x_train.shape[0] - train_count) * 100 / x_train.shape[0])

        # Testing phase
        test_count = 0  # Counter for testing mistakes
        for data in range(x_test.shape[0]):
            y_hat = []  # List to store predictions for each class
            for y_t in range(k):
                # Calculate feature vector for the current class
                F = F_calc(x_test[data], y_t, weight_len, d)
                # Append the dot product of weights and feature vector to predictions
                y_hat.append(np.dot(weight, F))
            y_hat_max = np.argmax(y_hat)  # Get the predicted class with the highest score

            y_actual = int(y_test[data])  # Get the actual class label
            # Count mistakes in predictions
            if y_hat_max != y_actual:
                test_count += 1
        test_mistakes.append(test_count)  # Store testing mistakes
        # Calculate and store testing accuracy
        test_accuracy.append((x_test.shape[0] - test_count) * 100 / x_test.shape[0])

    if log:
        # Save results to CSV
        file_name = f"multi_perceptron_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
        with open(file_name, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Training Accuracy", "Testing Accuracy"])  # Write header
            for row in zip(train_accuracy, test_accuracy):  # Write each accuracy pair
                writer.writerow(row)

        # Plotting accuracies
        plt.figure()
        plt.plot(train_accuracy, color='blue', label='Training Data Accuracy')
        plt.plot(test_accuracy, color='red', label='Testing Data Accuracy')
        plt.xlabel('Number of Iterations')  # Set x-axis label
        plt.ylabel('Accuracy')  # Set y-axis label
        plt.title('Accuracy vs Number of Iterations')  # Set title
        plt.legend(loc='lower right')  # Add legend
        plt.show()  # Display plot

    return weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes  # Return weights and accuracy metrics

# Multiclass Passive-Aggressive algorithm function
def multiclass_passive_aggressive(x_train, y_train, x_test, y_test, n_iter, log):
    # Determine the number of unique classes and the dimensionality of the input
    k = len(np.unique(y_train))
    d = x_train.shape[1]
    weight_len = k * d  # Total weight length for all classes
    weight = np.zeros(weight_len)  # Initialize weights to zero

    train_accuracy = []  # List to store training accuracy
    test_accuracy = []  # List to store testing accuracy
    train_mistakes = []  # List to store training mistakes count
    test_mistakes = []  # List to store testing mistakes count

    # Epoch loop for the specified number of iterations
    for iter in range(n_iter):
        if log:
            print("epoch:" + str(iter))

        # Training phase
        train_count = 0  # Counter for training mistakes
        for data in range(x_train.shape[0]):
            y_hat = []  # List to store predictions for each class
            for y_t in range(k):
                # Calculate feature vector for the current class
                F_y_t = F_calc(x_train[data], y_t, weight_len, d)
                # Append the dot product of weights and feature vector to predictions
                y_hat.append(np.dot(weight, F_y_t))
            y_hat_max = np.argmax(y_hat)  # Get the predicted class with the highest score

            y_actual = int(y_train[data])  # Get the actual class label

            # If the prediction is incorrect, update weights
            if y_hat_max != y_actual:
                F_y_t = F_calc(x_train[data], y_actual, weight_len, d)  # Feature vector for actual class
                F_y_hat = F_calc(x_train[data], y_hat_max, weight_len, d)  # Feature vector for predicted class
                # Calculate step size based on the passive-aggressive update rule
                tau = (1 - (np.dot(weight, F_y_t) - np.dot(weight, F_y_hat))) / (np.square(np.linalg.norm(F_y_t - F_y_hat)))
                weight += tau * (F_y_t - F_y_hat)  # Update weights
                train_count += 1  # Increment mistake counter
        train_mistakes.append(train_count)  # Store training mistakes
        # Calculate and store training accuracy
        train_accuracy.append((x_train.shape[0] - train_count) * 100 / x_train.shape[0])

        # Testing phase
        test_count = 0  # Counter for testing mistakes
        for data in range(x_test.shape[0]):
            y_hat = []  # List to store predictions for each class
            for y_t in range(k):
                # Calculate feature vector for the current class
                F = F_calc(x_test[data], y_t, weight_len, d)
                # Append the dot product of weights and feature vector to predictions
                y_hat.append(np.dot(weight, F))
            y_hat_max = np.argmax(y_hat)  # Get the predicted class with the highest score

            y_actual = int(y_test[data])  # Get the actual class label
            # Count mistakes in predictions
            if y_hat_max != y_actual:
                test_count += 1
        test_mistakes.append(test_count)  # Store testing mistakes
        # Calculate and store testing accuracy
        test_accuracy.append((x_test.shape[0] - test_count) * 100 / x_test.shape[0])

    if log:
        # Saving the results to a CSV file
        with open("pa_algorithm_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Training Accuracy", "Testing Accuracy"])  # Write header
            for row in zip(train_accuracy, test_accuracy):  # Write each accuracy pair
                writer.writerow(row)

        # Plotting accuracies
        plt.figure()
        plt.plot(train_accuracy, color='blue', label='Training Data Accuracy')
        plt.plot(test_accuracy, color='red', label='Testing Data Accuracy')
        plt.xlabel('Number of Iterations')  # Set x-axis label
        plt.ylabel('Accuracy')  # Set y-axis label
        plt.title('Accuracy vs Number of Iterations')  # Set title
        plt.legend(loc='lower right')  # Add legend
        plt.show()  # Display plot

    return weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes  # Return weights and accuracy metrics

# Multiclass Average Perceptron function
def multiclass_average_perceptron(x_train, y_train, x_test, y_test, n_iter, log):
    # Determine the number of unique classes and the dimensionality of the input
    k = len(np.unique(y_train))
    d = x_train.shape[1]
    weight_len = k * d  # Total weight length for all classes
    weight = np.zeros(weight_len)  # Initialize weights to zero
    weight_sum = np.zeros(weight_len)  # Sum of weights for averaging
    weight_count = 0  # Count of updates to compute average
    tau = 1  # Learning rate

    train_accuracy = []  # List to store training accuracy
    test_accuracy = []  # List to store testing accuracy
    train_mistakes = []  # List to store training mistakes count
    test_mistakes = []  # List to store testing mistakes count

    # Epoch loop for the specified number of iterations
    for iter in range(n_iter):
        if log:
            print("epoch:" + str(iter))

        # Training phase
        train_count = 0  # Counter for training mistakes
        for data in range(x_train.shape[0]):
            y_hat = []  # List to store predictions for each class
            for y_t in range(k):
                # Calculate feature vector for the current class
                F_y_t = F_calc(x_train[data], y_t, weight_len, d)
                # Append the dot product of weights and feature vector to predictions
                y_hat.append(np.dot(weight, F_y_t))
            y_hat_max = np.argmax(y_hat)  # Get the predicted class with the highest score

            y_actual = int(y_train[data])  # Get the actual class label

            # If the prediction is incorrect, update weights
            if y_hat_max != y_actual:
                F_y_t = F_calc(x_train[data], y_actual, weight_len, d)  # Feature vector for actual class
                F_y_hat = F_calc(x_train[data], y_hat_max, weight_len, d)  # Feature vector for predicted class
                weight += tau * (F_y_t - F_y_hat)  # Update weights
                train_count += 1  # Increment mistake counter
            weight_sum += weight  # Accumulate weights for averaging
            weight_count += 1  # Increment weight count
        train_mistakes.append(train_count)  # Store training mistakes
        # Calculate and store training accuracy
        train_accuracy.append((x_train.shape[0] - train_count) * 100 / x_train.shape[0])

        # Testing phase
        test_count = 0  # Counter for testing mistakes
        weight = weight_sum / weight_count  # Average the weights
        for data in range(x_test.shape[0]):
            y_hat = []  # List to store predictions for each class
            for y_t in range(k):
                # Calculate feature vector for the current class
                F = F_calc(x_test[data], y_t, weight_len, d)
                # Append the dot product of weights and feature vector to predictions
                y_hat.append(np.dot(weight, F))
            y_hat_max = np.argmax(y_hat)  # Get the predicted class with the highest score

            y_actual = int(y_test[data])  # Get the actual class label
            # Count mistakes in predictions
            if y_hat_max != y_actual:
                test_count += 1
        test_mistakes.append(test_count)  # Store testing mistakes
        # Calculate and store testing accuracy
        test_accuracy.append((x_test.shape[0] - test_count) * 100 / x_test.shape[0])

    if log:
        # Saving the results to a CSV file
        with open("average_perceptron_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Training Accuracy", "Testing Accuracy"])  # Write header
            for row in zip(train_accuracy, test_accuracy):  # Write each accuracy pair
                writer.writerow(row)

        # Plotting accuracies
        plt.figure()
        plt.plot(train_accuracy, color='blue', label='Training Data Accuracy')
        plt.plot(test_accuracy, color='red', label='Testing Data Accuracy')
        plt.xlabel('Number of Iterations')  # Set x-axis label
        plt.ylabel('Accuracy')  # Set y-axis label
        plt.title('Accuracy vs Number of Iterations')  # Set title
        plt.legend(loc='lower right')  # Add legend
        plt.show()  # Display plot

    return weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes  # Return weights and accuracy metrics

# Function to generate generalized curves for a standard perceptron
def generate_generalized_curves(x_train, y_train, x_test, y_test):
    samples_list = []  # List to store sample sizes
    p_train_accuracy = []  # List to store training accuracy for each sample size
    p_test_accuracy = []  # List to store testing accuracy for each sample size
    p_train_mistakes = []  # List to store training mistakes count for each sample size
    p_test_mistakes = []  # List to store testing mistakes count for each sample size
    n_iter = 20  # Number of iterations for training
    log = False  # Logging flag to control printing of logs

    # Loop to generate curves for different sample sizes
    for samples in range(100, 60001, 100):  # Sample sizes from 100 to 60,000, incrementing by 100
        samples_list.append(samples)  # Add current sample size to the list
        print(samples)  # Print the current sample size for tracking
        # Create subsets of training data
        x_train_sub = x_train[:samples]
        y_train_sub = y_train[:samples]

        # Train the perceptron and get the results
        weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes = standard_perceptron(x_train_sub, y_train_sub, x_test, y_test, n_iter, log)
        # Store the last training and testing accuracy and mistakes
        p_train_accuracy.append(train_accuracy[-1])
        p_test_accuracy.append(test_accuracy[-1])
        p_train_mistakes.append(train_mistakes[-1])
        p_test_mistakes.append(test_mistakes[-1])

    # Create a DataFrame to store the results
    generalized_curves = pd.DataFrame({
        'Number of Samples': samples_list,
        'Perceptron Training Accuracy': p_train_accuracy,
        'Perceptron Testing Accuracy': p_test_accuracy,
        'Perceptron Training Mistakes': p_train_mistakes,
        'Perceptron Testing Mistakes': p_test_mistakes,
    })

    # Save the DataFrame to a CSV file
    generalized_curves.to_csv("generalized_curves.csv", index=False)
    return None  # End of function

# Function to generate generalized curves for a multiclass perceptron
def generate_multi_generalized_curves(x_train, y_train, x_test, y_test):
    samples_list = []  # List to store sample sizes
    p_train_accuracy = []  # List to store training accuracy for each sample size
    p_test_accuracy = []  # List to store testing accuracy for each sample size
    p_train_mistakes = []  # List to store training mistakes count for each sample size
    p_test_mistakes = []  # List to store testing mistakes count for each sample size
    n_iter = 20  # Number of iterations for training
    log = False  # Logging flag to control printing of logs

    # Loop to generate curves for different sample sizes
    for samples in range(100, 60001, 100):  # Sample sizes from 100 to 60,000, incrementing by 100
        samples_list.append(samples)  # Add current sample size to the list
        print(samples)  # Print the current sample size for tracking
        # Create subsets of training data
        x_train_sub = x_train[:samples]
        y_train_sub = y_train[:samples]

        # Train the multiclass perceptron and get the results
        weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes = multiclass_perceptron(x_train_sub, y_train_sub, x_test, y_test, n_iter, log)
        # Store the last training and testing accuracy and mistakes
        p_train_accuracy.append(train_accuracy[-1])
        p_test_accuracy.append(test_accuracy[-1])
        p_train_mistakes.append(train_mistakes[-1])
        p_test_mistakes.append(test_mistakes[-1])

    # Create a DataFrame to store the results
    generalized_curves = pd.DataFrame({
        'Number of Samples': samples_list,
        'Perceptron Training Accuracy': p_train_accuracy,
        'Perceptron Testing Accuracy': p_test_accuracy,
        'Perceptron Training Mistakes': p_train_mistakes,
        'Perceptron Testing Mistakes': p_test_mistakes
    })

    # Save the DataFrame to a CSV file
    generalized_curves.to_csv("multi_generalized_curves.csv", index=False)
    return None  # End of function

# Main code

# Question 5.1.a
n_iter = 50  # Set the number of training iterations for the models
log = False  # Set logging flag to False to suppress output during training

# Train a standard perceptron and a passive-aggressive model
weight_p, train_accuracy_p, test_accuracy_p, train_mistakes_p, test_mistakes_p = standard_perceptron(x_train, y_train, x_test, y_test, n_iter, log)
weight_pa, train_accuracy_pa, test_accuracy_pa, train_mistakes_pa, test_mistakes_pa = passive_aggressive(x_train, y_train, x_test, y_test, n_iter, log)

# Save a figure comparing the number of mistakes made by each model
save_figure("perceptron_PA_mistakes", train_mistakes_p, train_mistakes_pa, "Perceptron", "Passive Aggressive", "Number of Training Iterations", "Number of Mistakes")

# Question 5.1.b
n_iter = 20  # Set the number of training iterations for the models
log = False  # Set logging flag to False to suppress output during training

# Train a standard perceptron and a passive-aggressive model
weight_p, train_accuracy_p, test_accuracy_p, train_mistakes_p, test_mistakes_p = standard_perceptron(x_train, y_train, x_test, y_test, n_iter, log)
weight_pa, train_accuracy_pa, test_accuracy_pa, train_mistakes_pa, test_mistakes_pa = passive_aggressive(x_train, y_train, x_test, y_test, n_iter, log)

# Create subplots to visualize training and testing accuracies for both models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First plot for the standard perceptron
ax1.plot(train_accuracy_p, color='blue', label="Training Accuracy")  # Plot training accuracy
ax1.plot(test_accuracy_p, color='red', label="Testing Accuracy")  # Plot testing accuracy
ax1.set_xlabel("Number of Training Iterations")  # Set x-axis label
ax1.set_ylabel("Accuracy")  # Set y-axis label
ax1.set_title("Perceptron")  # Set plot title
ax1.legend(loc='lower right')  # Add legend to the plot
ax1.set_xlim(0, 20)  # Set x-axis limits
ax1.set_xticks(range(0, 21, 2))  # Set x-axis ticks

# Second plot for the passive-aggressive model
ax2.plot(train_accuracy_pa, color='blue', label="Training Accuracy")  # Plot training accuracy
ax2.plot(test_accuracy_pa, color='red', label="Testing Accuracy")  # Plot testing accuracy
ax2.set_xlabel("Number of Training Iterations")  # Set x-axis label
ax2.set_ylabel("Accuracy")  # Set y-axis label
ax2.set_title("Passive Aggressive")  # Set plot title
ax2.legend(loc='lower right')  # Add legend to the plot
ax2.set_xlim(0, 20)  # Set x-axis limits
ax2.set_xticks(range(0, 21, 2))  # Set x-axis ticks

# Adjust the layout to prevent overlap and show the plots
plt.tight_layout()
file_full_name = f"Comparison_P_PA_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"  # Create a filename with a timestamp
plt.savefig(file_full_name)  # Save the figure as a PNG file
plt.show()  # Display the plot

# Question 5.1.c
n_iter = 20  # Set the number of training iterations for the models
log = False  # Set logging flag to False to suppress output during training

# Train a standard perceptron and an average perceptron model
weight_p, train_accuracy_p, test_accuracy_p, train_mistakes_p, test_mistakes_p = standard_perceptron(x_train, y_train, x_test, y_test, n_iter, log)
weight_pa, train_accuracy_pa, test_accuracy_pa, train_mistakes_pa, test_mistakes_pa = average_perceptron(x_train, y_train, x_test, y_test, n_iter, log)

# Create subplots to visualize training and testing accuracies for both models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First plot for the standard perceptron
ax1.plot(train_accuracy_p, color='blue', label="Training Accuracy")  # Plot training accuracy
ax1.plot(test_accuracy_p, color='red', label="Testing Accuracy")  # Plot testing accuracy
ax1.set_xlabel("Number of Training Iterations")  # Set x-axis label
ax1.set_ylabel("Accuracy")  # Set y-axis label
ax1.set_title("Perceptron")  # Set plot title
ax1.legend(loc='lower right')  # Add legend to the plot
ax1.set_xlim(0, 20)  # Set x-axis limits
ax1.set_xticks(range(0, 21, 2))  # Set x-axis ticks

# Second plot for the average perceptron
ax2.plot(train_accuracy_pa, color='blue', label="Training Accuracy")  # Plot training accuracy
ax2.plot(test_accuracy_pa, color='red', label="Testing Accuracy")  # Plot testing accuracy
ax2.set_xlabel("Number of Training Iterations")  # Set x-axis label
ax2.set_ylabel("Accuracy")  # Set y-axis label
ax2.set_title("Average Perceptron")  # Set plot title
ax2.legend(loc='lower right')  # Add legend to the plot
ax2.set_xlim(0, 20)  # Set x-axis limits
ax2.set_xticks(range(0, 21, 2))  # Set x-axis ticks

# Adjust the layout to prevent overlap and show the plots
plt.tight_layout()
file_full_name = f"Comparison_P_AP_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"  # Create a filename with a timestamp
plt.savefig(file_full_name)  # Save the figure as a PNG file
plt.show()  # Display the plot

# Plot testing accuracy comparison between the standard perceptron and average perceptron
plt.figure()
plt.plot(test_accuracy_p, color='blue', label="Perceptron")  # Plot testing accuracy for standard perceptron
plt.plot(test_accuracy_pa, color='red', label="Average Perceptron")  # Plot testing accuracy for average perceptron
plt.xlabel('Number of Training Samples')  # Set x-axis label
plt.ylabel('Test Accuracy')  # Set y-axis label
plt.title('Test Accuracy vs Number of Training Samples')  # Set plot title
plt.legend(loc='lower right')  # Add legend to the plot
file_full_name = f"Comparison_testaccuracy_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"  # Create a filename with a timestamp
plt.savefig(file_full_name)  # Save the figure as a PNG file
plt.show()  # Display the plot

# Question 5.1.d
generate_generalized_curves(x_train, y_train, x_test, y_test)  # Generate generalized curves for the models

# Load the generalized curves data from CSV
df = pd.read_csv("generalized_curves.csv")

# Plot testing accuracy for perceptron and passive-aggressive models against number of samples
plt.figure()
plt.plot(df['Number of Samples'], df['Perceptron Testing Accuracy'], color='blue', label='Perceptron')  # Plot testing accuracy for perceptron
plt.plot(df['Number of Samples'], df['Passive Aggressive Testing Accuracy'], color='red', label='Passive Aggressive')  # Plot testing accuracy for passive aggressive
plt.xlabel('Number of Training Samples')  # Set x-axis label
plt.ylabel('Accuracy')  # Set y-axis label
plt.title('Accuracy vs Number of Training Samples')  # Set plot title
plt.legend(loc='lower right')  # Add legend to the plot
file_full_name = f"Generalized_Curve_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"  # Create a filename with a timestamp
plt.savefig(file_full_name)  # Save the figure as a PNG file
plt.show()  # Display the plot

# Question 5.2.a
n_iter = 50  # Set the number of training iterations for the models
log = True  # Set logging flag to True to enable output during training

# Train a multiclass perceptron and a multiclass passive-aggressive model
weight_p, train_accuracy_p, test_accuracy_p, train_mistakes_p, test_mistakes_p = multiclass_perceptron(x_train, y_train, x_test, y_test, n_iter, log)
weight_pa, train_accuracy_pa, test_accuracy_pa, train_mistakes_pa, test_mistakes_pa = multiclass_passive_aggressive(x_train, y_train, x_test, y_test, n_iter, log)

# Save a figure comparing the number of mistakes made by each model
save_figure("perceptron_PA_mistakes", train_mistakes_p, train_mistakes_pa, "Perceptron", "Passive Aggressive", "Number of Training Iterations", "Number of Mistakes")

# Question 5.2.b
n_iter = 20  # Set the number of training iterations for the models
log = False  # Set logging flag to False to suppress output during training

# Train a multiclass perceptron and a multiclass passive-aggressive model
weight_p, train_accuracy_p, test_accuracy_p, train_mistakes_p, test_mistakes_p = multiclass_perceptron(x_train, y_train, x_test, y_test, n_iter, log)
weight_pa, train_accuracy_pa, test_accuracy_pa, train_mistakes_pa, test_mistakes_pa = multiclass_passive_aggressive(x_train, y_train, x_test, y_test, n_iter, log)

# Create subplots to visualize training and testing accuracies for both models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First plot for the multiclass perceptron
ax1.plot(train_accuracy_p, color='blue', label="Training Accuracy")  # Plot training accuracy
ax1.plot(test_accuracy_p, color='red', label="Testing Accuracy")  # Plot testing accuracy
ax1.set_xlabel("Number of Training Iterations")  # Set x-axis label
ax1.set_ylabel("Accuracy")  # Set y-axis label
ax1.set_title("Perceptron")  # Set plot title
ax1.legend(loc='lower right')  # Add legend to the plot
ax1.set_xlim(0, 20)  # Set x-axis limits
ax1.set_xticks(range(0, 21, 2))  # Set x-axis ticks

# Second plot for the multiclass passive-aggressive model
ax2.plot(train_accuracy_pa, color='blue', label="Training Accuracy")  # Plot training accuracy
ax2.plot(test_accuracy_pa, color='red', label="Testing Accuracy")  # Plot testing accuracy
ax2.set_xlabel("Number of Training Iterations")  # Set x-axis label
ax2.set_ylabel("Accuracy")  # Set y-axis label
ax2.set_title("Passive Aggressive")  # Set plot title
ax2.legend(loc='lower right')  # Add legend to the plot
ax2.set_xlim(0, 20)  # Set x-axis limits
ax2.set_xticks(range(0, 21, 2))  # Set x-axis ticks

# Adjust the layout to prevent overlap and show the plots
plt.tight_layout()
file_full_name = f"Comparison_P_PA_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"  # Create a filename with a timestamp
plt.savefig(file_full_name)  # Save the figure as a PNG file
plt.show()  # Display the plot

# Question 5.2.c
n_iter = 20  # Set the number of training iterations for the models
log = True  # Set logging flag to True to enable output during training

# Train a multiclass perceptron and a multiclass average perceptron model
weight_p, train_accuracy_p, test_accuracy_p, train_mistakes_p, test_mistakes_p = multiclass_perceptron(x_train, y_train, x_test, y_test, n_iter, log)
weight_pa, train_accuracy_pa, test_accuracy_pa, train_mistakes_pa, test_mistakes_pa = multiclass_average_perceptron(x_train, y_train, x_test, y_test, n_iter, log)

# Create subplots to visualize training and testing accuracies for both models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First plot for the multiclass perceptron
ax1.plot(train_accuracy_p, color='blue', label="Training Accuracy")  # Plot training accuracy
ax1.plot(test_accuracy_p, color='red', label="Testing Accuracy")  # Plot testing accuracy
ax1.set_xlabel("Number of Training Iterations")  # Set x-axis label
ax1.set_ylabel("Accuracy")  # Set y-axis label
ax1.set_title("Perceptron")  # Set plot title
ax1.legend(loc='lower right')  # Add legend to the plot
ax1.set_xlim(0, 20)  # Set x-axis limits
ax1.set_xticks(range(0, 21, 2))  # Set x-axis ticks

# Second plot for the average perceptron
ax2.plot(train_accuracy_pa, color='blue', label="Training Accuracy")  # Plot training accuracy
ax2.plot(test_accuracy_pa, color='red', label="Testing Accuracy")  # Plot testing accuracy
ax2.set_xlabel("Number of Training Iterations")  # Set x-axis label
ax2.set_ylabel("Accuracy")  # Set y-axis label
ax2.set_title("Average Perceptron")  # Set plot title
ax2.legend(loc='lower right')  # Add legend to the plot
ax2.set_xlim(0, 20)  # Set x-axis limits
ax2.set_xticks(range(0, 21, 2))  # Set x-axis ticks

# Adjust the layout to prevent overlap and show the plots
plt.tight_layout()
file_full_name = f"Comparison_P_AP_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"  # Create a filename with a timestamp
plt.savefig(file_full_name)  # Save the figure as a PNG file
plt.show()  # Display the plot

# Plot testing accuracy comparison between the multiclass perceptron and average perceptron
plt.figure()
plt.plot(test_accuracy_p, color='blue', label="Perceptron")  # Plot testing accuracy for multiclass perceptron
plt.plot(test_accuracy_pa, color='red', label="Average Perceptron")  # Plot testing accuracy for average perceptron
plt.xlabel('Number of Training Samples')  # Set x-axis label
plt.ylabel('Test Accuracy')  # Set y-axis label
plt.title('Test Accuracy vs Number of Training Samples')  # Set plot title
plt.legend(loc='lower right')  # Add legend to the plot
file_full_name = f"Comparison_testaccuracy_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"  # Create a filename with a timestamp
plt.savefig(file_full_name)  # Save the figure as a PNG file
plt.show()  # Display the plot

# Question 5.2.d
generate_multi_generalized_curves(x_train, y_train, x_test, y_test)  # Generate generalized curves for the models

# Load the generalized curves data from CSV
df = pd.read_csv("multi_generalized_curves.csv")

# Plot testing accuracy for multiclass perceptron against the number of samples
plt.figure()
plt.plot(df['Number of Samples'], df['Perceptron Testing Accuracy'], color='blue', label='Perceptron')  # Plot testing accuracy for multiclass perceptron
plt.xlabel('Number of Training Samples')  # Set x-axis label
plt.ylabel('Accuracy')  # Set y-axis label
plt.title('Accuracy vs Number of Training Samples')  # Set plot title
plt.legend(loc='lower right')  # Add legend to the plot
file_full_name = f"Generalized_Curve_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"  # Create a filename with a timestamp
plt.savefig(file_full_name)  # Save the figure as a PNG file
plt.show()  # Display the plot

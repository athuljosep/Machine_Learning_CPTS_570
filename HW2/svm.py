# loading libraries
import numpy as np
from fashion_mnist.utils import mnist_reader
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

# Linear
# Load and scale the data
x_train, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Split training data into train and validation sets
x_Train, x_valid = x_train_scaled[:48000], x_train_scaled[48000:]
y_Train, y_valid = y_train[:48000], y_train[48000:]

# Array of regularization parameters (C values)
C_values = 10.0 ** np.arange(-4, 5)

# Initialize arrays to store accuracy scores and support vector counts
train_scores = np.zeros_like(C_values)
valid_scores = np.zeros_like(C_values)
test_scores = np.zeros_like(C_values)

# Train and evaluate models with different C values
for i, C in enumerate(C_values):
    clf = LinearSVC(C=C, multi_class='ovr', max_iter=3000)
    clf.fit(x_Train, y_Train)
    
    train_scores[i] = clf.score(x_Train, y_Train)
    valid_scores[i] = clf.score(x_valid, y_valid)
    test_scores[i] = clf.score(x_test_scaled, y_test)
    print(f"C={C}, Train Score={train_scores[i]:.4f}, Valid Score={valid_scores[i]:.4f}")

# Plot the accuracy scores
plt.figure(figsize=(8, 6))
plt.scatter(C_values, train_scores, label='Training Accuracy', color='black')
plt.scatter(C_values, test_scores, label='Test Accuracy', color='red')
plt.scatter(C_values, valid_scores, label='Validation Accuracy', color='blue')

# Set custom x-axis tick labels without changing to log scale
labels = [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', 
          r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$']
plt.xticks(C_values, labels)

# Adding labels and title
plt.xlabel('C', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs C', fontsize=14)

# Adding legend
plt.legend(loc='best')

# Generate a timestamp to append to the file name for uniqueness
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# Create the full file name with the timestamp
file_full_name = f"fig1a_{timestamp}.png"
plt.savefig(file_full_name)

# Display the plot
plt.show()

# Train on the full training set and compute confusion matrix on the test set
best_C = C_values[3]  # Use C[3] as specified in the original code
clf = LinearSVC(C=best_C, multi_class='ovr', max_iter=3000)
clf.fit(x_train_scaled, y_train)

y_pred = clf.predict(x_test_scaled)
test_accuracy = clf.score(x_test_scaled, y_test)

# Print test accuracy and confusion matrix
print(f"Test Accuracy with C={best_C}: {test_accuracy:.4f}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

################################################################################
# Polynomial
# Scale data
x_train_scaled = preprocessing.scale(x_train)
x_test_scaled = preprocessing.scale(x_test)

# Split training data into training and validation sets
x_train_split, x_valid_split = x_train_scaled[:48000], x_train_scaled[48000:]
y_train_split, y_valid_split = y_train[:48000], y_train[48000:]

# Polynomial degrees to test
degrees = np.array([2, 3, 4])

# Initialize score and support vector tracking arrays
train_scores = np.zeros_like(degrees, dtype=float)
valid_scores = np.zeros_like(degrees, dtype=float)
test_scores = np.zeros_like(degrees, dtype=float)
support_vectors_count = np.zeros((len(degrees), 10), dtype=int)

# Train and evaluate SVM models with different polynomial degrees
for idx, degree in enumerate(degrees):
    clf = SVC(C=0.01, kernel='poly', degree=degree, decision_function_shape='ovr', cache_size=4000)
    clf.fit(x_train_split, y_train_split)

    # Store scores for training, validation, and test sets
    train_scores[idx] = clf.score(x_train_split, y_train_split)
    valid_scores[idx] = clf.score(x_valid_split, y_valid_split)
    test_scores[idx] = clf.score(x_test_scaled, y_test)

    # Store the number of support vectors for each class
    support_vectors_count[idx] = clf.n_support_

# Plotting the data
plt.figure(figsize=(8, 4))
plt.scatter(train_scores, label='Training accuracy', color='black')
plt.scatter(valid_scores, label='Validation accuracy', color='blue')
plt.scatter(test_scores, label='Test accuracy', color='red')

# Setting x-axis and y-axis labels
plt.xlabel('Degree of polynomial', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# Adding legend
plt.legend(loc='best')

# Generate a timestamp to append to the file name for uniqueness
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# Create the full file name with the timestamp
file_full_name = f"fig1c1_{timestamp}.png"

plt.savefig(file_full_name)

# Display the plot
plt.show()

classes = range(1,11)  # Polynomial degrees

# Create subplots: 3 rows, 1 column
fig, axes = plt.subplots(1,3, figsize=(12, 4))  # Adjust the figsize as needed

# Plotting for P=2
axes[0].plot(classes, support_vectors_count[0], label='P=2', color='black')
axes[0].set_xlabel('Classes', fontsize=12)
axes[0].set_ylabel('Support Vectors', fontsize=12)
axes[0].legend(loc='best')

# Plotting for P=3
axes[1].plot(classes, support_vectors_count[1], label='P=3', color='blue')
axes[1].set_xlabel('Classes', fontsize=12)
axes[1].legend(loc='best')

# Plotting for P=4
axes[2].plot(classes, support_vectors_count[2], label='P=4', color='red')
axes[2].set_xlabel('Classes', fontsize=12)
axes[2].legend(loc='best')

# Add space between plots for readability
plt.tight_layout()

# Generate a timestamp to append to the file name for uniqueness
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# Create the full file name with the timestamp
file_full_name = f"fig1c2_{timestamp}.png"

plt.savefig(file_full_name)

# Display the plot
plt.show()

# Display results
for i, degree in enumerate(degrees):
    print(f"Degree: {degree}")
    print(f"Train Score: {train_scores[i]:.4f}")
    print(f"Validation Score: {valid_scores[i]:.4f}")
    print(f"Test Score: {test_scores[i]:.4f}")
    print(f"Number of Support Vectors: {support_vectors_count[i]}\n")

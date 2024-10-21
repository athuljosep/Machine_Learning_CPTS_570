# from sklearn import svm
# from sklearn import metrics
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix
from datetime import datetime


# # 1.a
# # Provided data
# C_values = range(1, 10)  # Linear C values
# training_accuracy =   [0.835, 0.855, 0.87, 0.875, 0.877, 0.841, 0.821, 0.819, 0.815]
# test_accuracy =       [0.819, 0.83, 0.832, 0.838, 0.836, 0.805, 0.788, 0.78, 0.778]
# validation_accuracy = [0.83, 0.845, 0.848, 0.847, 0.842, 0.81, 0.79, 0.793, 0.795]

# # Plotting the data
# plt.figure(figsize=(8, 6))
# plt.scatter(C_values, training_accuracy, label='Training Accuracy', color='black')
# plt.scatter(C_values, test_accuracy, label='Test Accuracy', color='red')
# plt.scatter(C_values, validation_accuracy, label='Validation Accuracy', color='blue')

# # Set custom x-axis tick labels without changing to log scale
# labels = [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', 
#           r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$']
# plt.xticks(C_values, labels)

# # Adding labels and title
# plt.xlabel('C', fontsize=12)
# plt.ylabel('Accuracy', fontsize=12)
# plt.title('Accuracy vs C', fontsize=14)

# # Adding legend
# plt.legend(loc='best')

# # Generate a timestamp to append to the file name for uniqueness
# timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# # Create the full file name with the timestamp
# file_full_name = f"fig1a_{timestamp}.png"

# plt.savefig(file_full_name)

# # Display the plot
# plt.show()

# 1 c

# # Data for plotting (adjust based on visual inspection of the provided figure)
# degree_of_polynomial = [2, 3, 4]  # Polynomial degrees
# training_accuracy = [0.73, 0.681, 0.61]  # Example values for training accuracy
# validation_accuracy = [0.726, 0.681, 0.596]  # Example values for validation accuracy
# test_accuracy = [0.72, 0.676, 0.599]  # Example values for test accuracy

# # Plotting the data
# plt.figure(figsize=(8, 4))
# plt.scatter(degree_of_polynomial, training_accuracy, label='Training accuracy', color='black')
# plt.scatter(degree_of_polynomial, validation_accuracy, label='Validation accuracy', color='blue')
# plt.scatter(degree_of_polynomial, test_accuracy, label='Test accuracy', color='red')

# # Setting x-axis and y-axis labels
# plt.xlabel('Degree of polynomial', fontsize=12)
# plt.ylabel('Accuracy', fontsize=12)

# # Adding legend
# plt.legend(loc='best')

# # Generate a timestamp to append to the file name for uniqueness
# timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# # Create the full file name with the timestamp
# file_full_name = f"fig1c1_{timestamp}.png"

# plt.savefig(file_full_name)

# # Display the plot
# plt.show()


with open('svm_results_3.pkl', 'rb') as f:
    sv_count_p_2_3,sv_count_p_3_3,sv_count_p_4_3,tr_acc_p_2_3,tr_acc_p_3_3,tr_acc_p_4_3,test_acc_p_2_3,test_acc_p_3_3,test_acc_p_4_3,val_acc_p_2_3,val_acc_p_3_3,val_acc_p_4_3 = pickle.load(f)

# Data for plotting (adjust based on visual inspection of the provided figure)
classes = range(1,11)  # Polynomial degrees

# Create subplots: 3 rows, 1 column
fig, axes = plt.subplots(1,3, figsize=(12, 4))  # Adjust the figsize as needed

# Plotting for P=2
axes[0].plot(classes, sv_count_p_2_3, label='P=2', color='black')
axes[0].set_xlabel('Classes', fontsize=12)
axes[0].set_ylabel('Support Vectors', fontsize=12)
axes[0].legend(loc='best')

# Plotting for P=3
axes[1].plot(classes, sv_count_p_3_3, label='P=3', color='blue')
axes[1].set_xlabel('Classes', fontsize=12)
axes[1].legend(loc='best')

# Plotting for P=4
axes[2].plot(classes, sv_count_p_4_3, label='P=4', color='red')
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


1+1

# # 2.0
# # Data from the image
# iterations = [1, 2, 3, 4, 5]  # Iteration numbers
# mistakes = [10345, 8388, 7718, 7260, 7068]  # Corresponding mistakes

# plt.figure()

# plt.plot(iterations, mistakes, marker='o', linestyle='-', linewidth=1.5, markersize=8, color = 'green')
# plt.title('Mistakes vs Iterations', fontsize=14)
# plt.xlabel('Iteration', fontsize=12)
# plt.ylabel('Mistakes', fontsize=12)

# # Adding grid and setting limits to make the plot clearer
# plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# # Generate a timestamp to append to the file name for uniqueness
# timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# # Create the full file name with the timestamp
# file_full_name = f"kernel_{timestamp}.png"

# plt.savefig(file_full_name)

# # Display the plot
# plt.show()


'''Author: Athul Jose P'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from fashion_mnist.utils import mnist_reader
x_train, y_train = mnist_reader.load_mnist('C:/Users/athul.p/Documents/GitHub/Machine_Learning_CPTS_570/HW1/fashion_mnist/data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('C:/Users/athul.p/Documents/GitHub/Machine_Learning_CPTS_570/HW1/fashion_mnist/data/fashion', kind='t10k')

def standard_perceptron(x_train, y_train, x_test, y_test, n_iter):
    weight = np.zeros(x_train.shape[1])
    tau = 1
    train_accuracy = []
    test_accuracy = []

    # epoch
    for iter in range(n_iter):
        print("epoch:" + str(iter))

        # training
        train_count = 0
        for data in range(x_train.shape[0]):
            y_hat = np.sign(np.dot(weight, x_train[data]))
            if(y_train[data] % 2) == 0:
                y_actual = 1.0
            else:
                y_actual = -1.0
            if y_hat != y_actual:
                weight += tau * y_actual * x_train[data]
                train_count += 1
        train_accuracy.append((x_train.shape[0]-train_count)*100 / x_train.shape[0])

        # testing
        test_count = 0
        for data in range(x_test.shape[0]):
            y_hat = np.sign(np.dot(weight, x_test[data]))
            if (y_test[data] % 2) == 0:
                y_actual = 1.0
            else:
                y_actual = -1.0
            if y_hat != y_actual:
                test_count += 1
        test_accuracy.append((x_test.shape[0] - test_count)*100 / x_test.shape[0])

    # saving the results
    with open("perceptron_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".csv", mode = 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Training Accuracy", "Testing Accuracy"])
        for row in zip(train_accuracy, test_accuracy):
            writer.writerow(row)

    # plotting accuracies
    plt.figure()
    plt.plot(train_accuracy, color='blue', label='Training Data Accuracy')
    plt.plot(test_accuracy, color='red', label='Testing Data Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Iterations')
    plt.legend(loc='lower right')
    plt.show()

    return weight

def passive_aggressive(x_train, y_train, x_test, y_test, n_iter):
    weight = np.zeros(x_train.shape[1])
    train_accuracy = []
    test_accuracy = []

    # epoch
    for iter in range(n_iter):
        print("epoch:" + str(iter))

        # training
        train_count = 0
        for data in range(x_train.shape[0]):
            y_hat = np.sign(np.dot(weight, x_train[data]))
            if(y_train[data] % 2) == 0:
                y_actual = 1.0
            else:
                y_actual = -1.0
            if y_hat != y_actual:
                tau = (1-(y_actual * np.dot(weight, x_train[data])))/(np.square(np.linalg.norm(x_train[data])))
                weight += tau * y_actual * x_train[data]
                train_count += 1
        train_accuracy.append((x_train.shape[0]-train_count)*100 / x_train.shape[0])

        # testing
        test_count = 0
        for data in range(x_test.shape[0]):
            y_hat = np.sign(np.dot(weight, x_test[data]))
            if (y_test[data] % 2) == 0:
                y_actual = 1.0
            else:
                y_actual = -1.0
            if y_hat != y_actual:
                test_count += 1
        test_accuracy.append((x_test.shape[0] - test_count)*100 / x_test.shape[0])

    # saving the results
    with open("pa_algorithm_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".csv", mode = 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Training Accuracy", "Testing Accuracy"])
        for row in zip(train_accuracy, test_accuracy):
            writer.writerow(row)

    # plotting accuracies
    plt.figure()
    plt.plot(train_accuracy, color='blue', label='Training Data Accuracy')
    plt.plot(test_accuracy, color='red', label='Testing Data Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Iterations')
    plt.legend(loc='lower right')
    plt.show()

    return weight

def average_perceptron(x_train, y_train, x_test, y_test, n_iter):
    weight = np.zeros(x_train.shape[1])
    tau = 1
    train_accuracy = []
    test_accuracy = []

    # epoch
    for iter in range(n_iter):
        print("epoch:" + str(iter))

        # training
        train_count = 0
        for data in range(x_train.shape[0]):
            y_hat = np.sign(np.dot(weight, x_train[data]))
            if(y_train[data] % 2) == 0:
                y_actual = 1.0
            else:
                y_actual = -1.0
            if y_hat != y_actual:
                weight_sum += weights
                weight += tau * y_actual * x_train[data]
                train_count += 1
        train_accuracy.append((x_train.shape[0]-train_count)*100 / x_train.shape[0])

        # testing
        test_count = 0
        for data in range(x_test.shape[0]):
            y_hat = np.sign(np.dot(weight, x_test[data]))
            if (y_test[data] % 2) == 0:
                y_actual = 1.0
            else:
                y_actual = -1.0
            if y_hat != y_actual:
                test_count += 1
        test_accuracy.append((x_test.shape[0] - test_count)*100 / x_test.shape[0])

    # saving the results
    with open("average_perceptron"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".csv", mode = 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Training Accuracy", "Testing Accuracy"])
        for row in zip(train_accuracy, test_accuracy):
            writer.writerow(row)

    # plotting accuracies
    plt.figure()
    plt.plot(train_accuracy, color='blue', label='Training Data Accuracy')
    plt.plot(test_accuracy, color='red', label='Testing Data Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Iterations')
    plt.legend(loc='lower right')
    plt.show()

    return weight

# main
n_iter = 50
# w = standard_perceptron(x_train, y_train, x_test, y_test, n_iter)
# w = passive_aggressive(x_train, y_train, x_test, y_test, n_iter)
w = average_perceptron(x_train, y_train, x_test, y_test, n_iter)

'''Author: Athul Jose P'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from fashion_mnist.utils import mnist_reader
x_train, y_train = mnist_reader.load_mnist('C:/Users/athul.p/Documents/GitHub/Machine_Learning_CPTS_570/HW1/fashion_mnist/data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('C:/Users/athul.p/Documents/GitHub/Machine_Learning_CPTS_570/HW1/fashion_mnist/data/fashion', kind='t10k')

def standard_perceptron(x_train, y_train, x_test, y_test, n_iter, log):
    weight = np.zeros(x_train.shape[1])
    tau = 1
    train_accuracy = []
    test_accuracy = []
    train_mistakes = []
    test_mistakes = []

    # epoch
    for iter in range(n_iter):
        # print("epoch:" + str(iter))

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
        train_mistakes.append(train_count)
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
        test_mistakes.append(test_count)
        test_accuracy.append((x_test.shape[0] - test_count)*100 / x_test.shape[0])

    if log:
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

    return weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes

def passive_aggressive(x_train, y_train, x_test, y_test, n_iter, log):
    weight = np.zeros(x_train.shape[1])
    train_accuracy = []
    test_accuracy = []
    train_mistakes = []
    test_mistakes = []

    # epoch
    for iter in range(n_iter):
        # print("epoch:" + str(iter))

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
        train_mistakes.append(train_count)
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
        test_mistakes.append(test_count)
        test_accuracy.append((x_test.shape[0] - test_count)*100 / x_test.shape[0])

    if log:
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

    return weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes

def average_perceptron(x_train, y_train, x_test, y_test, n_iter, log):
    weight = np.zeros(x_train.shape[1])
    weight_sum = np.zeros(x_train.shape[1])
    weight_count = 0
    tau = 1
    train_accuracy = []
    test_accuracy = []
    train_mistakes = []
    test_mistakes = []

    # epoch
    for iter in range(n_iter):
        # print("epoch:" + str(iter))

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
            weight_sum += weight
            weight_count += 1
        train_mistakes.append(train_count)
        train_accuracy.append((x_train.shape[0]-train_count)*100 / x_train.shape[0])

        # testing
        test_count = 0
        weight = weight_sum / weight_count
        for data in range(x_test.shape[0]):
            y_hat = np.sign(np.dot(weight, x_test[data]))
            if (y_test[data] % 2) == 0:
                y_actual = 1.0
            else:
                y_actual = -1.0
            if y_hat != y_actual:
                test_count += 1
        test_mistakes.append(test_count)
        test_accuracy.append((x_test.shape[0] - test_count)*100 / x_test.shape[0])

    if log:
        # saving the results
        with open("average_perceptron_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".csv", mode = 'w', newline='') as f:
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

    return weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes

# main
n_iter = 50
# w = standard_perceptron(x_train, y_train, x_test, y_test, n_iter)
# w = passive_aggressive(x_train, y_train, x_test, y_test, n_iter)
# w = average_perceptron(x_train, y_train, x_test, y_test, n_iter)

def generate_generalized_curves(x_train, y_train, x_test, y_test):
    samples_list = []
    p_train_accuracy = []
    p_test_accuracy = []
    p_train_mistakes = []
    p_test_mistakes = []
    pa_train_accuracy = []
    pa_test_accuracy = []
    pa_train_mistakes = []
    pa_test_mistakes = []
    n_iter = 20
    log = False

    # Generalized curves
    for samples in range(100,60001,100):
        samples_list.append(samples)
        print(samples)
        x_train_sub = x_train[:samples]
        y_train_sub = y_train[:samples]

        weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes = standard_perceptron(x_train_sub, y_train_sub, x_test, y_test, n_iter, log)
        p_train_accuracy.append(train_accuracy[-1])
        p_test_accuracy.append(test_accuracy[-1])
        p_train_mistakes.append(train_mistakes[-1])
        p_test_mistakes.append(test_mistakes[-1])
        weight, train_accuracy, test_accuracy, train_mistakes, test_mistakes = passive_aggressive(x_train_sub, y_train_sub,x_test, y_test, n_iter, log)
        pa_train_accuracy.append(train_accuracy[-1])
        pa_test_accuracy.append(test_accuracy[-1])
        pa_train_mistakes.append(train_mistakes[-1])
        pa_test_mistakes.append(test_mistakes[-1])

    generalized_curves = pd.DataFrame({
        'Number of Samples': samples_list,
        'Perceptron Training Accuracy': p_train_accuracy,
        'Perceptron Testing Accuracy': p_test_accuracy,
        'Perceptron Training Mistakes': p_train_mistakes,
        'Perceptron Testing Mistakes': p_test_mistakes,
        'Passive Aggressive Training Accuracy': pa_train_accuracy,
        'Passive Aggressive Testing Accuracy': pa_test_accuracy,
        'Passive Aggressive Training Mistakes': pa_train_mistakes,
        'Passive Aggressive Testing Mistakes': pa_test_mistakes
    })

    generalized_curves.to_csv("generalized_curves.csv", index=False)
    return None

# generate_generalized_curves(x_train, y_train, x_test, y_test)

df = pd.read_csv("generalized_curves.csv")

plt.figure()
plt.plot(df['Number of Samples'],df['Perceptron Testing Accuracy'], color='blue', label='Perceptron')
plt.plot(df['Number of Samples'],df['Passive Aggressive Testing Accuracy'],  color='red', label='Passive Aggressive')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Training Samples')
plt.legend(loc='lower right')
plt.show()

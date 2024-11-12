## Overview
This project involves generating results for various questions using defined functions and a structured script. The required packages and instructions to run the code are listed below.

## Required Packages
Ensure you have the following Python packages installed before running the code:
- `numpy`
- `pandas`
- `matplotlib`
- `csv`
- `datetime`
- `fashion_mnist`
- `sklearn`
- `ucimlrepo`

## Contents

1. **svm.py**
   - Implements both linear and polynomial SVM using scikit-learn.
   - Handles model training and evaluation for both kernel types.

2. **perceptron_kernalized.py**
   - Contains the implementation of the Kernelized Perceptron.
   - Demonstrates how the kernelized perceptron model performs on a given dataset.

3. **decision_tree.py**
   - Includes code for building a Decision Tree model using scikit-learn.
   - Performs training and evaluation on the provided dataset.

## Instructions to Run

1. **Add the Dataset**
   - Ensure that the dataset is correctly placed in the appropriate location specified by the scripts.
   - Without the dataset, the scripts will not run properly.

2. **Script Structure**
   - Each script defines all necessary functions at the beginning.
   - The main execution code is placed towards the end of the script for easy readability and customization.

3. **How to Run**
   - Ensure Python and required libraries (like scikit-learn) are installed.
   - Run each script from the terminal or your preferred IDE. Example:

`python svm.py`
`python perceptron_kernalized.py`
`python decision_tree.py`

## Notes
- Ensure that the required datasets (if any) are available in the correct directories before running the code.
- The results will be displayed via terminal or saved as outputs in the specified location, depending on the question.
# loading packages
import numpy as np

# create a vocabulary
def create_vocabulary(train_data, stop_words):
    sentences_list = train_data.split('\n')
    unique_words = set()

    # unique words set
    for sentence in sentences_list:
        words = sentence.split()
        unique_words.update(words)

    # stop words set
    stop_words_set = set(stop_words.split('\n'))

    # vocabulary words
    vocabulary_words = unique_words - stop_words_set
    vocabulary = sorted(vocabulary_words)
    return vocabulary

# generate feature vectors
def generate_feature_vectors(data, vocabulary):
    samples = data.split('\n')
    num_samples = len(samples)
    vocab_size = len(vocabulary)

    feature_matrix = np.zeros((num_samples, vocab_size), dtype=int)
    vocab_index = {word: idx for idx, word in enumerate(vocabulary)}

    # feature matrix
    for i, sample in enumerate(samples):
        indices = [vocab_index[word] for word in sample.split() if word in vocab_index]
        feature_matrix[i, indices] = 1  # Set presence of words in the vocabulary to 1

    return feature_matrix

# train Naive Bayes Classifier
def naive_bayes_classifier(feature_vectors, labels):
    label_list = labels.split('\n')
    num_samples = len(label_list)
    class_1_indices = [i for i, label in enumerate(label_list) if label == '1']
    class_0_indices = [i for i, label in enumerate(label_list) if label == '0']

    # Calculate prior probabilities for each class
    prior_prob_class_1 = len(class_1_indices) / num_samples
    prior_prob_class_0 = len(class_0_indices) / num_samples

    conditional_probabilities = {}
    num_features = feature_vectors.shape[1]
    for feature_index in range(num_features):
        feature_column = feature_vectors[:, feature_index]

        # feature values for each class
        feature_values_class_0 = feature_column[class_0_indices]
        feature_values_class_1 = feature_column[class_1_indices]

        # count occurrences of 0 and 1
        count_0_given_class_0 = (feature_values_class_0 == 0).sum()
        count_1_given_class_0 = (feature_values_class_0 == 1).sum()
        count_0_given_class_1 = (feature_values_class_1 == 0).sum()
        count_1_given_class_1 = (feature_values_class_1 == 1).sum()

        # Laplace smoothing
        prob_0_given_class_0 = (count_0_given_class_0 + 1) / (len(class_0_indices) + 2)
        prob_1_given_class_0 = (count_1_given_class_0 + 1) / (len(class_0_indices) + 2)
        prob_0_given_class_1 = (count_0_given_class_1 + 1) / (len(class_1_indices) + 2)
        prob_1_given_class_1 = (count_1_given_class_1 + 1) / (len(class_1_indices) + 2)

        conditional_probabilities[feature_index] = {
            'P_0_given_0': prob_0_given_class_0,
            'P_1_given_0': prob_1_given_class_0,
            'P_0_given_1': prob_0_given_class_1,
            'P_1_given_1': prob_1_given_class_1
        }

    return prior_prob_class_0, prior_prob_class_1, conditional_probabilities

# make prediction
def predict(feature_vectors, labels, prior_prob_0, prior_prob_1, conditional_probs):
    label_list = labels.strip().split('\n')
    num_samples, num_features = feature_vectors.shape
    num_errors = 0

    for i in range(num_samples):
        prob_0 = prior_prob_0
        prob_1 = prior_prob_1

        # update probabilities
        for j in range(num_features):
            if feature_vectors[i, j] == 1:
                prob_0 *= conditional_probs[j]['P_1_given_0']
                prob_1 *= conditional_probs[j]['P_1_given_1']
            else:
                prob_0 *= conditional_probs[j]['P_0_given_0']
                prob_1 *= conditional_probs[j]['P_0_given_1']

        # predict class
        predicted_label = 0 if prob_0 > prob_1 else 1
        actual_label = int(label_list[i])
        if predicted_label != actual_label:
            num_errors += 1

    # accuracy calculation
    accuracy = 100 * (num_samples - num_errors) / num_samples
    return accuracy

## main code
# file paths of data
file_paths = {
    'train_data': 'fortune-cookie-data/traindata.txt',
    'train_labels': 'fortune-cookie-data/trainlabels.txt',
    'test_data': 'fortune-cookie-data/testdata.txt',
    'test_labels': 'fortune-cookie-data/testlabels.txt',
    'stop_words_list': 'fortune-cookie-data/stoplist.txt'
}

# load data
data = {name: open(path, 'r').read() for name, path in file_paths.items()}
train_data = data['train_data']
train_labels = data['train_labels']
test_data = data['test_data']
test_labels = data['test_labels']
stop_words_list = data['stop_words_list']

# Create the vocabulary using the function
vocabulary = create_vocabulary(train_data, stop_words_list)

# generate feature vectors
train_feature_vectors = generate_feature_vectors(train_data, vocabulary)
test_feature_vectors = generate_feature_vectors(test_data, vocabulary)

# train Naive Bayes Classifier
prior_prob_0,prior_prob_1,conditional_prob= naive_bayes_classifier(train_feature_vectors,train_labels)

# calculate training and testing accuracy
train_accuracy = predict(train_feature_vectors,train_labels,prior_prob_0,prior_prob_1,conditional_prob)
test_accuracy = predict(test_feature_vectors,test_labels,prior_prob_0,prior_prob_1,conditional_prob)

# print accuracies
print(f"Training Accuracy= {train_accuracy}")
print(f"Testing Accuracy= {test_accuracy}")

# writing output
outfile = open("output.txt", "w")
print("Training Accuracy:" + str(train_accuracy) + "\n", file=outfile)
print("Testing Accuracy:" + str(test_accuracy) + "\n", file=outfile)
outfile.close()

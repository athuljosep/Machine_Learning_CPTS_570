import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Function to process data 
def process_and_prepare(data_path, percentage, batch_size, shuffle):
    # preprocessing
    data = pd.read_csv(data_path)
    samples = process_and_sample_data(data, percentage)
    features, labels = split_features_and_labels(samples)
    features_tensor, labels_tensor = convert_to_tensors(features, labels)
    
    # Create DataLoader
    data_loader = DataLoader(
        TensorDataset(features_tensor, labels_tensor),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_loader

def process_and_sample_data(data, sample_percentage):
    # Extract labels and calculate dataset size
    labels = data.iloc[:, 0]
    total_samples = int(sample_percentage / 100 * len(labels))

    # Determine unique labels and sample size per label
    unique_labels = labels.unique()
    num_classes = len(unique_labels)
    samples_per_class = total_samples // num_classes

    # Collect sampled data for each label
    sampled_data_list = []
    for label in unique_labels:
        label_subset = data[labels == label]
        samples_to_draw = min(samples_per_class, len(label_subset))
        sampled_subset = label_subset.sample(n=samples_to_draw, random_state=42)
        sampled_data_list.append(sampled_subset)

    # Combine all samples and shuffle
    balanced_sampled_data = pd.concat(sampled_data_list).reset_index(drop=True)
    shuffled_data = balanced_sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return shuffled_data

def split_features_and_labels(samples):
    labels = samples.iloc[:, 0].values
    features = samples.iloc[:, 1:].values
    return features, labels

def convert_to_tensors(features, labels):
    features_tensor = torch.tensor(features, dtype=torch.float32) / 255.0
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Reshape features for CNN input
    features_tensor = features_tensor.view(-1, 1, 28, 28)
    return features_tensor, labels_tensor

# CNN Class
class CNN_(nn.Module):
    def __init__(self):
        super(CNN_, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, device, train_loader, optimizer, loss_function):
    model.train()  # Set the model to training mode
    total_samples = 0
    correct_predictions = 0
    cumulative_loss = 0.0

    for features, labels in train_loader:
        # Move data to the specified device
        features, labels = features.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(features)
        loss = loss_function(predictions, labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate loss and correct predictions
        cumulative_loss += loss.item() * features.size(0)
        _, predicted_labels = torch.max(predictions.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted_labels == labels).sum().item()

    # Calculate average loss and accuracy
    average_loss = cumulative_loss / total_samples
    accuracy = 100 * correct_predictions / total_samples

    return average_loss, accuracy

def evaluate_model(model, device, test_loader, loss_function):
    model.eval()  # Set the model to evaluation mode
    total_samples = 0
    correct_predictions = 0
    cumulative_loss = 0.0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for features, labels in test_loader:
            # Move data to the specified device
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            predictions = model(features)
            loss = loss_function(predictions, labels)

            # Accumulate loss and correct predictions
            cumulative_loss += loss.item() * features.size(0)
            _, predicted_labels = torch.max(predictions.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()

    # Calculate average loss and accuracy
    average_loss = cumulative_loss / total_samples
    accuracy = 100 * correct_predictions / total_samples

    return average_loss, accuracy

# main code

# File paths
train_path = "fashion-mnist_train.csv"
test_path = "fashion-mnist_test.csv"

# Parameters
train_percent = 50
test_percent = 10
batch_size = 64
num_epochs = 20

# Create DataLoaders
train_loader = process_and_prepare(train_path, train_percent, batch_size, shuffle=True)
test_loader = process_and_prepare(test_path, test_percent, batch_size, shuffle=False)

# Initialize model, loss function, optimizer, and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_accuracies = []
test_accuracies = []
# Training loop
for epoch in range(1, num_epochs + 1):
    train_loss, train_accuracy = train_model(model, device, train_loader, optimizer, criterion)
    test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    print(f'Training Accuracy: {train_accuracy:.2f}%, Testing Accuracy: {test_accuracy:.2f}%\n')

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Testing Accuracy')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.show()
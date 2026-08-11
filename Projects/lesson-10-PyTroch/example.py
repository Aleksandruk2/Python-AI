import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ці дані так само вже зберігаються в середі google colab
train_data_path = 'mnist_train_small.csv'
test_data_path = 'mnist_test.csv'

# В данному випадку хедера в таблицях нема. Значення категоріі зберігається
# в першій колонці.
train_data = pd.read_csv('mnist_train_small.csv', header=None)
test_data = pd.read_csv('mnist_test.csv', header=None)

# Розділяємо наші ознаки від вчителя
X_train = train_data.drop(0, axis=1).values
y_train = train_data[0].values

X_test = test_data.drop(0, axis=1).values
y_test = test_data[0].values

# Visualize some samples
plt.figure(figsize=(10, 5))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

import torch

# Convert MNIST data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32) / 255.0  # Normalize pixel values to [0, 1]
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32) / 255.0
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

from torch import nn

# Update the model for 10-class classification
class MLPClassifierMNIST(nn.Module):
    def __init__(self):
        super(MLPClassifierMNIST, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),  # Input size is 28x28 pixels
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # Output size is 10 classes
        )

    def forward(self, x):
        return self.model(x)

from torch import optim

# Instantiate the model
model = MLPClassifierMNIST()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# наконуємо тренування нашої моделі
# Training loop
epochs = 50 # Дуже добре проводить розпізнавання
batch_size = 64

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_loss /= 2

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        predictions = torch.argmax(test_outputs, dim=1)
        accuracy = (predictions == y_test_tensor).float().mean()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy.item():.4f}")


import matplotlib.pyplot as plt

# Select random samples from the test set
num_samples = 9
indices = np.random.choice(len(X_test_tensor), num_samples, replace=False)
sample_images = X_test_tensor[indices]
sample_labels = y_test_tensor[indices]

# Get model predictions
model.eval()
with torch.no_grad():
    predictions = torch.argmax(model(sample_images), dim=1)

# Plot the images with predicted and true labels
plt.figure(figsize=(10, 5))
for i in range(num_samples):
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {sample_labels[i].item()}, Pred: {predictions[i].item()}")
    plt.axis('off')

plt.tight_layout()
plt.show()
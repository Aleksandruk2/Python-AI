import os
import random

import kagglehub
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from tqdm import tqdm

# завантаження датасету

path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia"
)

print("Датасет завантажено:")
print(path)

data_path = os.path.join(path, "chest_xray")

train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

print("\nTrain:", train_path)
print("Test:", test_path)

# завдання 1

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    train_path,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    test_path,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("\nКласи:")
print(train_dataset.classes)

print("\nКількість тренувальних зображень:",
      len(train_dataset))

print("Кількість тестових зображень:",
      len(test_dataset))

images, labels = next(iter(train_loader))

print("\nФорма batch зображень:", images.shape)
print("Форма labels:", labels.shape)

# завдання 2

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

print(model)

for param in model.parameters():
    param.requires_grad = False


print("до",model.fc)
model.fc = nn.Linear(512, 2)
print("після",model.fc)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.0001
)

trainable_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)

print("Trainable parameters:")
for name in trainable_params:
    print(name)

epochs = 5

train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.4f}"
    )

# Loss
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, epochs + 1),
    train_losses,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.xticks(range(1, epochs + 1))
plt.grid(True)
plt.show()

# Accuracy
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, epochs + 1),
    train_accuracies,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.xticks(range(1, epochs + 1))
plt.grid(True)
plt.show()

# завдання 3

model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

print("===== MODEL METRICS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        zero_division=0
    )
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(
    ax=ax,
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.show()


MODEL_PATH = "pneumonia_resnet18.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(f"Model saved to: {MODEL_PATH}")

print(
    "File exists:",
    os.path.exists(MODEL_PATH)
)

if os.path.exists(MODEL_PATH):
    print(
        "File size:",
        os.path.getsize(MODEL_PATH) / (1024 * 1024),
        "MB"
    )

# завдання 4

model.eval()

num_images = 8

random_indices = random.sample(
    range(len(test_dataset)),
    num_images
)

plt.figure(figsize=(16, 8))

for i, index in enumerate(random_indices):
    image, label = test_dataset[index]
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()

    true_class = test_dataset.classes[label]
    predicted_class_name = test_dataset.classes[predicted_class]
    image_for_plot = image.permute(1, 2, 0).numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_for_plot = image_for_plot * std + mean

    image_for_plot = np.clip(
        image_for_plot,
        0,
        1
    )

    plt.subplot(2, 4, i + 1)
    plt.imshow(image_for_plot)
    plt.title(
        f"True: {true_class}\n"
        f"Pred: {predicted_class_name}"
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "pneumonia_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
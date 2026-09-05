import os
import time
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



# Налаштування
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Пристрій:", device)


# Підготовка MNIST
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# Створення DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


print("Кількість навчальних зображень:", len(train_dataset))
print("Кількість тестових зображень:", len(test_dataset))

print("Кількість навчальних батчів:", len(train_loader))
print("Кількість тестових батчів:", len(test_loader))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = CNN().to(device)
print(model)
model_path = "mnist_cnn.pth"

if os.path.exists(model_path):
    print("Знайдено збережену модель.")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Збереженої моделі немає. Починаємо навчання...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    start_training_time = time.time()

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        print(
            f"Loss: {average_loss:.4f}"
        )

    training_time = time.time() - start_training_time
    print(f"\nЧас навчання: {training_time:.2f} секунд")

    # Зберігаємо навчену модель
    torch.save(model.state_dict(), model_path)
    print(f"Модель збережено у {model_path}")

model.eval()

correct = 0
total = 0

start_test_time = time.time()

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_time = time.time() - start_test_time

accuracy = 100 * correct / total

print(f"\nТочність на тестовій вибірці: {accuracy:.2f}%")
print(f"Час перевірки тестової вибірки: {test_time:.2f} секунд")


# Приклад розпізнавання зображень
indices = random.sample(range(len(test_dataset)), 9)

plt.figure(figsize=(8, 8))

for i, index in enumerate(indices):
    image, label = test_dataset[index]
    image_for_model = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_for_model)
        predicted_digit = torch.argmax(output, dim=1).item()

    plt.subplot(3, 3, i + 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(
        f"Правильна: {label}\n"
        f"Модель: {predicted_digit}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

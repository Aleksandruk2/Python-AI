import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm


# Завдання 1-2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Замість того щоб загружати дані вручну як ми робили до цього, використаємо
# спеціальний клас із бібліотеки torchvision
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)

        # Вирівнюємо тензор для повнозв'язного шару
        x = x.view(-1, 64 * 7 * 7) # Bx64x7x7 -> Bx3136
        # Застосовуємо повнозв'язний шар
        x = self.fc1(x)
        return x

# Створюємо екземпляр моделі
model = SimpleCNN()

# Приклад використання (для демонстрації, без реального навчання)
# Створюємо фіктивний вхідний тензор (Batch size = 1, Channels = 1, Height = 28, Width = 28)
dummy_input = torch.randn(1, 1, 28, 28)

# Пропускаємо вхідний тензор через модель
output = model(dummy_input)

# Виводимо розмір вихідного тензора (для одного зображення, 10 класів)
print("Розмір вихідного тензора:", output.shape)
print("Вихідний тензор (логарифм ймовірностей):", output)

# Виводимо структуру моделі
print("\nСтруктура моделі:")
print(model)

# Додатково: Виведемо кількість параметрів моделі
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'\nКількість нав|чальних параметрів: {count_parameters(model)}')




# Визначення пристрою (GPU, якщо доступно, інакше CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
model.to(device)

# Визначення функції втрат та оптимізатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Навчання моделі
epochs = 10 # Кількість епох для навчання

loss_history = []

for epoch in range(epochs):
    running_loss = 0.0
    # Використовуємо tqdm для відображення прогресу
    for images, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}'):
        # Переміщуємо дані на пристрій
        images, labels = images.to(device), labels.to(device)

        # Обнуляємо градієнти оптимізатора
        optimizer.zero_grad()

        # Прямий прохід
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Зворотний прохід та оптимізація
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        optimizer.step()

        running_loss += loss.item()
        loss_history.append(running_loss / len(trainloader))

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

print('Finished Training')

# Оцінка моделі на тестовому датасеті
model.eval() # Переводимо модель в режим оцінки
correct = 0
total = 0
with torch.no_grad(): # Вимикаємо розрахунок градієнтів
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # Отримуємо індекс класу з найбільшою ймовірністю
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

plt.plot(loss_history)
plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.title("Зміна Loss під час навчання")
plt.grid()
plt.show()

# Завдання 3-4

class CNNWithoutNorm(nn.Module):
    def __init__(self):
        super(CNNWithoutNorm, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        # Застосовуємо перший згортковий шар, ReLU активацію та пулінг
        x = self.pool(F.relu(self.conv1(x)))  # Bx32x14x14
        # Застосовуємо другий згортковий шар, ReLU активацію та пулінг
        x = self.pool(F.relu(self.conv2(x)))  # Bx64x7x7
        # Вирівнюємо тензор для повнозв'язного шару
        x = x.view(-1, 64 * 7 * 7)  # Bx64x7x7 -> Bx3136
        # Застосовуємо повнозв'язний шар
        x = self.fc1(x)
        return x


class CNNWithNormDropout(nn.Module):
    def __init__(self):
        super(CNNWithNormDropout, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)

        # Вирівнюємо тензор для повнозв'язного шару
        x = x.view(-1, 64 * 7 * 7)  # Bx64x7x7 -> Bx3136
        # Застосовуємо повнозв'язний шар
        x = self.fc1(x)
        return x

def train_model(model, epochs=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.001
    )

    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(
            trainloader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )
            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = (
            running_loss / len(trainloader)
        )
        loss_history.append(
            epoch_loss
        )

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {epoch_loss:.4f}"
        )
    return model, loss_history

def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )
            total_loss += loss.item()
            _, predicted = torch.max(
                outputs,
                1
            )
            total += labels.size(0)
            correct += (
                predicted == labels
            ).sum().item()
    accuracy = (
        100 * correct / total
    )
    average_loss = (
        total_loss / len(testloader)
    )

    return accuracy, average_loss

model_without = CNNWithoutNorm()
model_bn_dropout = CNNWithNormDropout()

print("\nТренування model_without (без нормалізації)")
model_without, loss_without = train_model(
    model_without,
    epochs=10
)
print("\nТренування model_bn_dropout (BatchNorm + Dropout)")
model_bn_dropout, loss_bn_dropout = train_model(
    model_bn_dropout,
    epochs=10
)

accuracy_without, test_loss_without = evaluate_model(
    model_without
)
accuracy_bn_dropout, test_loss_bn_dropout = evaluate_model(
    model_bn_dropout
)

plt.figure(figsize=(9, 5))
plt.plot(
    range(1, len(loss_without) + 1),
    loss_without,
    marker="o",
    label="Без нормалізації"
)
plt.plot(
    range(1, len(loss_bn_dropout) + 1),
    loss_bn_dropout,
    marker="o",
    label="BatchNorm + Dropout"
)
plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.title(
    "Порівняння Loss моделей"
)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

result = pd.DataFrame({
    "Конфігурація": [
        "Без нормалізації",
        "BatchNorm + Dropout"
    ],
    "Accuracy": [
        accuracy_without,
        accuracy_bn_dropout
    ],
    "Loss": [
        test_loss_without,
        test_loss_bn_dropout
    ]
})

print(result)
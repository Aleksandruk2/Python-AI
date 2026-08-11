import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Створення штучного набору даних
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Перетворення на тензори
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

print(X_tensor.shape, y_tensor.shape)
# Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Побудова моделі
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential( # використовується для виконування шарів один за одним
            nn.Linear(10, 32), # z=Wx+b
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # для бінарної класифікації
        )

    # Визначаємо функцію для того в якому порядку порядку і як виконувати шари нейромережі
    def forward(self, x):
        return self.model(x)

model = MLPClassifier()

# Оптимізатор та функція втрат

# Оптимізатор і функція втрат
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters()) # Оптимізатор градієнтного спуска

epochs = 200 # Кількість епох де ми можемо порівняти результат навчання нашої моделі

for epoch in range(epochs):

    # перевод моделі в режим тренування
    # змінює певні налаштування моделі для тренування
    model.train()

    # обнулення попередніх градієнтів
    optimizer.zero_grad()

    #--- forward propagation
    outputs = model(X_train)

    # підрахунок помилки (forward pass)
    loss = criterion(outputs, y_train)

    # підрахунок градієнтів - backward propagation
    loss.backward()

    # оновлення параметрів моделі
    optimizer.step()

    # вимкнення підрахунку градієнтів
    # Цей виклик вимикає збереження проміжних значень виходів
    # кожного шару нейромережі для подальшого обчислення градієнтів
    # за допомогою виклику backward()
    with torch.no_grad():

        # перевод моделі в режим інференсу
        model.eval()

        # отримання прогнозів
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        accuracy = ((val_outputs > 0.5) == y_test).float().mean()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy.item():.4f}")


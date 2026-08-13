from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

import matplotlib.pyplot as plt
import pandas as pd
import torch


# завдання 1

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

y = (y["num"] > 0).astype(int)

data = X.copy()
data["target"] = y

data = data.dropna()

X = data.drop("target", axis=1)
y = data["target"]

print("Баланс класів:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nФорма даних:")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# завдання 2

X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(13, 16),
    nn.ReLU(),

    nn.Linear(16, 8),
    nn.ReLU(),

    nn.Linear(8, 1),
    nn.Sigmoid()
)

print(model)


# завдання 3

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 30

loss_history = []
accuracy_history = []


for epoch in range(epochs):
    model.train()

    outputs = model(X_train_tensor)

    loss = criterion(
        outputs,
        y_train_tensor
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predictions = (outputs >= 0.5).float()

    accuracy = (
        predictions == y_train_tensor
    ).float().mean()

    loss_history.append(loss.item())
    accuracy_history.append(accuracy.item())

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {loss.item():.4f}, "
        f"Accuracy: {accuracy.item()*100:.2f}"
    )

model.eval()

with torch.no_grad():
    test_outputs = model(X_test_tensor)

    test_loss = criterion(
        test_outputs,
        y_test_tensor
    )

    test_predictions = (
        test_outputs >= 0.5
    ).float()

    test_accuracy = (
        test_predictions == y_test_tensor
    ).float().mean()

result = pd.DataFrame({
    "Метрика": [
        "Loss",
        "Accuracy"
    ],
    "Значення": [
        f"{test_loss.item():.4f}",
        f"{test_accuracy.item()*100:.2f}%"
    ]
})

print("\nПідсумкові метрики:")
print(result)


# завдання 4

# =========================
# Графік Loss
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    label="Loss"
)

plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.title("Зміна втрати під час навчання")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "loss_healthrisk_mlp.png",
    dpi=300
)

plt.show()


# =========================
# Графік Accuracy
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    accuracy_history,
    label="Accuracy"
)

plt.xlabel("Епоха")
plt.ylabel("Accuracy")
plt.title("Зміна точності під час навчання")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "accuracy_healthrisk_mlp.png",
    dpi=300
)

plt.show()
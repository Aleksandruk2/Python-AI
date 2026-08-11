from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
import matplotlib.pyplot as plt
import torch
import pandas as pd

# завдання 1

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# завдання 2

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 50

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(
        predictions,
        y_train_tensor
    )
    loss.backward()
    optimizer.step()
    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {loss.item():.4f}"
    )


# завдання 3

def train_model(weight_decay=0):
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay = weight_decay
    )

    epochs = 150

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(
            predictions,
            y_train_tensor
        )
        loss.backward()
        optimizer.step()

    return model

model_without = train_model()
model_l2 = train_model(weight_decay=1e-4)

model_without.eval()
model_l2.eval()

with torch.no_grad():
    pred_without = model_without(X_test_tensor)
    pred_l2 = model_l2(X_test_tensor)

pred_without = pred_without.numpy().flatten()
pred_l2 = pred_l2.numpy().flatten()


mae_without = mean_absolute_error(y_test, pred_without)
mae_l2 = mean_absolute_error(y_test, pred_l2)

r2_without = r2_score(y_test, pred_without)
r2_l2 = r2_score(y_test, pred_l2)

result = pd.DataFrame({
    "Конфігурація": [
        "Без регуляризації",
        "З регуляризацією (L2)"
    ],
    "MAE": [
        mae_without,
        mae_l2
    ],
    "R²": [
        r2_without,
        r2_l2
    ]
})

print(result)

# завдання 4

with torch.no_grad():
    y_pred_without = model_without(X_test_tensor)
    y_pred_l2 = model_l2(X_test_tensor)

y_pred_without = y_pred_without.numpy().flatten()
y_pred_l2 = y_pred_l2.numpy().flatten()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.scatter(
    y_test,
    y_pred_without,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Реальні значення")
plt.ylabel("Передбачені значення")
plt.title("Без регуляризації")
plt.grid(True)


plt.subplot(1, 2, 2)

plt.scatter(
    y_test,
    y_pred_l2,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Реальні значення")
plt.ylabel("Передбачені значення")
plt.title("З L2-регуляризацією")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "diabetes_healthrisk_analysis.png",
    dpi=300
)

plt.show()
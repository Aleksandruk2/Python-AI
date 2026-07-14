from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Завдання 1

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

df = df.sample(n=5000,random_state=42)

X = df[["MedInc"]]
Y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test,y_pred)

result = pd.DataFrame({
    "Модель": ["LinearRegression"],
    "Ознака": ["MedInc"],
    "R²": [r2]
})

print(result)

# Завдання 2

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

df = df.sample( n=5000,random_state=42)

X1 = df[["MedInc"]]
X2 = df[["MedInc", "AveRooms"]]
y = df["Price"]

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1, X2,
    y,
    test_size=0.2,
    random_state=42
)

model1 = LinearRegression()
model2 = LinearRegression()
model1.fit(X1_train,y_train)
model2.fit(X2_train,y_train)

y_pred1 = model1.predict(X1_test)
y_pred2 = model2.predict(X2_test)

r2_1 = r2_score(y_test,y_pred1)
r2_2 = r2_score(y_test,y_pred2)

result = pd.DataFrame({
    "Модель": ["LinearRegression", "LinearRegression"],
    "Ознака": ["MedInc", "MedInc + AveRooms"],
    "R²": [r2_1, r2_2]
})

print(result)

# Завдання 3

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

df = df.sample(
    n=5000,
    random_state=42
)

X = df[["MedInc", "AveRooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

result = pd.DataFrame({
    "Метрика": [
        "MAE",
        "MSE",
        "RMSE",
        "R²"
    ],
    "Значення": [
        mae,
        mse,
        rmse,
        r2
    ]
})

print(result)


# Завдання 4

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

df = df.sample(
    n=5000,
    random_state=42
)

X = df[["MedInc", "AveRooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(7,7))

plt.scatter(y_test,y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
plt.xlabel("Реальні значення")
plt.ylabel("Передбачені значення")
plt.title("Порівняння реальних і передбачених значень")

plt.savefig("prediction.png")
plt.show()
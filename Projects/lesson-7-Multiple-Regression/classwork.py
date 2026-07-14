from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# Завдання 1

X, y = make_regression(
    n_samples=2000,
    n_features=4,
    noise=10,
    random_state=42
)

df = pd.DataFrame(
    X,
    columns=["x1","x2","x3","x4"]
)

df["Target"] = y



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

r2 = r2_score(
    y_test,
    y_pred
)

coef_table = pd.DataFrame({
    "Ознака": [
        "x1",
        "x2",
        "x3",
        "x4"
    ],
    "Коефіцієнт": model.coef_
})

print(coef_table)

r2_table = pd.DataFrame({
    "Метрика": ["R²"],
    "Значення": [r2]
})

print(r2_table)



# Завдання 2

X, y = make_regression(
    n_samples=2000,
    n_features=4,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model1 = LinearRegression()
model1.fit(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LinearRegression()
model2.fit(X_train_scaled,y_train)

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test_scaled)

r2_without = r2_score(y_test, y_pred1)
r2_with = r2_score(y_test,y_pred2)

result = pd.DataFrame({
    "Масштабування": [
        "Ні",
        "StandardScaler"
    ],
    "R²": [
        r2_without,
        r2_with
    ]
})

print(result)


# Завдання 3

X, y = make_regression(
    n_samples=2000,
    n_features=4,
    noise=10,
    random_state=42
)

X_missing = X.copy()
X_missing[100:200,0] = np.nan

X_train1, X_test1, X_train2, X_test2, y_train, y_test = train_test_split(
    X,
    X_missing,
    y,
    test_size=0.2,
    random_state=42
)

model1 = LinearRegression()
model1.fit(
    X_train1,
    y_train
)

y_pred1 = model1.predict(
    X_test1
)

r2_without = r2_score(
    y_test,
    y_pred1
)

imputer = SimpleImputer(
    strategy="mean"
)
imputer.fit(X_train2)

X_train_imputed = imputer.transform(X_train2)
X_test_imputed = imputer.transform(X_test2)

model2 = LinearRegression()
model2.fit(
    X_train_imputed,
    y_train
)

y_pred2 = model2.predict(
    X_test_imputed
)

r2_imputed = r2_score(
    y_test,
    y_pred
)

result = pd.DataFrame({
    "Сценарій": [
        "Без пропусків",
        "Після імпутації"
    ],
    "R²": [
        r2_without,
        r2_imputed
    ]
})

print(result)



# Завдання 4

X, y = make_regression(
    n_samples=2000,
    n_features=4,
    noise=15,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

result = pd.DataFrame({
    "Метрика": [
        "Середнє R²",
        "Стандартне відхилення"
    ],
    "Значення": [
        scores.mean(),
        scores.std()
    ]
})

print(result)
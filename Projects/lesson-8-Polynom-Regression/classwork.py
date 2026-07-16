from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
import pandas as pd

# Завдання 1

housing = fetch_california_housing()

X = housing.data[:, [0]]
y = housing.target

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

print(f"R² на тестовій вибірці: {r2:.4f}")

# Завдання 2

housing = fetch_california_housing(as_frame=True)

df = housing.frame

X = df[["MedInc"]]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


models = {
    "Поліном": Pipeline([
        ("poly", PolynomialFeatures(degree=1, include_bias=False)),
        ("regressor", LinearRegression()),
    ]),
    "Поліном ^2": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("regressor", LinearRegression()),
    ]),
    "Поліном ^3": Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("regressor", LinearRegression()),
    ])
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    results.append([
        name,
        train_r2,
        test_r2,
    ])

result = pd.DataFrame(
        results,
        columns=[
            "Ступінь полінома",
            "Train R²",
            "Test R²",
        ]
    )

print(result)

# Завдання 3

housing = fetch_california_housing(as_frame=True)

df = housing.frame

X = df[[
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms"
]]

y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    test_r2 = model.score(X_test, y_test)

    results.append([
        name,
        test_r2,
        *model.coef_,
    ])

    if name == "Lasso":
        zero_count = (model.coef_ == 0).sum()


result = pd.DataFrame(
        results,
        columns=[
            "Ступінь полінома",
            "Test R²",
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms"
        ]
    )


print(result)
print(f"Кількість коефіцієнтів, що стали 0 у Lasso: {zero_count}")

# Завдання 4

housing = fetch_california_housing(as_frame=True)

df = housing.frame

X = df[[
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms"
]]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=0.1)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    test_r2 = model.score(X_test, y_test)
    non_zero = (model.coef_ != 0).sum()

    results.append([
        name,
        test_r2,
        non_zero,
    ])

result = pd.DataFrame(
        results,
        columns=[
            "Ступінь полінома",
            "Test R²",
            "Кількість ненульових коеф."
        ]
    )

print(result)

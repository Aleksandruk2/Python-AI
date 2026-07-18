
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Завдання 1

housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df[
    [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms"
    ]
]

y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

r2 = r2_score(
    y_test,
    y_pred
)

result = pd.DataFrame({
    "Модель": [
        "DecisionTreeRegressor"
    ],
    "Test R²": [
        r2.__round__(4)
    ]
})

print(result)

# Завдання 2

models = {
    "DecisionTree": DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)
    test_r2 = r2_score(
        y_test,
        y_pred
    )

    results.append([
        name,
        test_r2
    ])

result = pd.DataFrame(
    results,
    columns=[
        "Модель",
        "Test R²"
    ]
)

print(result)

# Завдання 3

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    results.append([
        name,
        test_r2
    ])

result = pd.DataFrame(
    results,
    columns=[
        "Модель",
        "Test R²"
    ]
)

print(result)

# Завдання 4

median_price = np.median(y)

y_class = (y > median_price).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred
)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

result = pd.DataFrame({
    "Метрика": ["Accuracy", "Precision", "Recall", "F1"],
    "Значення": [accuracy, precision, recall, f1]
})

print(result)
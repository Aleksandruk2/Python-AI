import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(42)

# Ціни на автомобілі
price = np.arange(10, 63, 1).reshape(-1, 1)

# Робимо залежності між цінами авто та продажами
base_sales = 480 * np.exp(-0.28 * price.ravel())

# print("price", len(price))
# print("price", price)

# print("base_sales", len(base_sales))

# Додамо випадковий шум
noise = rng.normal(0, 6, size=price.shape[0])
# print("noise = ",noise)
sales = np.round(base_sales + noise).astype(int)
# print(sales)
# Наші продажі авто
sales = np.clip(sales, 1, None)
# print(sales)

# Ділимо дані на 2 групи тестові і навчальні
X_train, X_test, y_train, y_test = train_test_split(price,
                                                    sales,
                                                    test_size=0.25,
                                                    random_state=42)
# Проводимо налаштування
models = {
    "Лінійна": Pipeline([
        ("regressor", LinearRegression()),
    ]),
    "Полімно ^2": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("regressor", LinearRegression()),
    ]),
    "Полімно ^3": Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("regressor", LinearRegression()),
    ])
}

colors = {
    "Лінійна": "red",
    "Полімно ^2": "green",
    "Полімно ^3": "darkorange"
}

styles = {
    "Лінійна": "--",
    "Полімно ^2": "-",
    "Полімно ^3": "-."
}

new_price = np.array([[45]])  # Двох мірний масив із 1 елементо для прогнозування ціни

results = {}

# навчання моделі
for name, model in models.items():
    # Навчання моделі
    model.fit(X_train, y_train)
    # прогнозування на основі тестових даних
    y_pred = model.predict(X_test)
    # Обчислення метрики
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Для запобігання перенавчання моделі
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    # Робимо прогноз
    predication = model.predict(new_price)[0]

    results[name] = {
        "cv": cv.mean(),
        "predication": predication,
    }
    print(f"\n{name}")
    print(f"R^2 = {r2:.3f}")
    print(f"MAE = {mae:.2f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"CV = {cv.mean():.3f}")
    print(f"Прогноз = {predication:.1f}")

# ------------------------------------------------------------------
# Перенавчаємо моделі на всіх даних
# ------------------------------------------------------------------
final_models = {}

for name, model in models.items():
    model.fit(price, sales)
    final_models[name] = model

# Створюємо плавну вісь X
X_smooth = np.linspace(price.min(), price.max(), 300).reshape(-1, 1)

# ------------------------------------------------------------------
# Будуємо графік
# ------------------------------------------------------------------
plt.figure(figsize=(12, 7))

plt.scatter(price, sales,
            color="royalblue",
            s=45,
            label="Навчальні дані")

for name, model in final_models.items():
    curve = model.predict(X_smooth)

    plt.plot(
        X_smooth,
        curve,
        color=colors[name],
        linestyle=styles[name],
        linewidth=2.5,
        label=name
    )

    pred = results[name]["predication"]

    plt.scatter(
        new_price,
        pred,
        color=colors[name],
        marker="X",
        s=180
    )

plt.title("Прогноз продажу автомобілів")
plt.xlabel("Ціна автомобіля (тис. $)")
plt.ylabel("Продано автомобілів")
plt.grid(alpha=0.3)
plt.legend()

plt.show()

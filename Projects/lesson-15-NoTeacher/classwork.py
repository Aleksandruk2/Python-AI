import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import axis

from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# завдання 1

# Завантажуємо датасет Wine
wine = load_wine()

print("\nДатасет Wine завантажено.")

# Перетворюємо дані на pandas.DataFrame
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

df["target"] = wine.target

print("\nПерші 5 рядків датасету:")
print(df.head())

print("\nРозмір датасету:")
print(f"Кількість об'єктів: {df.shape[0]}")
print(f"Кількість ознак: {len(wine.feature_names)}")


X = df.drop("target", axis=1)
y = df["target"]

# Розділяємо дані на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nРозподіл даних:")
print(f"Навчальна вибірка: {X_train.shape[0]} об'єктів")
print(f"Тестова вибірка:   {X_test.shape[0]} об'єктів")

# Масштабуємо ознаки за допомогою StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X.columns,
    index=X_train.index
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=X.columns,
    index=X_test.index
)

print("\nПерші 5 рядків масштабованих навчальних даних:")
print(X_train_scaled_df.head())

print("\nСередні значення масштабованих ознак:")
print(X_train_scaled_df.mean().round(3))

print("\nСтандартні відхилення масштабованих ознак:")
print(X_train_scaled_df.std().round(3))

# завдання 2

pca = PCA(n_components=2)

# Виконуємо перетворення навчальних даних
X_train_pca = pca.fit_transform(X_train_scaled)

print("\nРозмір даних після PCA:")
print(f"До PCA:    {X_train_scaled.shape}")
print(f"Після PCA: {X_train_pca.shape}")

print("\nПояснена дисперсія:")
print(
    f"PC1: {pca.explained_variance_ratio_[0]:.2%}"
)
print(
    f"PC2: {pca.explained_variance_ratio_[1]:.2%}"
)

print(
    f"Разом: {pca.explained_variance_ratio_.sum():.2%}"
)

pca_df = pd.DataFrame(
    X_train_pca,
    columns=["PC1", "PC2"]
)

pca_df["target"] = y_train.to_numpy()

print("\nПерші 5 об'єктів після PCA:")
print(pca_df.head())

# Будуємо scatter-графік

plt.figure(figsize=(10, 7))

for target_class in sorted(pca_df["target"].unique()):
    class_data = pca_df[
        pca_df["target"] == target_class
    ]
    print(f"{target_class}: {class_data.shape}")

    plt.scatter(
        class_data["PC1"],
        class_data["PC2"],
        s=60,
        label=f"Клас {target_class}"
    )

plt.xlabel("Головна компонента 1 (PC1)")
plt.ylabel("Головна компонента 2 (PC2)")

plt.title("PCA Components Visualization")

plt.legend()
plt.grid(alpha=0.3)

plt.show()

# завдання 3

components_list = [2, 5, 10]
results = []

for n_components in components_list:
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(
        X_train_pca,
        y_train
    )

    y_pred = model.predict(X_test_pca)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    results.append({
        "Компоненти PCA": n_components,
        "Accuracy": accuracy
    })

print("\nРезультати класифікації:")
for result in results:
    print(
        f"PCA = {result['Компоненти PCA']:2d} | "
        f"Accuracy = {result['Accuracy']:.4f}"
    )

# завдання 4

X = [result["Компоненти PCA"] for result in results]
Y = [result["Accuracy"] for result in results]

print(X)
print(Y)

plt.plot(X, Y, marker="o")

plt.xlabel("Кількість компонентів")
plt.ylabel("Точність класифікації")
plt.title("Залежність точності від кількості компонентів PCA")
plt.grid(True)
plt.savefig("pca_accuracy_results.png")

plt.show()

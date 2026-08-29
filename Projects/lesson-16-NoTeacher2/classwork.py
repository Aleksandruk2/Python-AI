import pandas as pd
import matplotlib.pyplot as plt
from fontTools.merge import cmap
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# Завдання 1

# Створюємо набір даних
X, y = make_blobs(
    n_samples=600,
    centers=5,
    cluster_std=1.2,
    random_state=0
)

# Перетворюємо дані в DataFrame
df = pd.DataFrame(X, columns=["income", "spending_score"])

print(df.head())

plt.scatter(
    df["income"],
    df["spending_score"]
)

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Розподіл клієнтів")

# plt.show()

# Завадання 2

# Створюємо модель K-Means
kmeans = KMeans(
    n_clusters=5,
    random_state=0
)

# Навчаємо модель
kmeans.fit(df[["income", "spending_score"]])

# Додаємо результати кластеризації в DataFrame
df["segment"] = kmeans.labels_

# Виводимо результат
print(df.head())

# Будуємо scatter-графік
plt.scatter(
    df["income"],
    df["spending_score"],
    c=df["segment"]
)

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Сегментація клієнтів за допомогою K-Means")

# plt.show()

# Завдання 3

dbscan = DBSCAN(
    eps=1.5,
    min_samples=10
)

dbscan.fit(df[["income", "spending_score"]])

# Додаємо результати кластеризації в DataFrame
df["cluster_dbscan"] = dbscan.labels_

print(df.head())

plt.scatter(
    df["income"],
    df["spending_score"],
    c=df["cluster_dbscan"]
)

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Кластеризація клієнтів за допомогою DBSCAN")

# plt.show()

# Завдання 4

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc1 = axes[0].scatter(
    df["income"],
    df["spending_score"],
    c=df["segment"],
    s=40,
)
axes[0].set_title("K-Means Segmentation", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Income", fontsize=12)
axes[0].set_ylabel("Spending Score", fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.3)

sc2 = axes[1].scatter(
    df["income"],
    df["spending_score"],
    c=df["cluster_dbscan"],
    s=40,
)
axes[1].set_title("DBSCAN Segmentation", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Income", fontsize=12)
axes[1].set_ylabel("Spending Score", fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("client_segmentation_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
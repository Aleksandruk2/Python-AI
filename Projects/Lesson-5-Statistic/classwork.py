import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Завдання 1

dates = pd.date_range("2025-01-01", periods=30)

df = pd.DataFrame({
    "date": dates,
    "users": np.random.randint(100, 500, 30),
    "sessions": np.random.randint(150, 700, 30),
    "revenue": np.random.randint(1000, 10000, 30)
})

print(df)

corr = df[["users", "sessions", "revenue"]].corr()

print("\nКореляційна матриця:")
print(corr)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(df["users"], df["sessions"], color="blue")
axes[0, 0].set_title("Users vs Sessions")
axes[0, 0].set_xlabel("Users")
axes[0, 0].set_ylabel("Sessions")
axes[0, 0].grid(True)

axes[0, 1].scatter(df["users"], df["revenue"], color="green")
axes[0, 1].set_title("Users vs Revenue")
axes[0, 1].set_xlabel("Users")
axes[0, 1].set_ylabel("Revenue")
axes[0, 1].grid(True)

axes[1, 0].scatter(df["sessions"], df["revenue"], color="red")
axes[1, 0].set_title("Sessions vs Revenue")
axes[1, 0].set_xlabel("Sessions")
axes[1, 0].set_ylabel("Revenue")
axes[1, 0].grid(True)

axes[1, 1].plot(df["date"], df["revenue"], marker="o")
axes[1, 1].set_title("Revenue за датами")
axes[1, 1].set_xlabel("Дата")
axes[1, 1].set_ylabel("Revenue")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].grid(True)

plt.tight_layout()
# plt.show()

# Завдання 2

np.random.seed(42)

group = ["A"] * 100 + ["B"] * 100

converted_A = np.random.binomial(1, 0.20, 100)
converted_B = np.random.binomial(1, 0.30, 100)

converted = np.concatenate([converted_A, converted_B])

df = pd.DataFrame({
    "group": group,
    "converted": converted
})

print(df.head())

conversion = df.groupby("group")["converted"].mean()

print("\nКонверсія:")
print(conversion)

absolute_difference = conversion["B"] - conversion["A"]

relative_change = absolute_difference / conversion["A"] * 100

print(f"\nАбсолютна різниця: {absolute_difference:.3f}")
print(f"Відносна зміна: {relative_change:.2f}%")

n = 100

errors = []

for p in conversion:
    se = np.sqrt(p * (1 - p) / n)
    ci = 1.96 * se
    errors.append(ci)

print("\nДовірчі інтервали:")
for g, e in zip(conversion.index, errors):
    print(f"{g}: ±{e:.3f}")

fig, ax = plt.subplots(figsize=(6,5))

ax.bar(
    conversion.index,
    conversion.values,
    yerr=errors,
    capsize=8,
    color=["skyblue","orange"]
)

ax.set_title("Конверсія груп A/B")
ax.set_xlabel("Група")
ax.set_ylabel("Конверсія")

# plt.show()

# Завдання 3

population = np.random.exponential(scale=2, size=50000)

def sample_means(population, n, repeats=1000):
    means = []

    for i in range(repeats):
        sample = np.random.choice(population, size=n)
        means.append(np.mean(sample))

    return np.array(means)

means_5 = sample_means(population, 5)
means_30 = sample_means(population, 30)
means_100 = sample_means(population, 100)

print("Статистика вибіркових середніх:\n")

print(f"n = 5")
print(f"Середнє: {np.mean(means_5):.4f}")
print(f"Стандартне відхилення: {np.std(means_5):.4f}\n")

print(f"n = 30")
print(f"Середнє: {np.mean(means_30):.4f}")
print(f"Стандартне відхилення: {np.std(means_30):.4f}\n")

print(f"n = 100")
print(f"Середнє: {np.mean(means_100):.4f}")
print(f"Стандартне відхилення: {np.std(means_100):.4f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(population, bins=50, color="skyblue", edgecolor="black")
axes[0, 0].set_title("Генеральна сукупність")

axes[0, 1].hist(means_5, bins=30, color="orange", edgecolor="black")
axes[0, 1].set_title("Вибіркові середні (n = 5)")

axes[1, 0].hist(means_30, bins=30, color="green", edgecolor="black")
axes[1, 0].set_title("Вибіркові середні (n = 30)")

axes[1, 1].hist(means_100, bins=30, color="red", edgecolor="black")
axes[1, 1].set_title("Вибіркові середні (n = 100)")

plt.tight_layout()
# plt.show()

# Завдання 4

dates = pd.date_range(start="2025-01-01", periods=90)
sales = np.random.randint(100, 500, 90)

df = pd.DataFrame({
    "date": dates,
    "sales": sales
})

window = 7
df["rolling_mean"] = df["sales"].rolling(window).mean()
df["rolling_std"] = df["sales"].rolling(window).std()

print(df.head(20))

fig, ax = plt.subplots(2,1, figsize=(8,5))

ax[0].plot(df["date"],
         df["sales"],
         label="Продажі")

ax[0].plot(df["date"],
         df["rolling_mean"],
         linewidth=3,
         label="Ковзне середнє")

ax[0].set_title("Продажі та ковзне середнє")
ax[0].set_xlabel("Дата")
ax[0].set_ylabel("Продажі")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(df["date"],
         df["rolling_std"],
         color="red")

ax[1].set_title("Ковзне стандартне відхилення")
ax[1].set_xlabel("Дата")
ax[1].set_ylabel("Std")
ax[1].grid(True)

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
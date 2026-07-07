import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(10, 16))

# Завдання 1

months = ["Січ", "Лют", "Бер", "Квіт", "Трав", "Чер"]
plan = [100, 120, 140, 160, 180, 200]
fact = [95, 130, 135, 170, 175, 210]

axes[0,0].plot(months, plan, marker="o", label="План")
axes[0,0].plot(months, fact, marker="s", label="Факт")

axes[0,0].set_title("Планові та фактичні продажі")
axes[0,0].set_xlabel("Місяці")
axes[0,0].set_ylabel("Продажі")
axes[0,0].legend()
axes[0,0].grid(True)

# Завдання 2

ages = np.random.randint(18, 66, 100)

average = np.mean(ages)

axes[0,1].hist(ages, bins=10, color="skyblue", edgecolor="black")
axes[0,1].axvline(average, color="red", linestyle="--", label=f"Середнє = {average:.1f}")

axes[0,1].set_title("Розподіл віку")
axes[0,1].set_xlabel("Вік")
axes[0,1].set_ylabel("Кількість осіб")
axes[0,1].legend()

# Завдання 3

group1 = [78, 82, 85, 88, 90, 91, 95]
group2 = [65, 70, 72, 75, 80, 85, 89]
group3 = [90, 92, 94, 95, 96, 98, 100]

axes[1,0].boxplot([group1, group2, group3])
axes[1,0].set_xticklabels(["Група 1", "Група 2", "Група 3"])

axes[1,0].set_title("Порівняння оцінок студентів")
axes[1,0].set_xlabel("Групи")
axes[1,0].set_ylabel("Оцінки")

# Завдання 4

days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
temperature = [20, 22, 24, 23, 25, 27, 26]
humidity = [65, 60, 55, 58, 52, 50, 54]

axes[1,1].plot(days, temperature, marker="o", label="Температура")

axes[1,1].plot(days, humidity, marker="s", label="Вологість")

axes[1,1].set_title("Температура та вологість за тиждень")
axes[1,1].set_xlabel("Дні")
axes[1,1].set_ylabel("Значення")
axes[1,1].legend()

plt.xticks(rotation=45)
axes[1,1].grid(True)

# Завдання 5

hours = list(range(24))
load = [20, 18, 15, 12, 10, 15, 25, 40, 55, 65, 70, 75,
        80, 78, 72, 68, 60, 55, 50, 45, 40, 35, 30, 25]

axes[2,0].plot(hours, load, color="blue", label="Навантаження")
axes[2,0].fill_between(hours, load, color="skyblue", alpha=0.4)

axes[2,0].set_title("Навантаження сервера протягом доби")
axes[2,0].set_xlabel("Години")
axes[2,0].set_ylabel("Навантаження (%)")
axes[2,0].grid(True)
axes[2,0].legend()

axes[2,1].axis("off")

plt.show()

# Завдання 6

time = [1, 2, 3, 4, 5, 6]

conversion = [2.1, 2.4, 2.8, 3.0, 3.2, 3.5]
retention = [80, 78, 77, 75, 76, 78]
average_check = [500, 520, 510, 530, 550, 560]
orders = [120, 135, 150, 145, 160, 170]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

fig.suptitle("Метрики продукту")

# Конверсія
axes[0, 0].plot(time, conversion, marker="o")
axes[0, 0].set_title("Конверсія")
axes[0, 0].grid(True)

# Утримання
axes[0, 1].plot(time, retention, marker="o", color="green")
axes[0, 1].set_title("Утримання")
axes[0, 1].grid(True)

# Середній чек
axes[1, 0].plot(time, average_check, marker="o", color="red")
axes[1, 0].set_title("Середній чек")
axes[1, 0].grid(True)

# Кількість замовлень
axes[1, 1].plot(time, orders, marker="o", color="orange")
axes[1, 1].set_title("Кількість замовлень")
axes[1, 1].grid(True)

plt.tight_layout()

plt.show()


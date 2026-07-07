import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(10, 16))

# Завдання 1

x = list(range(1, 11))
y = [i ** 2 for i in x]

axes[0,0].plot(x, y, color="red", marker="o")
axes[0,0].set_title("Графік функції y = x²")
axes[0,0].set(xlabel='X', ylabel='Y')
axes[0,0].grid(True)

# Завдання 2

months = ["Січ", "Лют", "Бер", "Квіт", "Трав", "Чер"]
sales = [150, 200, 250, 300, 280, 320]

axes[0,1].bar(months, sales, color="blue")
axes[0,1].set_title("Продажі за 6 місяців")
axes[0,1].set(xlabel='Місяці', ylabel='Продажі')

# Завдання 3

categories = ["Продукти", "Одяг", "Техніка"]
sales = [40, 30, 30]

axes[1,0].pie(
    sales,
    labels=categories,   # Підписи секторів
    autopct="%1.1f%%"    # Відображення відсотків
)
axes[1,0].set_title("Частки продажів за категоріями")

# Завдання 4

height = [160, 165, 170, 175, 180]
weight = [55, 60, 70, 72, 80]

axes[1,1].scatter(height, weight, color="green")
axes[1,1].set_title("Залежність ваги від зросту")
axes[1,1].set_xlabel("Зріст (см)")
axes[1,1].set_ylabel("Вага (кг)")
axes[1,1].grid(True)

# Завдання 5

hours = [0, 4, 8, 12, 16, 20, 24]
temperature = [15, 14, 18, 22, 25, 20, 17]

axes[2,0].plot(hours, temperature, marker="o")
axes[2,0].set_title("Температура протягом доби")
axes[2,0].set_xlabel("Години")
axes[2,0].set_ylabel("Температура (°C)")
axes[2,0].grid(True)

# Завдання 6

students = ["Alex", "Irene", "John", "Mary", "Boris"]
results = [85, 92, 78, 90, 88]

axes[2,1].bar(students, results, color="orange")
axes[2,1].set_title("Результати тесту")
axes[2,1].set_xlabel("Студенти")
axes[2,1].set_ylabel("Бали")


plt.show()
import matplotlib.pyplot as plt  # Імпортуємо модуль для побудови графіків
import numpy as np               # Імпортуємо бібліотеку NumPy

# Фіксуємо генератор випадкових чисел
np.random.seed(42)

# Створюємо масив значень X від 0 до 10
x = np.linspace(0, 10, 100)

# Обчислюємо три функції
y1 = np.sin(x)          # Синус
y2 = np.cos(x)          # Косинус
y3 = np.sin(x) + np.cos(x)   # Сума синуса і косинуса

# Створюємо фігуру та область побудови
fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')

# Будуємо три графіки
ax.plot(x, y1, label='sin(x)')
ax.plot(x, y2, label='cos(x)')
ax.plot(x, y3, '--', label='sin(x)+cos(x)')

# Додаємо заголовок
ax.set_title("Приклад побудови графіків у Matplotlib")

# Додаємо підписи осей
ax.set_xlabel("Вісь X")
ax.set_ylabel("Вісь Y")

# Встановлюємо власні поділки осі X
ax.set_xticks(
    [0, 2, 4, 6, 8, 10],
    ['0', '2', '4', '6', '8', '10']
)

# Встановлюємо власні поділки осі Y
ax.set_yticks([-2, -1, 0, 1, 2])

# Знаходимо максимум функції sin(x)
index = np.argmax(y1)

# Додаємо анотацію
ax.annotate(
    'Максимум sin(x)',
    xy=(x[index], y1[index]),
    xytext=(6, 1.4),
    arrowprops=dict(facecolor='black', shrink=0.05)
)

# Додаємо легенду
ax.legend()

# Відображаємо сітку
ax.grid(True)

# Показуємо графік
plt.show()
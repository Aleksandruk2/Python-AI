import numpy as np
import matplotlib.pyplot as plt
from fontTools.diff import color
from matplotlib.lines import lineStyles
from matplotlib.pyplot import xlabel, ylabel
from scipy import stats

np.random.seed(42)

grades = np.array([
    72, 75, 80, 82, 78,
    75, 74, 81, 79, 76,
    75, 83, 77, 84, 80,
    75, 73, 85, 79, 82,
    75, 78, 81, 76, 80,
    75, 74, 83, 79, 77
])

# Хочемо зломати систему, додамо випадково дуже високу оцінку
grades = np.append(grades, 1000)

# Обчислюємо статистичні данні
mean_grades = np.mean(grades)
median_grades = np.median(grades)
mode_grades = stats.mode(grades, keepdims=True).mode[0] # мода

fig, ax = plt.subplots(figsize=(10,5))
# bins - кількість стовбчивків на графіку
ax.hist(grades, bins=10, color='skyblue', edgecolor='black')
ax.axvline(mean_grades, color='red', linewidth=3, label=f'Середнє = {mean_grades:.1f}')
ax.axvline(median_grades, color='green', label=f'Медіана = {median_grades:.1f}')
ax.axvline(mode_grades, color='yellow', linewidth=3, linestyle='--', label=f'Мода = {mode_grades:.1f}')

ax.set_title("Результати контрольної роботи")
ax.set(xlabel='Бали', ylabel='Кількість студентів')
ax.legend()

plt.tight_layout()
plt.show()

print("Оцінки:")
print(grades)
print(f"Середнє: {mean_grades}")
print(f"Медіана: {median_grades}")
print(f"Мода: {mode_grades}")
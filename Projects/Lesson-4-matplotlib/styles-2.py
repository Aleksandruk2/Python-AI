import numpy as np
import matplotlib.pyplot as plt
from fontTools.merge import layout

# Розподіл від 0 до 2
x = np.linspace(0, 2, 100)

# Фігура та область
# figsize - це розмір фігури у дюймах
# layout - constrained - автоматично відбуваються відступи між елементами
fig, axes = plt.subplots(figsize=(5, 2.7), layout='constrained')

# Лінійна функція: y = x
axes.plot(x, x, label='Лінійна функція')
# Квадратична функція: y = x^2
axes.plot(x,x**2, label='Квадратична функція')
# Кубічна функція: y = x^3
axes.plot(x,x**3, label='Кубічна функція')

# Додаємо підписи осей
axes.set_xlabel('X значення')
axes.set_ylabel('Y значення')
# Заголовок графіку
axes.set_title("Відображення графіків")
# Відобразити параметри label на графіку
axes.legend()
# Відобразити вікно із графіком
plt.show()
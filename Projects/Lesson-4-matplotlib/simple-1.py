import numpy as np
import matplotlib.pyplot as plt

# Набір чисел від 0 до 2*pi = 6.28
x = np.linspace(0, 2*np.pi, 200)
# Знаходимо y
y = np.sin(x) # вертає набір для кожного x

# Створюємо фігуру та область малювання
# axes - область де малюємо сам графік
# figure - усе вікно графіка
figure, axes = plt.subplots()

axes.plot(x, y) # малюємо звичайний лінійний графік
plt.show() # Відображаємо вікно графіку
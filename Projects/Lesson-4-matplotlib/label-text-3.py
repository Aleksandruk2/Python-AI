import numpy as np
import matplotlib.pyplot as plt

# Середнє значення та стандартне відхилення
mu, sigma = 115, 15

# Генеруємо випадкові данні згідно вхідних параметрів
x = mu + sigma * np.random.randn(10000)

fig, axes = plt.subplots(figsize=(5, 2.7), layout='constrained')

# Будеємо гістаграму на основі вхідних данних
# x - набір значення
# 50 - кількість стовбчиків
# density - нормалізує гістограму до ймовірності
# alpha - прозорість стовбчивків
# facecolor - колір стовбчиків
n, bins, patches = axes.hist(x, 50, density=True, facecolor='C0', alpha=0.75)

axes.set(xlabel='Довжина в [см]', ylabel='Ймовірність')
axes.set_title('Розподіл довжини\n(база)')

axes.text(75,0.025, r'$\mu = 115, \sigma = 15$')

axes.axis([55,175,0,0.035])

plt.show() # Малюю гістаграму
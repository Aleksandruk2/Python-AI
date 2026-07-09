import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Для відтворюваності результатів
np.random.seed(42)

# =====================================================================
# 4. РОЗПОДІЛИ: рівномірний, нормальний, Пуассона
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Рівномірний
data_uniform = np.random.uniform(0, 10, 5000)
axes[0].hist(data_uniform, bins=30, color='mediumseagreen', edgecolor='black', density=True)
axes[0].axhline(1 / 10, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Рівномірний розподіл\nПриклад: час очікування ліфта (0-10 хв)')
axes[0].set_xlabel('Хвилини')

# Нормальний
data_normal = np.random.normal(170, 8, 5000)
axes[1].hist(data_normal, bins=40, color='cornflowerblue', edgecolor='black', density=True)
x = np.linspace(140, 200, 200)
axes[1].plot(x, stats.norm.pdf(x, 170, 8), color='red', linewidth=2)
axes[1].set_title('Нормальний розподіл (Гаусса)\nПриклад: зріст людей (μ=170, σ=8)')
axes[1].set_xlabel('см')

# Пуассона
data_poisson = np.random.poisson(4, 5000)
values, counts = np.unique(data_poisson, return_counts=True)
axes[2].bar(values, counts / len(data_poisson), color='sandybrown', edgecolor='black')
k = np.arange(0, 15)
axes[2].plot(k, stats.poisson.pmf(k, 4), 'ro-', linewidth=2, markersize=4)
axes[2].set_title('Розподіл Пуассона (λ=4)\nПриклад: дзвінки в кол-центр за годину')
axes[2].set_xlabel('Кількість подій')

plt.tight_layout()
plt.show()

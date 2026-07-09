import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Для відтворюваності результатів
np.random.seed(42)

# =====================================================================
# 5. КОВАРІАЦІЯ ТА КОРЕЛЯЦІЯ
# =====================================================================

n = 200
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Сильна позитивна
ad_spend = np.random.uniform(0, 100, n)
sales = 2 * ad_spend + np.random.normal(0, 20, n) + 50
corr1 = np.corrcoef(ad_spend, sales)[0, 1]
cov1 = np.cov(ad_spend, sales)[0, 1]
axes[0].scatter(ad_spend, sales, alpha=0.5, color='green')
axes[0].set_title(f"Реклама vs Продажі\ncov={cov1:.0f}, r={corr1:.2f} (сильний зв'язок)")
axes[0].set_xlabel('Витрати на рекламу, тис $')
axes[0].set_ylabel('Продажі, шт')

# Відсутня
height = np.random.normal(170, 10, n)
fav_number = np.random.uniform(0, 100, n)
corr2 = np.corrcoef(height, fav_number)[0, 1]
cov2 = np.cov(height, fav_number)[0, 1]
axes[1].scatter(height, fav_number, alpha=0.5, color='gray')
axes[1].set_title(f"Зріст vs Улюблене число\ncov={cov2:.1f}, r={corr2:.2f} (зв'язку немає)")
axes[1].set_xlabel('Зріст, см')
axes[1].set_ylabel('Улюблене число')

# Негативна
price = np.random.uniform(10, 100, n)
purchases = -0.8 * price + np.random.normal(0, 8, n) + 100
corr3 = np.corrcoef(price, purchases)[0, 1]
cov3 = np.cov(price, purchases)[0, 1]
axes[2].scatter(price, purchases, alpha=0.5, color='crimson')
axes[2].set_title(f'Ціна vs Кількість покупок\ncov={cov3:.0f}, r={corr3:.2f} (обернений зв\'язок)')
axes[2].set_xlabel('Ціна, $')
axes[2].set_ylabel('Кількість покупок')

plt.tight_layout()
plt.show()
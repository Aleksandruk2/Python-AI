import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Будуємо 2 класи розподілів
# Набір зі 100 даних - середнє значення буде 70 - групування навколо 4
class_s = np.random.normal(loc=70, scale=4, size=100)
class_c = np.random.normal(loc=70, scale=18, size=100)

# sharex=True - спільна вісь Х
# Буде спільна вісь Y - sharey=True
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

for ax, data, title in zip(
        axes,
        [class_s, class_c],
        ["Клас А - стабільні результати", "Клас Б - нестабільні результати"]
):
    mean = np.mean(data)
    variance = np.var(data)
    std = np.std(data)

    # Гістограма
    ax.hist(
        data,
        bins=18,
        color="cornflowerblue",
        edgecolor="black",
        alpha=0.75,
    )

    ax.axvline(mean,
               color="red",
               linewidth=2.5,
               label="Середнє"
               )

    ax.axvline(
        mean + std,
        color="green",
        linestyle="--",
        linewidth=2
    )

    ax.set_title(title)
    ax.set_xlabel("Бали")
    ax.set_ylabel("Кількість студентів")
    ax.set_xlim([0, 140])

    ax.text(
        0.98,
        0.95,
        f"Середнє = {mean:.1f}\n"
        f"Дисперсія = {variance:.1f}\n"
        f"СКВ = {std:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85)
    )

    ax.legend()

plt.suptitle(
    "Дисперсія показує розкид даних відносно середнього значення",
    fontsize=14,
    fontweight="bold",
)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

deta1 = np.array([1, 2, 3, 4, 5, 6])
deta2 = np.array([2, 2, 3, 1, 5, 3])
deta3 = np.array([3, 4, 6, 2, 1, 5])

fig, axes = plt.subplots(figsize=(5, 2.7))

axes.plot(np.arange(len(deta1)), deta1, label='deta1')
axes.plot(np.arange(len(deta2)), deta2, label='deta2')
# d - це буде ромб
axes.plot(np.arange(len(deta3)), deta3, 'd', label='deta3')

# axes.set(xlabel='Номер елемента', ylabel='Значення')
axes.set_xlabel('Номер елемента')
axes.set_ylabel('Значення')

axes.set_title("Порівнення наборів даних")

axes.legend()
axes.grid(True)
plt.show()
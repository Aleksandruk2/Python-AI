import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Попит на морозиво відносно температури
temperature = np.array([2,5,8,12,15,18,20,22,25,28,30,33])
ice_cream_sales = np.array([15,22,35,60,90,130,160,200,230,250,243,210])

print("Temperature: ", len(temperature))
print("Ice Cream: ", len(ice_cream_sales))
X = temperature.reshape(-1,1) # 2D масив ознак тому, що sklearn приймає 2D
y = ice_cream_sales
# print("X", X)

# Будуємо модель для регресії полінома і даємо степінь
model = make_pipeline(PolynomialFeatures(4), LinearRegression())
model.fit(X, y)

# Для прогнузування і плавної кривої зробимо набір
X_smooth = np.linspace(temperature.min(), temperature.max(), 200).reshape(-1,1)
y_smooth = model.predict(X_smooth) # Робимо прогнозування

# Робимо прогноз по даних
new_temperature = np.array([[27]])
prediction = model.predict(new_temperature) # Робимо прогноз для температури 27

print(f"Прогноз продіжів для 27 C: {prediction[0]:.0f} шт")
print(f"R² моделі = {model.score(X,y):.3f}")

plt.scatter(temperature, ice_cream_sales, color="orange", label="Реальні продажі")
plt.plot(X_smooth, y_smooth, color="navy", label="Поліном 4-го степеня")
plt.xlabel("Температура")
plt.ylabel("Продажі шт")
plt.legend()
plt.show()
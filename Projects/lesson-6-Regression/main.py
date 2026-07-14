# Для реалізації лінійної регресії
from sklearn.linear_model import LinearRegression
# для роботи з масивами
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# робить із рядка стовбець
area = np.array([50, 100, 150, 200, 250]).reshape(-1, 1)
price = np.array([100,150,200,300,400])

print("Площі ", area)
print("Ціни ", price)
# створили модель
model = LinearRegression()
# передали дані у модель
model.fit(area, price)

# Формуля для навчання моделі - y = b0 + b1*х
# Нова площа площа
new_area = np.array([[180]]) #прогнозування ціні згідно вказаної площі житла
# Передбачення, яка буде ціна для площі 180
prediction = model.predict(new_area)

print(f"Площа: {new_area[0,0]} м2 -> Пере: {prediction[0]} тис. грн.")
print(f"Формула: ціна = {model.intercept_:.2f} + {model.coef_[0]:.2f} * площа")

print("-"*70)
#Будуємо предикат - Отримуємо передбачення
predictions = model.predict(area)
#Середня абсолютна помилка
# Шукаємо різницю між ціною та пронозом
# шукаємо без знаку по модулю
# error - на скільки модель буде помилятися
mea = mean_absolute_error(price, predictions)
#Середньо квадратична помилка
#Помилки підносять до квадрату.
#Якщо помилки більші, то вони будуть важливішими
rmse = np.sqrt(mea)
# Коефіцієнт детермінації
# На скільки добре модель пояснює дані
r2 = r2_score(price, predictions)

print(f"MAE: {mea:.2f} (середня помилка)")
print(f"MAE: {rmse:.2f} (коренева квадратична помилка)")
print(f"MAE: {r2:.2f} (якість моделі, 1.0 - ідеальна модель)")

print("+++"*20)
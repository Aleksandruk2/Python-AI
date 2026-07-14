
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Отримуємо набір тестових даних із Каліфорнії - житло 1990
# as_frame=True - дані будуть у вигляді DataFrame
housing = fetch_california_housing(as_frame=True)

# Отримали дані у вигляді DataFrame
df = housing.frame

print(df.head()) # виводимод перші 5 рядків, якщо без head() - консоль просто впаде від кількості даних

df = df[[
    "MedInc", # Середіній дохід населення району
    "HouseAge", # середній вік будинку в найоні
    "AveRooms", # середня кількість кімнат у будинку
    "MedHouseVal" # медіана вартості будинку - по суті це вартість будику - цільова змінна
]]

# Створюжмо категоріальні ознаки
# Буде доданий новий стовпчик у таблицю для категорій
df["Category"] = np.where(df["HouseAge"] > 30, "Old", "New")

print("Нова таблиця із категоріями")
print(df.head())

# Наводими порядок в Даних - Отримуємо список ознак без цільової ознаки
# Набір колонок без колонки MedHouseVal
x = df.drop(columns="MedHouseVal") # При цьому сам df він не змінюється, лише вертає колонку MedHouseVal

# print("x = ", x.head())

# y - цільові ознаки для навчання моделі. Тобто train
y = df["MedHouseVal"]
# print("y = ")
# print(y.head())

# розподі данних і налаштовуємо random
# X - це список ознак у яких будемо робити аналіз - експуртизу
# y - це значення яке потрібно прогнозувати - цільова озанка
# test_size - 20% даних для тесту, 80% - це для навчання - виявлення закономірностей данних
# random_state=42 - для того щоб данні завжди були однаковими
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Робимо попередню обробку даних
numeric_features = [
    "MedInc",
    "HouseAge",
    "AveRooms",
]

category_features = [
    "Category",
]

preprocessor = ColumnTransformer(
    # Правила для обробки інформації
    transformers=[
        (
            # для числових данних
            # Назва трансформації
            "num",
            # Буде усі числові колонки приводити до значення - близького до 1
            # Віднімаємо середнє і ділимо на стандартне відхилиення
            StandardScaler(),
            # це будемо робити на основі ось цих даних
            numeric_features,
        ),
        (
            # для категоріальних даних
            "cat",
            # Приводить дані до числового формату
            # при цьому буде лише 1 колонка у якій буде вказано 0 або 1
            OneHotEncoder(drop="first"),
            category_features, # Застосовуємо до цих даних
        )
    ]
)

# Робимо конвеїр обробки даних
model = Pipeline([
    (
        # Назва першого етапу
        "preprocessor",
        # Попередня обробка даних: стандартизація та кодування категоріальних ознак
        preprocessor
    ),
    (
        # Назва другого етапу
        "regressor",
        LinearRegression() # множина лінійна регресія
    )
])

# Модель уже знає як працювати із даними, ми передали і вказали усі параметри
# Проводими машинне навчання
model.fit(X_train, y_train) # якщо провели машинне навчання, можна робити передбаченя

y_pred = model.predict(X_test) # Отримуємо масив прогнозів на основі тестових даних

# Оцінка моделі
# Передаємо тестові та отримані із передбачення данні
r2 = r2_score(y_test, y_pred) # Коефіцієнт детермінації

# Шукаємо абсолютну похибку
mae = mean_absolute_error(y_test, y_pred)

# Шукаємо середню квадратичну похибку
rmse = root_mean_squared_error(y_test, y_pred)

print("r2 score:", r2)
print("mae:", mae) # похибка обчислень - 0.6 - якщо будино коштує сотні тисяч - то похибка 60 тисяч
print("rmse:", rmse) # дана змінна реагує на досить великі відхилення від норми




# ==========================================
# 11. Коефіцієнти моделі
# ==========================================
# Формуємо список назв усіх ознак,
# які були використані моделлю після попередньої обробки.
feature_names = (
    # Додаємо числові ознаки.
    numeric_features +
    # Додаємо назви нових стовпців,
    # створених OneHotEncoder.
    list(
        # Отримуємо об'єкт попередньої обробки (ColumnTransformer)
        # із Pipeline.
        model.named_steps["preprocessor"]
                # Отримуємо трансформатор категоріальних ознак.
             .named_transformers_["cat"]
# Повертаємо назви стовпців після OneHotEncoder.
             # Наприклад:
             # Category → Category_New
             .get_feature_names_out(category_features)
    )
)
# Отримуємо коефіцієнти навченої моделі
# множинної лінійної регресії.
coef = model.named_steps["regressor"].coef_

# Створюємо DataFrame,
# у якому кожній ознаці відповідає її коефіцієнт.
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coef
})
# Виводимо заголовок.
print("\nКоефіцієнти моделі:")
# Виводимо таблицю коефіцієнтів.
print(coef_df)

# Ознака Коефіцієнт         Інтерпретація
# MedInc     0.848     Найбільший позитивний вплив на прогнозовану вартість будинку.
# HouseAge     0.294     Старіші будинки, як правило, мають вищу прогнозовану вартість.
# AveRooms     −0.067     Незначний негативний вплив. За інших однакових умов збільшення середньої кількості кімнат трохи зменшує прогнозовану ціну.
# Category_Old −0.197     Будинки категорії Old мають нижчу прогнозовану вартість порівняно з базовою категорією (New)

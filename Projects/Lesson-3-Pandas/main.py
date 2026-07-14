import numpy as np
import pandas as pd
from pandas import read_excel

print("---------Pandas Working----------\n")

items = pd.Series([1,2,False,4,5,'a',7,'b',9])
print(items)


item_index = pd.Series([13,3,23], index=['a','b','c'])
print(item_index)

dict = pd.Series({'a':1,'b':2,'c':3})
print(dict)


# DataFrame - табличка - двохвимірний масив
df = pd.DataFrame({
    'name':['Anna','Bogdan','Jimin','Jimin','Jimin'],
    'age':[4,5,6,7,8],
    'City':['Kyiv','Lviv','Odesa','Kharkiv','Lutsk']
})
print("DataFrame - index")
print(df)


# Запис та читання csv - файлу
df.to_csv('data.csv',index=False)
print("DataFrame - columns")
print(df.columns)

my_reader = pd.read_csv('data.csv', encoding='utf-8')
print(my_reader)
print("---------Read DateFrame----------")
print(my_reader)


# Запис данних у Excel та читання
df.to_excel("friends.xlsx", sheet_name='Дівчата', index=False)


read_excel = pd.read_excel("friends.xlsx")
print("---------Read Excel----------")
print(read_excel)


# Вивести дані у БД
import sqlite3

conn = sqlite3.connect('friends.db')
df.to_sql('friends', con=conn, if_exists='replace', index=False)

np.random.seed(42) # Ініціалізація рандому

df = pd.DataFrame({
    'Відділ': np.random.choice(['HR','IT','Sales'], size=100),
    'Зарплата': np.random.randint(3000,8000, size=100),
    'Стаж': np.random.randint(1,20, size=100),
})

print("Перші 5 рядків:")
print(df.head())

# Статистика по даним
print("\n Статистика:")
print(df.describe())
# count (кількість)
# mean (середнє)
# std (стандартне відхилення)
# min (мінімум)
# 25% (перший квартиль)
# 50% (медіана)
# 75% (третій квартиль)
# max (максимум)


print("\nЗагальна інформація")
print(df.info())

print("\nВипадкові значення - 3 штуки:")
print(df.sample(3))

# Групування і аналіз

print("\nСередня зарплата по відділах:")
print(df.groupby("Відділ")['Зарплата'].mean())
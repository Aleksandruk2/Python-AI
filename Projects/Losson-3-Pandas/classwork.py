import pandas as pd

# Завдання 1

data = {
    "Ім'я": ["Іван", "Марія", "Петро", "Олена", "Андрій"],
    "Вік": [25, 30, 22, 28, 35],
    "Місто": ["Київ", "Львів", "Одеса", "Луцьк", "Дніпро"]
}

df = pd.DataFrame(data)

print("DataFrame:")
print(df)

rows, cols = df.shape
print("\nКількість рядків:", rows)
print("Кількість стовпців:", cols)

print("\nНазви стовпців:")
print(df.columns)

print("\nПерші три рядки:")
print(df.head(3))

# Завдання 2

data = {
    "Назва": [
        "Inception",
        "Titanic",
        "Avatar",
        "Interstellar",
        "The Matrix"
    ],
    "Жанр": [
        "Sci-Fi",
        "Drama",
        "Sci-Fi",
        "Sci-Fi",
        "Action"
    ],
    "Рік": [
        2010,
        1997,
        2009,
        2014,
        1999
    ],
    "Рейтинг": [
        8.8,
        7.9,
        7.8,
        8.7,
        8.7
    ]
}

df = pd.DataFrame(data)

df.to_csv("movies.csv", index=False, encoding="utf-8-sig")
print("Файл movies.csv успішно створено!")

df = pd.read_csv("movies.csv")

print("Перші 5 рядків:")
print(df.head())

rows, cols = df.shape

print("\nКількість рядків:", rows)
print("Кількість стовпців:", cols)

print("\nТипи даних:")
print(df.dtypes)

print("\nНаявність пропусків:")
print(df.isnull().sum())


# Завдання 3

data = {
    "Ім'я": ["Іван", "Марія", "Петро", "Олена", "Андрій"],
    "Вік": [25, 32, 29, 41, 35],
    "Відділ": ["IT", "HR", "Продажі", "Бухгалтерія", "Маркетинг"],
    "Зарплата": [60000, 45000, 55000, 70000, 50000]
}

df = pd.DataFrame(data)
print("\nСпівробітники із зарплатою понад 50000:")
print(df[df["Зарплата"] > 50000])

print("\nСортування за віком:")
print(df.sort_values(by="Вік"))

print("\nСортування за зарплатою:")
print(df.sort_values(by="Зарплата", ascending=False))


# Завдання 4

data = {
    "Товар": [
        "Ноутбук", "Миша", "Клавіатура", "Монітор", "Навушники",
    ],
    "Категорія": [
        "Електроніка", "Аксесуари", "Аксесуари", "Електроніка", "Аксесуари"
    ],
    "Ціна": [25000, 500, 1200, 8000, 3000],
    "Кількість": [5, 20, 15, 8, 12]
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)

df["Виручка"] = df["Ціна"] * df["Кількість"]

print("\nОновлений DataFrame:")
print(df)

# Завдання 5

data = {
    "Ім'я": [
        "Іван", "Марія", "Петро", "Олена",
        "Андрій", "Софія", "Максим", "Катерина"
    ],
    "Курс": [1, 1, 2, 2, 3, 3, 4, 4],
    "Оцінка": [85, 92, 78, 95, 88, 76, 90, 84]
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)

avg_grade = df.groupby("Курс")["Оцінка"].mean()
print("\nСередня оцінка за кожним курсом:")
print(avg_grade)

grades = df.groupby("Курс")["Оцінка"].agg(["max", "min"])
print("\nМаксимальна та мінімальна оцінка за кожним курсом:")
print(grades)


# Завдання 6

products = {
    "ID": [1, 2, 3, 4, 5],
    "Назва": ["Ноутбук", "Миша", "Клавіатура", "Монітор", "Навушники"],
    "Ціна": [25000, 500, 1200, 8000, 3000]
}

sales = {
    "ID": [1, 2, 3, 4, 5],
    "Кількість": [3, 15, 8, 5, 10]
}

df_products = pd.DataFrame(products)
df_sales = pd.DataFrame(sales)

# Виведення початкових таблиць
print("Товари:")
print(df_products)

print("\nПродажі:")
print(df_sales)

df = pd.merge(df_products, df_sales, on="ID")

df["Сума"] = df["Ціна"] * df["Кількість"]
print("\nПідсумковий DataFrame:")
print(df)
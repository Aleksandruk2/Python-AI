import numpy as np

# Завдання 1

arr = np.random.randint(0, 51, 20)

print("Масив:")
print(arr)

threshold = 42
# threshold = int(input("Введіть порогове значення: "))
count = np.sum(arr > threshold)
print(f"\nКількість елементів, більших за {threshold}: {count}")

max_value = np.max(arr)
max_index = int(np.argmax(arr))

print("Максимальне значення:", max_value)
print("Позиція першої появи:", max_index)

sorted_arr = np.sort(arr)[::-1]
print("\nВідсортований масив за спаданням:")
print(sorted_arr)


# Завдання 2

# low = int(input("Введіть нижню межу: "))
low = 1
# high = int(input("Введіть верхню межу: "))
high = 42

matrix = np.random.randint(low, high + 1, (5, 5))

print("Вихідна матриця:")
print(matrix)

diagonal = np.diag(matrix)

print("\nЕлементи головної діагоналі:")
print(diagonal)

diagonal_sum = np.sum(diagonal)

print("\nСума елементів головної діагоналі:", diagonal_sum)

new_matrix = matrix.copy()

new_matrix[np.triu_indices(5, k=1)] = 0

print("\nМатриця після обнулення елементів вище головної діагоналі:")
print(new_matrix)


# Завдання 3

# start = int(input("Введіть початок діапазону: "))
# end = int(input("Введіть кінець діапазону: "))

start = 1
end = 30

arr = np.arange(start, end + 1)

if len(arr) != 30:
    print("Помилка! Для матриці 6x5 потрібно рівно 30 чисел.")
else:
    matrix = arr.reshape(6, 5)

    print("Матриця:")
    print(matrix)

    row_sum = np.sum(matrix, axis=1)
    print("\nСуми за рядками:", row_sum)

    col_max = np.max(matrix, axis=0)
    print("\nМаксимуми за стовпцями:", col_max)


# Завдання 4

# start = int(input("Введіть початок діапазону(включаючи з відємними часлами): "))
# end = int(input("Введіть кінець діапазону: "))

start = -30
end = 30

arr = np.random.randint(start, end+1, 15)

print("Масив:")
print(arr)

negative = arr[arr < 0]

print("\nВід'ємні елементи:")
print(negative)

new_arr = arr.copy()
new_arr[new_arr < 0] = 0

print("\nМасив після заміни від'ємних елементів на 0:")
print(new_arr)

zero_count = np.sum(new_arr == 0)

print("\nКількість нульових елементів:", zero_count)


# Завдання 5

# n = int(input("Введіть довжину масивів: "))
n = 10

arr1 = np.random.randint(0, 11, n)
arr2 = np.random.randint(10, 21, n)

print("Перший масив:")
print(arr1)

print("\nДругий масив:")
print(arr2)

merged = np.concatenate((arr1, arr2))

print("\nОб'єднаний масив:")
print(merged)

sum_arr = arr1 + arr2

print("\nПоелементна сума:")
print(sum_arr)

diff_arr = arr1 - arr2

print("\nПоелементна різниця:")
print(diff_arr)


# Завдання 6

# rows = int(input("Введіть кількість рядків: "))
# cols = int(input("Введіть кількість стовпців: "))

rows = 3
cols = 5

matrix = np.arange(1, rows * cols + 1).reshape(rows, cols)

print("Вихідна матриця:")
print(matrix)

# new_rows = int(input("\nВведіть нову кількість рядків: "))
# new_cols = int(input("Введіть нову кількість стовпців: "))

new_rows = 5
new_cols = 3

if rows * cols != new_rows * new_cols:
    print("Помилка! Неможливо змінити форму матриці.")
else:
    new_matrix = matrix.reshape(new_rows, new_cols)

    print("\nНова матриця:")
    print(new_matrix)

    row_min = np.min(new_matrix, axis=1)

    row_max = np.max(new_matrix, axis=1)

    print("\nМінімальні значення по рядках:")
    print(row_min)

    print("\nМаксимальні значення по рядках:")
    print(row_max)

    total = np.sum(new_matrix)

    print("\nЗагальна сума елементів:", total)
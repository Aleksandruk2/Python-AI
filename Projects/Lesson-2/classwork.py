import numpy as np

# Завдання 1
arr = np.random.randint(0, 101, 10)

print("Масив до сортування:")
print(arr)

print("\nМінімальне значення:", np.min(arr))
print("Максимальне значення:", np.max(arr))
print("Середнє значення:", np.mean(arr))

sorted_arr = np.sort(arr)
print("\nМасив після сортування:")
print(sorted_arr)

# Завдання 2

matrix = np.random.randint(1, 51, (4, 4))

print("Масив:")
print(matrix)

row_sums = np.sum(matrix, axis=1)

col_sums = np.sum(matrix, axis=0)

average = np.mean(matrix)

print("\nСуми за рядками:", row_sums)
print("Суми за стовпцями:", col_sums)
print("Загальне середнє значення:", round(average, 2))

# Завдання 3

array = np.arange(15).reshape((3, 5))

print("Масив:")
print(array)

transpose1 = array.T
transpose2 = np.transpose(array)

print("Транспортований Масив 1:")
print(transpose1)

print("Транспортований Масив 2:")
print(transpose2)

reshaped = array.reshape(5, 3)

print("\nМасив після reshape(5, 3):")
print(reshaped)


# Завдання 4

arr = np.random.rand(10,10)
print("Масив:")
print(arr)

min_index_row, min_index_col = np.unravel_index(np.argmin(arr), arr.shape)
max_index_row, max_index_col = np.unravel_index(np.argmax(arr), arr.shape)

print("\nМінімальне значення:", np.min(arr))
print("Індекс мінімального елемента (рядок, стовпець):", min_index_row,",",min_index_col)

print("\nМаксимальне значення:", np.max(arr))
print("Індекс максимального елемента (рядок, стовпець):", max_index_row,",",max_index_col)


# Завдання 5

arr = np.random.randint(-10, 11, 15)

print("Масив цілих чисел:")
print(arr)

positive = arr[arr > 0]
even = arr[arr % 2 == 0]

print("\nДодатні елементи:")
print(positive)

print("\nПарні елементи:")
print(even)


# Завдання 6
arr1 = np.arange(1,6)
arr2 = np.arange(6,11)

print("Масив цілих чисел 1:")
print(arr1)

print("Масив цілих чисел 2:")
print(arr2)

arr3 = np.concatenate((arr1, arr2), axis=0)

print("Масив цілих чисел 3 (обєднаний):")
print(arr3)

arr2d = np.array([arr1,arr2])

print("Двохвимірний Масив цілих чисел:")
print(arr2d)

transposed = arr2d.T

print("Транспортований Двохвимірний Масив цілих чисел:")
print(transposed)
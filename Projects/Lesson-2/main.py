import numpy as np

# одномірний масив
# myArray = np.array([1,2,3])
# print(myArray)


# Багатомірний масив
# testMatrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(testMatrix)


# Нульовий масив, заповнюється нулями
# zeros =np.zeros((3,3)) # матриця 3х3
# print(zeros)


# Одиничний масив
# ones = np.ones((3,2))
# print(ones)


# Масив із множини значень
# Значення від 0 до 10 із кроком 3
# arr = np.arange(0,10,3)
# print(arr)


# Матриця у якої значення(1) генеруються по діагоналі
# myArr = np.eye(4)
# print(myArr)


# Одиничний масив - рандом від 0,1
# myrand = np.random.rand(3,4)
# print(myrand)


# Масив рандомних цілих чисвел від 0 (10 не включається)
# myRandMix = np.random.randint(low=0, high=100, size=(2,5))
# print(myRandMix)


# Не звичайний масив
# myArray = np.array([[1,"Salo",3], [3,2,False]])
# print(myArray)


# Розмір масиву
# print(f"Розмір масиву {myArray.shape}")
# print(f"Виміри масиву {myArray.ndim}") # Двох виміргий масив
# print(f"Тип даних масиву {myArray.dtype}")
# print(f"Кількість елементів масиву {myArray.size}")


# Отримання елементів із масиву
# myList = np.array([23,13,4,5,12,54,-2,3,-55])
# print("Перший елемент", myList[0])
# print("Зріз 1:4", myList[1:4]) # від index 1 до index 3
# print("Зріз від початку :3", myList[:3]) # від початку 3 елемента
# print("Зріз від заданого і до кінця 2:", myList[2:]) # від 2 індекса до кінця
# print("Обираємо кожен вказаний ::2", myList[::2]) # кожен другий елемент
# print("Обираємо кожен вказаний 4::2", myList[4::2]) # з 4 індексу через 2
# print("Обираємо кожен вказаний 1:7:2", myList[1:7:2]) # з 1 індексу через 2 по 7


# Обробка двох вимірних матриць
# matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(matrix)

# Зріз матриці
# print("2d array 1:,:2", matrix[1:,:2])


# Операції над масивами
# arr_a = np.array([2,4,2,-2,4,42,-21,33])
# arr_b = np.array([-1,4,8,3,-42,22,31,13])
# print("a:",arr_a)
# print("b:",arr_b)
# print("a + b:",arr_a + arr_b)
# print("a - b:",arr_a - arr_b)
# print("a * b:",arr_a * arr_b)
# print("a / b:",arr_a / arr_b)
# print("a // b:",arr_a // arr_b)

# items = np.linspace(0,1,12)
# print("linspace:",items)


# Статичтині методи у python
data = np.array([2,3,4,9,2,5,5,6,7,8])
print("data", data)
print("mean", np.mean(data)) #середнє арифметичне
print("median", np.median(data)) # медіана - середнє значення, більш точне
print("max", np.max(data)) # максимальне значення
print("min", np.min(data)) # мінімальне значення
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn # Модулі для створення нейронної мережі
import torch.optim as optim # Оптимізатори для навчання мережі


# Для роботи dataset
iris = load_iris()
# print("Iris dataset loaded", iris.data.shape)
# Маємо 4 класи даних по 150 наборів
# print(iris.data[:5])
#Отримуємо дані з датасет
X = iris.data # параметри даних квіток
# 1. sepal length — довжина чашолистка
# 2. sepal width  — ширина чашолистка
# 3. petal length — довжина пелюстки
# 4. petal width  — ширина пелюстки
y = iris.target ##['setosa' 'versicolor' 'virginica']
# print(y.shape) # 150 квітів
print("X shape: ", X.shape)
print("y shape: ", y.shape)

print("Характеристики")
for feature in iris.feature_names:
    print(" - ", feature)

print("\nКласи:")
for class_name in iris.target_names:
    print(" - ", class_name)

# Створюємо тестові дані і правильні дані
X_train, X_test, y_train, y_test = train_test_split(
    X, # Вхідні дані
    y, # Правильні класи
    test_size = 0.2, # 20% використовується для тесту
    random_state = 42, # Щоб рандом працював завжди однаково
    stratify = y # Зберігаємо позицію класів
)

print("Навчальні приклади: ", len(X_train))
print("Тестові приклади: ", len(X_test))

# print("X_train: ", X_train[:5])
# print("X_test: ", X_test[:5])

# Маштабування даних
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Змінюємо дані до зручного маштабу для мережі
X_test = scaler.transform(X_test)
# print("X_train: ", X_train[:5])
# print("X_test: ", X_test[:5])

# Перетворення даних у tenson
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_train, dtype=torch.long)


# Робимо багатошаровий персептрон MLP
class IrisMLP(nn.Module):
    def __init__(self):
        super(IrisMLP, self).__init__()

        #Створюємо шари для мережі
        self.model = nn.Sequential(
            #Вказуємо як будуть розподілятися дані На вхід у нас 4 ознаки
            # з 4 -> 16 ознак
            nn.Linear(4, 16),
            #Використовуємо функцію активації ReLU - щоб забрати не лінійність
            nn.ReLU(),
            # Робимо прохований шар
            # 16 -> 8 нейронів
            nn.Linear(16, 8),
            nn.ReLU(),

            #Завершуємо роботу і робимо вихідний шар
            #У ході роботи нейроної мережі у нас має вийти
            # Опис 3 класів
            # - setosa
            # - versicolor
            # - virginica
            nn.Linear(8, 3),
        )
    #Прямо поширення сигналу у нейроній мережі
    # для кожного x - виконуємо усі алгоритми MLP
    def forward(self, x):
        return self.model(x)

# Робимо об'єкти для MLP
model = IrisMLP()
print("архітектура MLP")
print(model)

# Будуємо фукнцію втрат
# Для того, щоб визначити на скільки прогноз відрізняється від правильної відповіді
criterion = nn.CrossEntropyLoss() # тут ми виконуємо крос ентропію
#Налаштовуємо оптимізматора
# Вкиористовується для зміни Ваги нейроної мережі
# Щоб на наступному етапі можна було врахувати краще вагу і був кращий результат
optimizer = optim.Adam(model.parameters(), # Параметри поделі
                       lr=0.01 # швидкість навчання
                       )
# Вказуємо параметри навчання
epochs = 400 # Кількість епох

# чим менший loss - тим краще модель навчилась
loss_history = [] # масив для виводу результатів loss на різних епорах

# На скільки відсотків правильний прогноз
acc_history = []

for epoch in range(epochs):
    model.train() # Тренуємо нашу модель

    outputs = model(X_train_tensor)  # навчальні дані передаємо в нейрону мережу

    loss = criterion(
        outputs, # порівнюємо передбачення з правильними класами
        y_train_tensor
    )

    optimizer.zero_grad()  # видаляємо градієнт попередньої операції
    loss.backward()  # повертайється нараз по мережі у шукає ваги для карощого прогнозування
    # на основі нових ваг - робить оптимізацію
    optimizer.step()

    # Вишуємо найбільше значення серед рядків і робимо один масив із набільшими значеннями
    predictions = torch.argmax(outputs, 1)
    # Порівняня із првильним значенням
    correct = (predictions == y_train_tensor)
    acc = correct.float().mean() # Знаходить середнє значення правильних відповідей

    loss_history.append(loss.item()) # для візуалізації
    acc_history.append(acc.item()) # для графіків

    if epoch % 20 == 0:
        print(f" - [{epoch+1:3}/{epochs}]")
        print(f" - Loss: {loss.item():.4f}")
        print(f" - Accuracy: {acc.item()*100:.2f}")

# Переводимо модель у режим оцінювання
model.eval()

# Відключаємо обчислення градієнтів
with torch.no_grad():

    # Передаємо тестові дані через мережу - Передати тестові дані через навчену нейронну мережу та отримати її прогнози.
    test_outputs = model(
        X_test_tensor
    )


    # Вибираємо клас з найбільшим значенням
    test_predictions = torch.argmax(
        test_outputs,
        dim=1
    )

# Перетворюємо Tensor у NumPy
predictions_numpy = test_predictions.numpy()


# Обчислюємо Accuracy
test_accuracy = accuracy_score(
    y_test,
    predictions_numpy
)

# Виводимо результат
print("\n================================")

# Виводимо точність
print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print("================================")

# Створюємо звіт класифікації
report = classification_report(
    y_test,                          # Правильні класи
    predictions_numpy,               # Передбачені класи
    target_names=iris.target_names   # Назви класів
)


# Виводимо звіт
print("\nClassification Report:")
# precision - Серед усіх квіток, які модель назвала певним класом, скільки дійсно належать до цього класу?
# recall - Скільки реальних об'єктів певного класу модель змогла знайти?
# F1-score — це показник, який одночасно враховує Precision та Recall.
print(report)




# Обчислюємо матрицю помилок - покаже, які класи шутає між собою модель
cm = confusion_matrix(
    y_test,
    predictions_numpy
)


# Виводимо матрицю в консоль
print("\nConfusion Matrix:")
print(cm)


# Візуалізація матриці
# Створюємо область для графіка
plt.figure(
    figsize=(7, 6)
)


# Відображаємо матрицю як зображення
plt.imshow(
    cm,
    interpolation="nearest"
)


# Додаємо заголовок
plt.title(
    "Confusion Matrix — Iris MLP"
)


# Додаємо назву осі X
plt.xlabel(
    "Передбачений клас"
)


# Додаємо назву осі Y
plt.ylabel(
    "Справжній клас"
)


# Створюємо список позицій класів
classes = [
    "setosa",
    "versicolor",
    "virginica"
]


# Встановлюємо підписи осі X
plt.xticks(
    range(3),
    classes
)


# Встановлюємо підписи осі Y
plt.yticks(
    range(3),
    classes
)


# Додаємо числові значення всередину матриці
for i in range(3):
    # Перебираємо стовпці
    for j in range(3):
        # Виводимо значення конкретної клітинки
        plt.text(
            j,                    # Позиція X
            i,                    # Позиція Y
            cm[i, j],             # Значення
            ha="center",          # Горизонтальне вирівнювання
            va="center"           # Вертикальне вирівнювання
        )
# Автоматично налаштовуємо відступи
plt.tight_layout()


# Показуємо графік
plt.show()

# Створюємо новий графік
plt.figure(
    figsize=(8, 5)
)


# Малюємо зміну Loss
plt.plot(
    loss_history
)


# Назва осі X
plt.xlabel(
    "Epoch"
)


# Назва осі Y
plt.ylabel(
    "Loss"
)


# Заголовок графіка
plt.title(
    "Зміна помилки під час навчання"
)


# Вмикаємо сітку
plt.grid()


# Показуємо графік
plt.show()

# Створюємо новий графік
plt.figure(
    figsize=(8, 5)
)


# Малюємо зміну Accuracy
plt.plot(
    acc_history
)


# Назва осі X
plt.xlabel(
    "Epoch"
)


# Назва осі Y
plt.ylabel(
    "Accuracy"
)


# Заголовок
plt.title(
    "Точність MLP під час навчання"
)


# Вмикаємо сітку
plt.grid()


# Показуємо графік
plt.show()


# Створюємо характеристики нової квітки
sample = [[
    5.1,                              # Довжина чашолистка
    3.5,                              # Ширина чашолистка
    1.4,                              # Довжина пелюстки
    0.2                               # Ширина пелюстки
]]


# Масштабуємо нові дані
# використовуючи scaler, навчений на TRAIN
sample_scaled = scaler.transform(
    sample
)


# Перетворюємо нові дані у Tensor
sample_tensor = torch.tensor(

    sample_scaled,

    dtype=torch.float32
)


# Переводимо модель у режим оцінювання
model.eval()


# Відключаємо градієнти
with torch.no_grad():

    # Передаємо нову квітку в MLP
    output = model(
        sample_tensor
    )


    # Знаходимо індекс найбільш імовірного класу
    prediction = torch.argmax(
        output,
        dim=1
    ).item()

# Виводимо роздільник
print("\n================================")


# Виводимо характеристики квітки
print("Характеристики нової квітки:")
print(sample)

# Виводимо передбачений клас
print("\nПередбачений клас:")

# Отримуємо назву класу за його індексом
print(
    iris.target_names[prediction]
)

# Виводимо роздільник
print("================================")
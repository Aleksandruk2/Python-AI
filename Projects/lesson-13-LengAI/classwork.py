import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import bigrams

# завдання 1

# Створити CSV з 10–15 англомовними відгуками. (reviews.csv)

reviews = [
    "The product is amazing and very easy to use!",
    "I am really happy with the quality of this product.",
    "The application works well, but the interface is confusing.",
    "I expected better performance for the price.",
    "Great customer service and fast delivery.",
    "The product stopped working after two weeks.",
    "I love this product! It saves me a lot of time.",
    "The quality is good and the price is reasonable.",
    "Very disappointing experience. I will not buy it again.",
    "Easy to install and works exactly as expected.",
    "The design is beautiful, but the battery life is poor.",
    "Excellent product and very helpful support team."
]

df_create = pd.DataFrame({
    "review": reviews
})

df_create.to_csv(
    "reviews.csv",
    index=False
)

print("CSV-файл створено: reviews.csv")

# Завантажити його через pandas.
df = pd.read_csv("reviews.csv")

print("\nПерші рядки датасету:")
print(df.head())

# Очистити текст:
# - перевести в нижній регістр;
# - прибрати спецсимволи;
# - прибрати цифри.

def clean_text(text):
    text = text.lower()
    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

df["clean_review"] = df["review"].apply(
    clean_text
)

df["word_count"] = df["clean_review"].apply(
    lambda text: len(text.split())
)

print("\nОброблені дані:")

print(
    df[
        [
            "review",
            "clean_review",
            "word_count"
        ]
    ]
)

# Перевірити:
# - кількість рядків;
# - довжину кожного відгуку в словах.

print("\nКількість рядків:", len(df))

print(
    "Середня довжина відгуку:",
    df["word_count"].mean()
)

print(
    "Мінімальна довжина:",
    df["word_count"].min()
)

print(
    "Максимальна довжина:",
    df["word_count"].max()
)

# завдяння 2

# Завантаження ресурсів NLTK
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Токенізація текстів
df["tokens"] = df["clean_review"].apply(
    word_tokenize
)

# Отримуємо англійські стоп-слова
stop_words = set(
    stopwords.words("english")
)
print("Кількість stop-words:", len(stop_words))

# Видалення stop-words
df["tokens_clean"] = df["tokens"].apply(
    lambda tokens: [
        token
        for token in tokens
        if token not in stop_words
    ]
)

# Видалення токенів довжиною < 3
df["tokens_clean"] = df["tokens_clean"].apply(
    lambda tokens: [
        token
        for token in tokens
        if len(token) >= 3
    ]
)

# Підрахунок загальної кількості токенів ДО та ПІСЛЯ очищення

# Підрахунок кількості токенів ДО очищення
df["tokens_before"] = df["tokens"].apply(
    len
)

# Підрахунок кількості токенів ПІСЛЯ очищення
df["tokens_after"] = df["tokens_clean"].apply(
    len
)

# Загальна кількість токенів ДО
total_tokens_before = df["tokens_before"].sum()

# Загальна кількість токенів ПІСЛЯ
total_tokens_after = df["tokens_after"].sum()


print("\n================================")
print("КІЛЬКІСТЬ ТОКЕНІВ")
print("================================")

print(
    "До очищення:",
    total_tokens_before
)

print(
    "Після очищення:",
    total_tokens_after
)

print("\nПРИКЛАДИ ОБРОБКИ")
for i in range(min(2, len(df))):
    print("\nОригінал:")
    print(df.iloc[i]["clean_review"])

    print("\nТокени:")
    print(df.iloc[i]["tokens"])

    print("\nПісля очищення:")
    print(df.iloc[i]["tokens_clean"])

    print(
        "До:",
        df.iloc[i]["tokens_before"],
        "| Після:",
        df.iloc[i]["tokens_after"]
    )

# завдання 3

# За допомогою Counter порахувати,
# скільки разів зустрічається кожне слово.
all_tokens = []

for tokens in df["tokens_clean"]:
    all_tokens.extend(tokens)

word_counter = Counter(all_tokens)

# Вивести 15 найчастіших слів.

top_15 = word_counter.most_common(15)

print("================================")
print("TOP 15 НАЙЧАСТІШИХ СЛІВ")
print("================================")

words = []
counts = []
top_counter = 1
for word, count in top_15:
    print(f"топ-{top_counter} [ {word}: {count} ]")

    words.append(word)
    counts.append(count)
    top_counter+=1

# print("Розділенні слова та їх кількість:")
# print("words:", words)
# print("counts:", counts)

# Побудувати горизонтальну діаграму.

plt.figure(figsize=(10, 7))

plt.barh(
    words[::-1],
    counts[::-1]
)
plt.xlabel("Частота")
plt.ylabel("Слово")
plt.title(
    "Top 15 Frequent Words"
)
plt.grid(
    axis="x",
    alpha=0.3
)
plt.tight_layout()
plt.show()

# Зберегти її як feedback_word_freq.png.

plt.savefig(
    "feedback_word_freq.png",
    dpi=300,
    bbox_inches="tight"
)

# завдання 4

# Створюємо список усіх біграм

all_bigrams = []

for tokens in df["tokens_clean"]:
    if len(tokens) >= 2:
        review_bigrams = bigrams(tokens)
        all_bigrams.extend(review_bigrams)

# Підраховуємо частоту біграм

bigram_counter = Counter(all_bigrams)

# Отримуємо 10 найчастіших біграм

top_10_bigrams = bigram_counter.most_common(10)

print("================================")
print("TOP 10 НАЙЧАСТІШИХ БІГРАМ")
print("================================")

for bigram, count in top_10_bigrams:
    bigram_text = " ".join(bigram)

    print(
        f"{bigram_text}: {count}"
    )

bigram_data = []

for bigram, count in top_10_bigrams:
    bigram_text = " ".join(bigram)
    bigram_data.append({
        "Біграма": bigram_text,
        "Частота": count
    })

bigram_df = pd.DataFrame(
    bigram_data
)

print("\n\n================================")
print("ТАБЛИЦЯ БІГРАМ")
print("================================")

print(bigram_df)

# Зберігаємо результати у CSV

bigram_df.to_csv(
    "feedback_bigrams.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nРезультати збережено у feedback_bigrams.csv"
)
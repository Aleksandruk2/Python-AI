import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk

from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from collections import Counter
from tabulate import tabulate


# Завантажуємо необхідний ресурс NLTK
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Завдання 1

df = pd.read_csv("reviews.csv")

print("Початкові дані:")
print(df.head())

# Очищаємо текст
df["clean_review"] = (
    df["review"]
    .str.lower()
    .str.replace(r"[^a-zA-Z\s]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Токенізація
stop_words = set(stopwords.words("english"))

df["tokens"] = df["clean_review"].apply(
    lambda text: [
        word
        for word in word_tokenize(text)
        if word not in stop_words
    ]
)

print("\nОчищений текст:")
print(df[["review", "clean_review"]].head())

print("\nТокени:")
print(df["tokens"].head())

tokenized_sentences = df["tokens"].tolist()

print("\nТокенізовані речення:")
print(tokenized_sentences[:5])

# завдання 2

# Створюємо та навчаємо модель Word2Vec
word2vec_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1
)

# Виводимо розмір словника
print(
    "\nКількість слів у словнику:",
    len(word2vec_model.wv)
)

# Підраховуємо частоту слів
word_counts = Counter(
    word
    for sentence in tokenized_sentences
    for word in sentence
)

# Визначаємо 10 ключових слів
key_words = [
    word
    for word, count in word_counts.most_common(10)
    if word in word2vec_model.wv
]

print("\n10 ключових слів:")
print(key_words)

# Знаходимо найближчі слова
results = []

for word in key_words:
    similar_words = word2vec_model.wv.most_similar(
        word,
        topn=5
    )

    words = [
        item[0]
        for item in similar_words
    ]

    similarities = [
        round(item[1], 3)
        for item in similar_words
    ]

    results.append({
        "Ключове слово": word,
        "Близькі слова": ", ".join(words),
        "Косинусна схожість": ", ".join(
            map(str, similarities)
        )
    })

# Створюємо таблицю
results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("                 WORD2VEC — СЕМАНТИЧНА БЛИЗЬКІСТЬ СЛІВ")
print("=" * 100)

print(
    tabulate(
        results_df,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    )
)

print("=" * 100)

# Завдання 3

# Отримуємо всі слова зі словника Word2Vec
words = word2vec_model.wv.index_to_key

# Отримуємо вектори всіх слів
word_vectors = word2vec_model.wv.vectors

print("\n" + "=" * 65)
print("                         КЛАСТЕРИЗАЦІЯ СЛІВ")
print("=" * 65)

print(f"Кількість слів для кластеризації: {len(words)}")
print(f"Розмірність векторів: {word_vectors.shape[1]}")

# Створюємо та навчаємо KMeans
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Визначаємо кластер для кожного слова
cluster_labels = kmeans.fit_predict(word_vectors)

# Створюємо таблицю зі словами та їх кластерами
clusters_df = pd.DataFrame({
    "Слово": words,
    "Кластер": cluster_labels
})

# Знаходимо 10 найближчих до центру слів
# для кожного кластера
cluster_results = []

for cluster_id in range(5):
    # Індекси слів, які належать до цього кластера
    cluster_indices = np.where(
        cluster_labels == cluster_id
    )[0]

    # Центр кластера
    center = kmeans.cluster_centers_[cluster_id]

    # Вектори слів поточного кластера
    cluster_vectors = word_vectors[cluster_indices]

    # Евклідова відстань кожного слова
    # до центру його кластера
    distances = np.linalg.norm(
        cluster_vectors - center,
        axis=1
    )

    # Сортуємо індекси за відстанню до центру
    sorted_positions = np.argsort(distances)

    # Беремо 10 найближчих до центру слів
    top_indices = cluster_indices[
        sorted_positions[:10]
    ]

    top_words = [
        words[i]
        for i in top_indices
    ]

    cluster_results.append({
        "Кластер": cluster_id + 1,
        "Кількість слів": len(cluster_indices),
        "10 найбільш характерних слів": ", ".join(top_words)
    })


# Створюємо таблицю результатів
cluster_results_df = pd.DataFrame(cluster_results)

print(
    tabulate(
        cluster_results_df,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    )
)

print("\n" + "=" * 65)
print("                    СМИСЛОВІ ГРУПИ КЛАСТЕРІВ")
print("=" * 65)

print(
    """
Смислова група визначається на основі найбільш характерних
слів кожного кластера.

Для цього аналізуємо 10 слів, найближчих до центру кожного
кластера, та визначаємо спільну тему, яку вони представляють.
"""
)

for _, row in cluster_results_df.iterrows():

    print(f"\nКластер {row['Кластер']}:")
    print(f"Характерні слова: {row['10 найбільш характерних слів']}")

# Завдання 4

print("\n" + "=" * 65)
print("                 ВІЗУАЛІЗАЦІЯ СЕМАНТИЧНИХ КЛАСТЕРІВ")
print("=" * 65)

# Зменшуємо розмірність векторів Word2Vec з 100 до 2
pca = PCA(
    n_components=2,
    random_state=42
)

word_vectors_2d = pca.fit_transform(word_vectors)

print(f"Початкова розмірність векторів: {word_vectors.shape[1]}")
print(f"Розмірність після PCA: {word_vectors_2d.shape[1]}")

# Створюємо графік
plt.figure(figsize=(16, 11))

# Кількість кластерів
n_clusters = 5

for cluster_id in range(n_clusters):
    cluster_indices = np.where(
        cluster_labels == cluster_id
    )[0]

    x = word_vectors_2d[cluster_indices, 0]
    y = word_vectors_2d[cluster_indices, 1]

    plt.scatter(
        x,
        y,
        s=45,
        alpha=0.7,
        label=f"Кластер {cluster_id + 1}"
    )

    for index in cluster_indices:

        plt.annotate(
            words[index],
            (
                word_vectors_2d[index, 0],
                word_vectors_2d[index, 1]
            ),
            fontsize=8,
            alpha=0.8,
            xytext=(4, 4),
            textcoords="offset points"
        )

plt.title(
    "Семантичні кластери слів Word2Vec",
    fontsize=16,
    pad=15
)

plt.xlabel(
    "Головна компонента 1 (PCA)",
    fontsize=12
)

plt.ylabel(
    "Головна компонента 2 (PCA)",
    fontsize=12
)

plt.legend(
    title="Кластери",
    fontsize=10,
    title_fontsize=11
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "semantic_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nГрафік збережено у файл: semantic_clusters.png")

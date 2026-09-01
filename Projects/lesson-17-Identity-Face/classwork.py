import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise import accuracy


# Завдання 1
books = pd.read_csv(
    "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
)

ratings = pd.read_csv(
    "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
)


books = books.dropna()
ratings = ratings.dropna()

books = books.drop_duplicates()
ratings = ratings.drop_duplicates()

books = books[["book_id", "title", "authors"]]
ratings = ratings[["book_id", "user_id", "rating"]]

df = pd.merge(
    books,
    ratings,
    on="book_id"
)

books_df = df.drop_duplicates(subset=["book_id"]).reset_index(drop=True)

print(books_df.head())
print(books_df.shape)


# Завдання 2

books_df["text"] = df["title"] + " " + df["authors"]


tfidf = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(books_df["text"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title, n=5):
    idx = df[df["title"] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:n + 1]

    book_indices = [i[0] for i in sim_scores]

    return books_df.iloc[book_indices][["book_id", "title", "authors"]]

recommendations = recommend(
    "The Hunger Games (The Hunger Games, #1)",
    n=5
)

print(recommendations)

recommendations.to_csv(
    "content_recommendations.csv",
    index=False
)


# Завдання 3

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    ratings[["user_id", "book_id", "rating"]],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

model = SVD(random_state=42)
model.fit(trainset)

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

print(f"RMSE: {rmse:.4f}")

with open("svd_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Модель збережено у svd_model.pkl")


# Завдання 4

def recommend_cf(user_id, n=5):
    all_book_ids = books["book_id"].unique()

    rated_books = ratings[
        ratings["user_id"] == user_id
    ]["book_id"].values

    books_to_predict = [
        book_id
        for book_id in all_book_ids
        if book_id not in rated_books
    ]

    predictions = [
        (book_id, model.predict(user_id, book_id).est)
        for book_id in books_to_predict
    ]

    predictions = sorted(
        predictions,
        key=lambda x: x[1],
        reverse=True
    )

    top_predictions = predictions[:n]

    book_ids = [x[0] for x in top_predictions]

    result = books[
        books["book_id"].isin(book_ids)
    ][["book_id", "title", "authors"]].copy()

    prediction_dict = dict(top_predictions)

    result["predicted_rating"] = result["book_id"].map(
        prediction_dict
    )

    result = result.sort_values(
        "predicted_rating",
        ascending=False
    )

    return result

user_ids = [1, 2, 3]

cf_results = []

for user_id in user_ids:
    recommendations = recommend_cf(user_id, n=5)

    recommendations["user_id"] = user_id

    cf_results.append(recommendations)

    print(f"\nРекомендації для користувача {user_id}:")
    print(recommendations)

cf_results_df = pd.concat(
    cf_results,
    ignore_index=True
)

cf_results_df.to_csv(
    "cf_results.csv",
    index=False
)

print("\nРезультати збережено у cf_results.csv")
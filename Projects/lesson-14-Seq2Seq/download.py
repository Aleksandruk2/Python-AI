import json

# Назви файлів
en_file = "Tatoeba.en-uk.en"
uk_file = "Tatoeba.en-uk.uk"

# Читаємо англійські речення
with open(en_file, "r", encoding="utf-8") as f:
    en_sentences = [line.strip() for line in f]

# Читаємо українські речення
with open(uk_file, "r", encoding="utf-8") as f:
    uk_sentences = [line.strip() for line in f]

# Перевіряємо, що кількість рядків однакова
if len(en_sentences) != len(uk_sentences):
    raise ValueError(
        f"Кількість речень не збігається: "
        f"EN = {len(en_sentences)}, UK = {len(uk_sentences)}"
    )

# Об'єднуємо у формат:
# [
#     ["англійське речення", "українське речення"],
#     ...
# ]
data = [
    [en, uk]
    for en, uk in zip(en_sentences, uk_sentences)
]

# Зберігаємо у JSON
with open("Tatoeba.en-uk.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Готово!")
print(f"Кількість пар: {len(data)}")
print("Файл: Tatoeba.en-uk.json")
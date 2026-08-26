import re
import time
import torch
import torch.nn as nn

import pandas as pd
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


# ЗАВДАННЯ 1

# 25 пар "запитання — відповідь"
data = [
    (
        "How do I change my email?",
        "You can update your email in your profile settings."
    ),
    (
        "How do I reset my password?",
        "You can reset your password using the password reset link."
    ),
    (
        "I forgot my password.",
        "Please use the password reset link to create a new password."
    ),
    (
        "How can I change my password?",
        "You can change your password in the account settings."
    ),
    (
        "How do I update my profile?",
        "You can update your profile information in account settings."
    ),
    (
        "How do I delete my account?",
        "You can request account deletion from the account settings."
    ),
    (
        "Can I create a new account?",
        "Yes, you can create a new account using the registration page."
    ),
    (
        "I cannot log into my account.",
        "Please check your email and password and try again."
    ),
    (
        "Why can I not log in?",
        "Please verify your login details and reset your password if necessary."
    ),
    (
        "How do I contact support?",
        "You can contact our support team through the help center."
    ),
    (
        "Where can I find the help center?",
        "The help center is available from the support section of the website."
    ),
    (
        "How do I change my username?",
        "You can change your username in your profile settings."
    ),
    (
        "Can I change my account information?",
        "Yes, you can edit your account information in profile settings."
    ),
    (
        "How do I update my phone number?",
        "You can update your phone number in your profile settings."
    ),
    (
        "Why has my account been locked?",
        "Your account may be locked after several failed login attempts."
    ),
    (
        "How can I unlock my account?",
        "Please wait and try again or contact support for assistance."
    ),
    (
        "How do I change notification settings?",
        "You can change notification settings in your account preferences."
    ),
    (
        "How do I turn off notifications?",
        "You can disable notifications in your account preferences."
    ),
    (
        "Where can I see my account settings?",
        "You can find account settings from your profile menu."
    ),
    (
        "How do I update my profile picture?",
        "You can change your profile picture from your profile settings."
    ),
    (
        "What should I do if I have a problem?",
        "Please contact our support team through the help center."
    ),
    (
        "How can I get help?",
        "You can get help by contacting our support team."
    ),
    (
        "How do I change my language?",
        "You can change the language in your account preferences."
    ),
    (
        "Can I update my personal details?",
        "Yes, you can update your personal details in your profile settings."
    ),
    (
        "How do I sign out?",
        "You can sign out using the logout option in your profile menu."
    )
]


# Очищення тексту
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Токенізація
def tokenize(text):
    return text.split()


# Очищаємо та токенізуємо всі запитання і відповіді
tokenized_data = []

for question, answer in data:
    question_clean = clean_text(question)
    answer_clean = clean_text(answer)

    question_tokens = tokenize(question_clean)
    answer_tokens = tokenize(answer_clean)

    tokenized_data.append(
        (question_tokens, answer_tokens)
    )

print(f"Кількість пар запитання-відповідь: {len(tokenized_data)}")

print("\nПриклад після очищення:")
print("Запитання:")
print(" ".join(tokenized_data[0][0]))
print("\nВідповідь:")
print(" ".join(tokenized_data[0][1]))



sequences = []

for question_tokens, answer_tokens in tokenized_data:
    sequence = (
        ["<bos>"]
        + question_tokens
        + ["<answer>"]
        + answer_tokens
        + ["<eos>"]
    )
    sequences.append(sequence)

special_tokens = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<answer>",
    "<eos>"
]

vocab = {}

for token in special_tokens:
    vocab[token] = len(vocab)

for sequence in sequences:
    for token in sequence:
        if token not in vocab:
            vocab[token] = len(vocab)


# Зворотний словник index -> token
idx_to_token = {
    index: token
    for token, index in vocab.items()
}
print(f"Розмір словника: {len(vocab)}")

print("\nПриклади token -> index:")
for token, index in list(vocab.items())[:20]:
    print(f"{token:15} -> {index}")


# Перетворюємо токени у числові індекси
indexed_sequences = []

for sequence in sequences:
    indexed_sequence = [
        vocab.get(token, vocab["<unk>"])
        for token in sequence
    ]
    indexed_sequences.append(indexed_sequence)


# Визначаємо максимальну довжину послідовності
max_length = max(
    len(sequence)
    for sequence in indexed_sequences
)

print(f"Максимальна довжина послідовності: {max_length}")


# Padding
pad_index = vocab["<pad>"]

padded_sequences = []

for sequence in indexed_sequences:
    padded_sequence = sequence + [
        pad_index
    ] * (max_length - len(sequence))
    padded_sequences.append(padded_sequence)

print("\nПриклад послідовності після padding:")
print(padded_sequences[0])


# ЗАВДАННЯ 2

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(
            sequences,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]

        x = sequence[:-1]
        y = sequence[1:]

        return x, y

dataset = TextDataset(padded_sequences)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_size=128,
        num_layers=1,
        padding_idx=0
    ):
        super().__init__()

        # Вхідний шар Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Повнозв'язний шар
        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output)

        return output

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = LSTMModel(
    vocab_size=len(vocab),
    embedding_dim=64,
    hidden_size=128,
    num_layers=1,
    padding_idx=pad_index
)

model = model.to(device)

print('model:',model)
print(f"\nПристрій: {device}")

criterion = nn.CrossEntropyLoss(
    ignore_index=pad_index
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in dataloader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        output = output.reshape(
            -1,
            len(vocab)
        )

        y = y.reshape(-1)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(dataloader)
    )

    print(
        f"Епоха {epoch + 1:2d}/{epochs} "
        f"| Loss: {average_loss:.4f}"
    )

def generate_answer(
    question,
    max_new_tokens=20
):
    model.eval()

    question = clean_text(question)

    tokens = (
        ["<bos>"]
        + tokenize(question)
        + ["<answer>"]
    )

    indices = [
        vocab.get(
            token,
            vocab["<unk>"]
        )
        for token in tokens
    ]

    generated = indices.copy()

    with torch.no_grad():

        for _ in range(max_new_tokens):
            input_indices = generated[
                -max_length + 1:
            ]

            x = torch.tensor(
                [input_indices],
                dtype=torch.long
            ).to(device)

            output = model(x)

            next_token_logits = output[
                0,
                -1
            ]

            next_token = torch.argmax(
                next_token_logits
            ).item()

            generated.append(next_token)

            if next_token == vocab["<eos>"]:
                break

    generated_tokens = [
        idx_to_token[index]
        for index in generated
    ]

    if "<answer>" in generated_tokens:
        answer_start = (
            generated_tokens.index("<answer>")
            + 1
        )
        answer_tokens = generated_tokens[
            answer_start:
        ]
    else:
        answer_tokens = generated_tokens

    answer_tokens = [
        token
        for token in answer_tokens
        if token not in [
            "<bos>",
            "<answer>",
            "<pad>",
            "<eos>"
        ]
    ]

    return " ".join(answer_tokens)


test_questions = [
    "How do I change my email?",
    "How do I reset my password?",
    "How do I contact support?",
    "How do I update my profile?"
]

for question in test_questions:

    answer = generate_answer(
        question,
        max_new_tokens=20
    )

    print(f"\nЗапит: {question}")
    print(f"Відповідь: {answer}")


# ЗАВДАННЯ 3


# 5 тестових запитів
test_questions = [
    "How do I reset my password?",
    "How do I change my email?",
    "How do I update my profile?",
    "How do I contact support?",
    "How do I change my language?"
]


# Функція простого оцінювання відповіді
def evaluate_answer(question, answer):
    question_words = set(
        tokenize(clean_text(question))
    )

    answer_words = set(
        tokenize(clean_text(answer))
    )

    common_words = (
        question_words & answer_words
    )

    if len(question_words) > 0:
        relevance = (
            len(common_words)
            / len(question_words)
        )
    else:
        relevance = 0

    answer_length = len(
        tokenize(clean_text(answer))
    )

    if answer_length == 0:
        comment = "Відповідь відсутня"

    elif relevance >= 0.4:
        comment = "Відповідь відповідає запитанню"

    elif relevance >= 0.2:
        comment = "Частково відповідає запитанню"

    else:
        comment = "Відповідь слабко пов'язана із запитом"

    return relevance, answer_length, comment


# Генеруємо відповіді LSTM
lstm_results = []

for question in test_questions:
    start_time = time.perf_counter()

    answer = generate_answer(
        question,
        max_new_tokens=20
    )

    generation_time = (
        time.perf_counter()
        - start_time
    )

    relevance, answer_length, comment = (
        evaluate_answer(
            question,
            answer
        )
    )

    lstm_results.append({
        "Запит": question,
        "Відповідь моделі": answer,
        "Відповідність": round(
            relevance,
            2
        ),
        "Довжина": answer_length,
        "Час, с": round(
            generation_time,
            4
        ),
        "Коментар": comment
    })


# Таблиця результатів
lstm_results_df = pd.DataFrame(
    lstm_results
)

print(
    tabulate(
        lstm_results_df,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    )
)


# Загальні показники LSTM
lstm_avg_relevance = (
    lstm_results_df["Відповідність"].mean()
)

lstm_avg_length = (
    lstm_results_df["Довжина"].mean()
)

lstm_avg_time = (
    lstm_results_df["Час, с"].mean()
)

print("\nСередні показники LSTM:")
print(
    f"Середня відповідність: "
    f"{lstm_avg_relevance:.2f}"
)

print(
    f"Середня довжина відповіді: "
    f"{lstm_avg_length:.2f} токенів"
)

print(
    f"Середній час генерації: "
    f"{lstm_avg_time:.4f} с"
)


# ЗАВДАННЯ 4

# Завантажуємо distilGPT2
model_name = "distilgpt2"

print("\n1. Починаємо завантаження tokenizer...", flush=True)

transformer_tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

print("2. Tokenizer завантажено!", flush=True)

print("3. Починаємо завантаження model...", flush=True)

transformer_model = AutoModelForCausalLM.from_pretrained(
    model_name
)

print("4. Model завантажено!", flush=True)

transformer_model = transformer_model.to(device)

print("5. Transformer готовий!", flush=True)

# distilGPT2 не має окремого PAD-токена,
# тому використовуємо EOS як PAD
transformer_tokenizer.pad_token = (
    transformer_tokenizer.eos_token
)

transformer_model.config.pad_token_id = (
    transformer_tokenizer.pad_token_id
)

print("Transformer завантажено.")


# Функція генерації відповіді Transformer
def generate_transformer_answer(
    question,
    max_new_tokens=30
):
    prompt = (
        "Customer question: "
        + question
        + "\nSupport answer:"
    )

    inputs = transformer_tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = transformer_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=(
                transformer_tokenizer.pad_token_id
            )
        )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = transformer_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# Генеруємо відповіді Transformer
transformer_results = []

for question in test_questions:
    start_time = time.perf_counter()

    answer = generate_transformer_answer(
        question,
        max_new_tokens=30
    )

    generation_time = (
        time.perf_counter()
        - start_time
    )

    relevance, answer_length, comment = (
        evaluate_answer(
            question,
            answer
        )
    )

    transformer_results.append({
        "Запит": question,
        "Відповідь моделі": answer,
        "Відповідність": round(
            relevance,
            2
        ),
        "Довжина": answer_length,
        "Час, с": round(
            generation_time,
            4
        ),
        "Коментар": comment
    })


# Таблиця результатів Transformer
transformer_results_df = pd.DataFrame(
    transformer_results
)

print(
    tabulate(
        transformer_results_df,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    )
)


# Загальні показники Transformer
transformer_avg_relevance = (
    transformer_results_df[
        "Відповідність"
    ].mean()
)

transformer_avg_length = (
    transformer_results_df[
        "Довжина"
    ].mean()
)

transformer_avg_time = (
    transformer_results_df[
        "Час, с"
    ].mean()
)

print("\nСередні показники Transformer:")

print(
    f"Середня відповідність: "
    f"{transformer_avg_relevance:.2f}"
)

print(
    f"Середня довжина відповіді: "
    f"{transformer_avg_length:.2f} токенів"
)

print(
    f"Середній час генерації: "
    f"{transformer_avg_time:.4f} с"
)

comparison_df = pd.DataFrame({
    "Критерій": [
        "Якість відповідей",
        "Збереження контексту",
        "Час генерації",
        "Логічність і зв'язність"
    ],

    "LSTM": [
        f"Відповідність: "
        f"{lstm_avg_relevance:.2f}",
        "Обмежена малим корпусом",
        f"{lstm_avg_time:.4f} с",
        "Залежить від якості навчання"
    ],

    "Трансформер": [
        f"Відповідність: "
        f"{transformer_avg_relevance:.2f}",
        "Краще зберігає контекст",
        f"{transformer_avg_time:.4f} с",
        "Зазвичай більш зв'язні відповіді"
    ]
})


print("\n" + "=" * 70)
print("ПОРІВНЯННЯ LSTM І TRANSFORMER")
print("=" * 70)

print(
    tabulate(
        comparison_df,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    )
)

# Зберігаємо LSTM
torch.save(
    model.state_dict(),
    "supportflow_lstm.pth"
)

print(
    "LSTM збережено: "
    "supportflow_lstm.pth"
)


# Зберігаємо Transformer
torch.save(
    transformer_model.state_dict(),
    "supportflow_gpt2.pth"
)

print(
    "Transformer збережено: "
    "supportflow_gpt2.pth"
)
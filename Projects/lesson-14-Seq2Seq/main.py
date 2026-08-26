import json
import random
import re # для регулярних виразів
import torch
import torch.nn as nn # Для нейроних мереж
import torch.optim as optim # Оптимізатори для навчання мережі

torch.manual_seed(42) # це рандом для PyTorch
random.seed(42) # щоб дані в рандом не змінювались

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Divece:",DEVICE)

# with open("pairs.json", "r", encoding="utf-8") as f:
#     raw_pairs = json.load(f)

with open("Tatoeba.en-uk.json", "r", encoding="utf-8") as f:
    raw_pairs = json.load(f)

print("Кількість пар слів",len(raw_pairs))
print("Приклад пари", raw_pairs[:3])

def tokenize(sentence: str):
    sentence = re.sub(r"[^a-zа-яіїєґ' ]+", " ", sentence.lower())
    return sentence.split() # повертає самі слова в тексті. Забрали усе лишнє

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>" # токени для роботи моделі
SPECIALS = [PAD, SOS, EOS, UNK] # у вигляді одного списку


class Vocab:
    def __init__(self, sentences: str):  # набір речень
        words = sorted({w for s in sentences for w in tokenize(s)})
        self.itos = SPECIALS + words  # Набір слів у словнику
        self.stoi = {w: i for i, w in enumerate(self.itos)}  # номерований набір із словами

    def __len__(self):
        return len(self.itos)

    def encoder(self, sentence, max_len):
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in tokenize(sentence)]  # Перетворюємо кожне слово на його індекс
        ids = ids[: max_len - 2]  # Обмежуємо кількість слів, залишаючи місце для SOS та EOS
        ids = [self.stoi[SOS]] + ids + [self.stoi[EOS]]  # Додаємо токени початку та кінця речення
        ids += [self.stoi[PAD]] * (max_len - len(ids))  # Доповнюємо коротке речення токенами PAD
        return ids  # Повертаємо готову послідовність індексів

    def decode(self, ids):  # Функція перетворює числові індекси назад у слова
        words = []  # Створюємо порожній список для слів
        for i in ids:  # Перебираємо всі індекси
            w = self.itos[i]  # Отримуємо слово за його індексом
            if w == EOS:  # Перевіряємо, чи зустріли кінець речення
                break  # Якщо зустріли EOS, припиняємо обробку
            if w not in (PAD, SOS):  # Ігноруємо службові токени PAD та SOS
                words.append(w)  # Додаємо звичайне слово до результату
        return " ".join(words)  # Об'єднуємо слова через пробіл і повертаємо речення

eng_sentences = [p[0] for p in raw_pairs] # Англійські речення
ukr_sentences = [p[1] for p in raw_pairs] # Українські речення

src_vocab = Vocab(eng_sentences) # Вхідний словник
trg_vocab = Vocab(ukr_sentences) # Вихідний словник

print(f"Розмір Англійського словника {len(src_vocab)}\nРозмір Українського словника {len(trg_vocab)}")

MAX_LEN = 10 # максимальна довжина речень

data = [] # Зберігає тензори для нейронної мережі

for eng, ukr in raw_pairs:
    src_ids = src_vocab.encoder(eng, MAX_LEN) # отримуємо індекси слів у словнику
    trg_ids = trg_vocab.encoder(ukr, MAX_LEN) # отримуємо індекси слів у словнику
    data.append((torch.tensor(src_ids), torch.tensor(trg_ids)))

random.shuffle(data) # перемішали дан, які у нас є
split = int(len(data) * 0.8)
train_data = data[:split] # для навчання
valid_data = data[split:] # для тесту

def get_batches(dataset, batch_size=32, shuffle=True):  # Створюємо функцію для формування пакетів даних
    idx = list(range(len(dataset)))  # Створюємо список індексів усіх елементів датасету
    if shuffle:  # Перевіряємо, чи потрібно перемішувати дані
        random.shuffle(idx)  # Перемішуємо індекси
    for i in range(0, len(idx), batch_size):  # Проходимо по індексах блоками розміру batch_size
        batch = [dataset[j] for j in idx[i : i + batch_size]]  # Формуємо один batch
        src = torch.stack([b[0] for b in batch]).to(DEVICE)  # Об'єднуємо англійські речення в один тензор і переносимо на CPU/GPU
        trg = torch.stack([b[1] for b in batch]).to(DEVICE)  # Об'єднуємо українські речення в один тензор і переносимо на CPU/GPU
        yield src, trg  # Повертаємо один пакет даних

EMB_DIM = 128  # Встановлюємо розмірність векторного представлення слова
HID_DIM = 256  # Встановлюємо розмір прихованого стану GRU


class Encoder(nn.Module):  # Створюємо клас Encoder на основі PyTorch
    """Читає англійське речення і стискає його у вектор прихованого стану."""  # Опис роботи Encoder

    def __init__(self, vocab_size, emb_dim, hid_dim):  # Конструктор Encoder отримує розмір словника та розміри шарів
        super().__init__()  # Викликаємо конструктор батьківського класу nn.Module
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)  # Створюємо Embedding для перетворення індексів слів у вектори
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)  # Створюємо GRU, яка обробляє послідовність слів

    def forward(self, src):  # Описуємо прямий прохід Encoder
        # src: (batch, seq_len)  # src містить числові індекси слів
        embedded = self.embedding(src)  # Перетворюємо індекси слів у вектори розмірності emb_dim
        outputs, hidden = self.gru(embedded)  # Передаємо послідовність через GRU та отримуємо прихований стан
        return hidden  # Повертаємо прихований стан, який містить інформацію про все речення


class Decoder(nn.Module):  # Створюємо клас Decoder
    """Генерує українське речення слово за словом."""  # Опис роботи Decoder

    def __init__(self, vocab_size, emb_dim, hid_dim):  # Конструктор Decoder
        super().__init__()  # Викликаємо конструктор nn.Module
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)  # Створюємо Embedding для українських слів
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)  # Створюємо GRU для генерації послідовності
        self.fc_out = nn.Linear(hid_dim, vocab_size)  # Створюємо повнозв'язний шар для прогнозування наступного слова

    def forward(self, input_token, hidden):  # Описуємо один крок роботи Decoder
        # input_token: (batch, 1)  # На вході одне слово для кожного речення batch
        embedded = self.embedding(input_token)  # Перетворюємо індекс слова у вектор
        output, hidden = self.gru(embedded, hidden)  # Передаємо слово та прихований стан у GRU
        prediction = self.fc_out(output.squeeze(1))  # Перетворюємо прихований стан у ймовірності наступного слова
        return prediction, hidden  # Повертаємо прогноз і новий прихований стан


class Seq2Seq(nn.Module):  # Створюємо загальну Seq2Seq модель
    def __init__(self, encoder, decoder, trg_vocab_size, device):  # Конструктор Seq2Seq
        super().__init__()  # Викликаємо конструктор nn.Module
        self.encoder = encoder  # Зберігаємо Encoder
        self.decoder = decoder  # Зберігаємо Decoder
        self.trg_vocab_size = trg_vocab_size  # Зберігаємо розмір українського словника
        self.device = device  # Зберігаємо пристрій CPU або GPU

    def forward(self, src, trg, teacher_forcing_ratio=0.5):  # Виконуємо прямий прохід усієї Seq2Seq моделі
        batch_size, trg_len = trg.shape  # Отримуємо розмір пакета та довжину цільового речення
        outputs = torch.zeros(batch_size, trg_len, self.trg_vocab_size).to(self.device)  # Створюємо тензор для збереження прогнозів

        hidden = self.encoder(src)  # Передаємо англійське речення через Encoder
        input_token = trg[:, 0].unsqueeze(1)  # Першим словом Decoder отримує спеціальний токен SOS

        for t in range(1, trg_len):  # Генеруємо українське речення слово за словом
            pred, hidden = self.decoder(input_token, hidden)  # Decoder прогнозує наступне слово
            outputs[:, t] = pred  # Зберігаємо прогноз у відповідній позиції
            teacher_force = random.random() < teacher_forcing_ratio  # Випадково вирішуємо, чи використовувати правильне слово
            top1 = pred.argmax(1)  # Вибираємо слово з найбільшою прогнозованою ймовірністю
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)  # Використовуємо правильне або передбачене слово

        return outputs  # Повертаємо всі прогнози Decoder

    @torch.no_grad()  # Вимикаємо обчислення градієнтів під час перекладу
    def translate(self, sentence, src_vocab, trg_vocab, max_len=MAX_LEN):  # Створюємо функцію перекладу нового речення
        self.eval()  # Переводимо модель у режим оцінювання
        src_ids = torch.tensor([src_vocab.encoder(sentence, max_len)]).to(self.device)  # Перетворюємо вхідне речення на тензор
        hidden = self.encoder(src_ids)  # Передаємо англійське речення через Encoder

        input_token = torch.tensor([[trg_vocab.stoi[SOS]]]).to(self.device)  # Починаємо генерацію з токена SOS
        result_ids = []  # Створюємо список для збереження прогнозованих слів

        for _ in range(max_len):  # Максимум max_len разів генеруємо наступне слово
            pred, hidden = self.decoder(input_token, hidden)  # Decoder прогнозує наступне слово
            next_id = pred.argmax(1).item()  # Отримуємо індекс слова з найбільшою ймовірністю
            if next_id == trg_vocab.stoi[EOS]:  # Перевіряємо, чи модель згенерувала кінець речення
                break  # Якщо отримали EOS, припиняємо генерацію
            result_ids.append(next_id)  # Додаємо передбачене слово до результату
            input_token = torch.tensor([[next_id]]).to(self.device)  # Передаємо передбачене слово на наступний крок

        self.train()  # Повертаємо модель у режим навчання
        return " ".join(trg_vocab.itos[i] for i in result_ids)  # Перетворюємо індекси назад у слова

encoder = Encoder(len(src_vocab), EMB_DIM, HID_DIM).to(DEVICE)  # Створюємо Encoder
decoder = Decoder(len(trg_vocab), EMB_DIM, HID_DIM).to(DEVICE)  # Створюємо Decoder
model = Seq2Seq(encoder, decoder, len(trg_vocab), DEVICE).to(DEVICE)  # Об'єднуємо Encoder і Decoder у Seq2Seq модель

n_params = sum(p.numel() for p in model.parameters())  # Підраховуємо загальну кількість параметрів нейронної мережі
print(f"Кількість параметрів моделі: {n_params:,}")  # Виводимо кількість параметрів моделі


optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Створюємо оптимізатор Adam зі швидкістю навчання 0.001
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Створюємо функцію втрат ігноруючи PAD-токени

N_EPOCHS = 30  # Встановлюємо кількість епох навчання

TEST_SENTENCES = [  # Створюємо список речень для перевірки моделі
    "the cat is here",  # Тестове англійське речення
    "i love you",  # Тестове англійське речення
    "she is my friend",  # Тестове англійське речення
    "this is a good idea",  # Тестове англійське речення
]  # Завершуємо список тестових речень

print("\n--- Початок навчання ---")  # Виводимо повідомлення про початок навчання

for epoch in range(1, N_EPOCHS + 1):  # Запускаємо цикл навчання на 30 епох
    model.train()  # Переводимо модель у режим навчання
    total_loss = 0.0  # Створюємо змінну для накопичення загальної похибки
    n_batches = 0  # Створюємо лічильник кількості batch

    for src, trg in get_batches(train_data, batch_size=32):  # Отримуємо навчальні дані пакетами по 32 речення
        optimizer.zero_grad()  # Обнуляємо старі градієнти
        output = model(src, trg, teacher_forcing_ratio=0.5)  # Передаємо дані через Seq2Seq модель
        output_flat = output[:, 1:].reshape(-1, output.shape[-1])  # Вирівнюємо прогноз для обчислення CrossEntropyLoss
        trg_flat = trg[:, 1:].reshape(-1)  # Вирівнюємо правильні відповіді
        loss = criterion(output_flat, trg_flat)  # Обчислюємо помилку моделі
        loss.backward()  # Обчислюємо градієнти
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Обмежуємо величину градієнтів для стабільності навчання
        optimizer.step()  # Оновлюємо параметри моделі
        total_loss += loss.item()  # Додаємо поточну похибку до загальної похибки
        n_batches += 1  # Збільшуємо кількість оброблених batch на один

    avg_loss = total_loss / n_batches  # Обчислюємо середню похибку за епоху

    if epoch % 5 == 0 or epoch == 1:  # Перевіряємо модель на першій та кожній п'ятій епосі
        print(f"\nЕпоха {epoch:2d}/{N_EPOCHS} | середня похибка (loss): {avg_loss:.3f}")  # Виводимо номер епохи та loss

        for s in TEST_SENTENCES:  # Перебираємо тестові речення
            translation = model.translate(s, src_vocab, trg_vocab)  # Отримуємо переклад тестового речення
            print(f"   '{s}' -> '{translation}'")  # Виводимо оригінал та переклад
    else:  # Якщо це не перша і не кожна п'ята епоха
        print(f"Епоха {epoch:2d}/{N_EPOCHS} | loss: {avg_loss:.3f}")  # Просто виводимо номер епохи та loss


print("\n--- Приклади перекладу на реченнях, яких модель НЕ бачила ---")  # Виводимо заголовок фінальної перевірки

random.shuffle(valid_data)  # Перемішуємо валідаційні дані

for src_ids, trg_ids in valid_data[:8]:  # Беремо перші 8 речень із валідаційного набору
    eng = src_vocab.decode(src_ids.tolist())  # Перетворюємо індекси англійського речення назад у текст
    ukr_true = trg_vocab.decode(trg_ids.tolist())  # Отримуємо правильний український переклад
    ukr_pred = model.translate(eng, src_vocab, trg_vocab)  # Отримуємо переклад від моделі
    print(f"EN: {eng}")  # Виводимо англійське речення
    print(f"   очікувалось: {ukr_true}")  # Виводимо правильний переклад
    print(f"   переклад:    {ukr_pred}\n")  # Виводимо переклад, отриманий моделлю

torch.save(model.state_dict(), "seq2seq_model.pt")  # Зберігаємо навчені параметри моделі у файл

print("Модель збережено у файл seq2seq_model.pt")  # Повідомляємо, що модель
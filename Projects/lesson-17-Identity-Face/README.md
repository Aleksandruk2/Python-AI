## Нейронні мережі
```
python --version
py -m venv .venv

.venv\Scripts\activate.bat
source .venv/bin/activate
python.exe -m pip install --upgrade pip
python3 -m pip install --upgrade pip

pip install jupyter

jupyter notebook

pip install ultralytics supervision

Для CUDA GPU
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Без CUDA GPU
pip install numpy torch torchvision

pip install tqdm scikit-learn matplotlib pandas scipy seaborn

py main.py

```

## Перевірна на натренованій моделі
```
from ultralytics import YOLO
import os

# Шлях до найкращих вагів, які зберегла модель
model_path = 'runs/detect/train/weights/best.pt'

# Завантаження натренованої моделі
model = YOLO(model_path)

# Запуск інференсу на власному зображенні (вмініть шлях на своє)
# Параметр show=True одразу відкрити вікно з результатами, а save=True збереже їх у папку runs/predict
results = model.predict(source='/Users/utereskovygmail.com/Downloads/2824384.jpg', show=True, save=True, device='mps')

print("Результати збережено в папці runs/predict")
 
```
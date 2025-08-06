# train_mnist_mlp.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import os
import random

# -----------------------------
# ① 데이터 로딩 및 전처리
# -----------------------------
(X_train, y_train), (X_val, y_val) = mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28 * 28)
X_val = X_val.reshape(-1, 28 * 28)

y_train_cat = to_categorical(y_train, 10)
y_val_cat = to_categorical(y_val, 10)

# -----------------------------
# ② 모델 정의 (MLP + Dropout)
# -----------------------------
inputs = Input(shape=(784,))
x = Dense(256, activation='relu')(inputs)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------
# ③ 모델 저장 디렉토리 및 파일명
# -----------------------------
now = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"mnist_model_{now}.h5")

# -----------------------------
# ④ 콜백 설정
# -----------------------------
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# -----------------------------
# ⑤ 학습
# -----------------------------
history = model.fit(X_train, y_train_cat,
                    epochs=30,
                    batch_size=128,
                    validation_data=(X_val, y_val_cat),
                    callbacks=[early_stopping, checkpoint],
                    verbose=1)

# -----------------------------
# ⑥ 정확도 시각화
# -----------------------------
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"모델 저장 완료: {model_path}")

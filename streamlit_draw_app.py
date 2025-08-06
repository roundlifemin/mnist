# ----------------------------------------
# ① MNIST 데이터 로딩 및 전처리
# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import os
import random

# MNIST 데이터 불러오기
(X_train, y_train), (X_val, y_val) = mnist.load_data()

# 정규화 및 reshape
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28 * 28)
X_val = X_val.reshape(-1, 28 * 28)

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_val_cat = to_categorical(y_val, 10)

# ----------------------------------------
# ② Dropout 포함 MLP 모델 정의
# ----------------------------------------
print("X_train.shape:", X_train.shape)  # (60000, 784)

inputs = Input(shape=(784,), name="input")
x = Dense(256, activation='relu', name="dense1")(inputs)
x = Dropout(0.5, name="dropout1")(x)
x = Dense(128, activation='relu', name="dense2")(x)
x = Dropout(0.3, name="dropout2")(x)
outputs = Dense(10, activation='softmax', name="output")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------------------------------
# ③ EarlyStopping 및 ModelCheckpoint 설정
# ----------------------------------------
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"mnist_model_{timestamp}.keras")

checkpoint = ModelCheckpoint(filepath=model_path,
                             monitor='val_accuracy',
                             save_best_only=True,
                             verbose=1)

# ----------------------------------------
# ④ 모델 학습
# ----------------------------------------
history = model.fit(X_train, y_train_cat,
                    epochs=30,
                    batch_size=128,
                    validation_data=(X_val, y_val_cat),
                    callbacks=[early_stopping, checkpoint],
                    verbose=1)

# ----------------------------------------
# ⑤ Accuracy 시각화
# ----------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('MNIST Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------
# ⑥ 저장된 모델 불러와서 테스트
# ----------------------------------------
loaded_model = load_model(model_path)
print(f"\n저장된 모델 로드 완료: {model_path}")

loss, acc = loaded_model.evaluate(X_val, y_val_cat, verbose=0)
print(f"불러온 모델의 검증 정확도: {acc * 100:.2f}%")

# ----------------------------------------
# ⑦ 특정 샘플 예측
# ----------------------------------------
sample_idx = random.randint(0, X_val.shape[0] - 1)
sample_input = X_val[sample_idx].reshape(1, -1)
true_label = y_val[sample_idx]

# 예측 수행
pred_probs = loaded_model.predict(sample_input)[0]
pred_class = np.argmax(pred_probs)

# 결과 출력
print("특정 샘플 예측")
print(f"샘플 인덱스: {sample_idx}")
print(f"실제 레이블 (정답): {true_label}")
print(f"예측 확률 분포: {np.round(pred_probs, 4)}")
print(f"예측 결과: {pred_class}")

# 시각화
plt.imshow(X_val[sample_idx].reshape(28, 28), cmap='gray')
plt.title(f"실제: {true_label}, 예측: {pred_class}")
plt.axis("off")
plt.show()

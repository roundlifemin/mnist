import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os

# ---------------------------
# 모델 로드
# ---------------------------
MODEL_DIR = "saved_models"
def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

latest_model_path = get_latest_model()
model = tf.keras.models.load_model(latest_model_path) if latest_model_path else None

# ---------------------------
# 앱 UI
# ---------------------------
st.title("숫자 그리기 - MNIST 예측기")
st.markdown("아래에 숫자를 **직접 그려보세요** (0~9)")

# ---------------------------
# 캔버스 구성
# ---------------------------
canvas_result = st_canvas(
    fill_color="#000000",       # 내부 채움색
    stroke_width=12,            # 선 두께
    stroke_color="#FFFFFF",     # 선 색 (흰색)
    background_color="#000000", # 배경 (검정색)
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------------------
# 예측 버튼
# ---------------------------
if st.button("예측 실행") and canvas_result.image_data is not None and model:
    # 이미지 가져오기
    img = canvas_result.image_data
    img = Image.fromarray((img[:, :, 0]).astype('uint8'))  # 흑백 채널만 사용

    # 28x28로 축소, 반전 및 정규화
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    # 예측
    pred = model.predict(img)
    pred_class = np.argmax(pred)

    # 결과 출력
    st.subheader(f"예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])

elif not model:
    st.warning("모델이 없습니다. 먼저 학습하여 저장하세요.")

# streamlit_draw_app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os
from scipy.ndimage import center_of_mass, shift

# ---------------------------
# 모델 로드 (.h5 형식)
# ---------------------------
MODEL_DIR = "saved_models"

def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

latest_model_path = get_latest_model()
model = tf.keras.models.load_model(latest_model_path) if latest_model_path else None

# ---------------------------
# UI 구성
# ---------------------------
st.title("숫자 그리기 - MNIST 예측기")
st.markdown("검정 배경에 흰색으로 **숫자 (0~9)** 를 그려보세요.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------------------
# 예측
# ---------------------------
if st.button("예측 실행") and canvas_result.image_data is not None and model:
    img = canvas_result.image_data
    img = Image.fromarray((img[:, :, 0]).astype('uint8'))  # 흑백으로 추출
    img = img.resize((28, 28))
    img = ImageOps.invert(img)  # 배경 반전

    # 무게중심 정렬
    img_arr = np.array(img)
    img_arr[img_arr < 100] = 0
    img_arr[img_arr >= 100] = 255
    cy, cx = center_of_mass(img_arr)
    shift_y = int(img_arr.shape[0] / 2 - cy)
    shift_x = int(img_arr.shape[1] / 2 - cx)
    img_arr = shift(img_arr, shift=(shift_y, shift_x), mode='constant', cval=0)

    # 정규화 및 reshape
    img_arr = img_arr.astype('float32') / 255.0
    img_arr = img_arr.reshape(1, 784)

    # 예측
    pred = model.predict(img_arr)
    pred_class = np.argmax(pred)

    st.subheader(f"예측 결과: **{pred_class}**")
    st.bar_chart(pred[0])

elif not model:
    st.warning("모델을 먼저 학습하고 saved_models 폴더에 .h5로 저장하세요.")

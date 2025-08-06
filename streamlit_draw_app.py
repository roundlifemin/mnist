import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os
from scipy.ndimage import center_of_mass, shift

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
# 단, TensorFlow 2.13에서는 safe_mode=False를 명시하지 않으면 복원 과정에서 일부 레이어가 누락되어 예외가 발생할 수 있다.
model = tf.keras.models.load_model(latest_model_path , safe_mode=False) if latest_model_path else None

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
# 예측 버튼
if st.button("예측 실행") and canvas_result.image_data is not None and model:
    img = canvas_result.image_data

    # 흑백 채널만 사용
    img = Image.fromarray((img[:, :, 0]).astype('uint8'))

    # 크기 조정 및 색 반전
    img = img.resize((28, 28))
    img = ImageOps.invert(img)

    # 중심 이동 (중요)
    img_arr = np.array(img)

    # 픽셀이 너무 연하면 threshold 처리 (흰색이 숫자로 간주되도록)
    img_arr[img_arr < 100] = 0
    img_arr[img_arr >= 100] = 255

    # 중심 맞춤: 이미지 무게중심을 계산해 중앙으로 이동
    from scipy.ndimage import center_of_mass, shift
    cy, cx = center_of_mass(img_arr)
    shift_y = int(img_arr.shape[0]/2 - cy)
    shift_x = int(img_arr.shape[1]/2 - cx)
    img_arr = shift(img_arr, shift=(shift_y, shift_x), mode='constant', cval=0)

    # 정규화 및 reshape
    img_arr = img_arr.astype("float32") / 255.0
    img_arr = img_arr.reshape(1, 784)  # MLP용

    # 예측
    pred = model.predict(img_arr)
    pred_class = np.argmax(pred)

    st.subheader(f"예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])


elif not model:
    st.warning("모델이 없습니다. 먼저 학습하여 저장하세요.")

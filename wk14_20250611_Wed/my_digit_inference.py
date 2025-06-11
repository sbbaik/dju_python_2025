import numpy as np
import cv2   # pip install opencv-python opencv-python-headless
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. 모델 불러오기
model = load_model('mnist_cnn_exp0_model.h5')

# 2. 이미지 불러오기 (그림판에서 저장한 파일 경로 사용)
img = cv2.imread('my_digit.png', cv2.IMREAD_GRAYSCALE)

# 3. 이미지 전처리
# - 그림판은 보통 하얀 배경에 검은 글씨 → MNIST는 반대 (검정 배경, 흰 숫자)
# - 크기를 28x28로 맞추고, 반전시킴
img = cv2.resize(img, (28, 28))
img = cv2.bitwise_not(img)  # 흑백 반전
img = img.astype('float32') / 255.0  # 정규화
img = np.expand_dims(img, axis=-1)  # 채널 차원 추가 (28, 28, 1)
img = np.expand_dims(img, axis=0)   # 배치 차원 추가 (1, 28, 28, 1)

# 4. 예측
pred = model.predict(img)
digit = np.argmax(pred)

print(f'모델이 인식한 숫자: {digit}')
